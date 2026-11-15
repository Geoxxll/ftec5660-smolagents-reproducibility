"""Main experiment: preserve task constraints in upstream max-step fallback.

The sole treatment is an appended constraint block in final_answer.pre_messages.
Original agent code, action loop, tools and step budget are unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from datetime import datetime,timezone
from pathlib import Path
from dotenv import load_dotenv
from smolagents import ToolCallingAgent
from experiment import FIXTURES,TASKS,build_tools,LoggedModel
from fallback_evaluation import assess,summarize_main

ROOT=Path(__file__).resolve().parent
TASK_CONSTRAINTS='''Task constraints:
Use only invoice amounts, exchange rates and fees retrieved from tools; do not invent missing data.
If currency is HKD, use rate 1; otherwise use the retrieved HKD-per-unit exchange rate.
If fee_plan is NONE, there is no fee; otherwise apply the retrieved fee policy.
Total HKD = amount * rate * (1 + basis_points / 10000) + fixed_hkd.
Round ONLY the final total to 2 decimal places, half up.
The final answer must be a single JSON object with exactly one field: {"amount_hkd": "123.45"}.
Return no prose, Markdown, or code fences.'''
NORMAL_INSTRUCTIONS=TASK_CONSTRAINTS+'''
Use the available tools to obtain invoice data. Independent FX and fee lookups may be batched.
When tools are available, submit this JSON object as the answer argument of final_answer.'''


def build_agent(model,events,condition):
    if condition not in {'original','preserved'}:raise ValueError('Unknown main-study condition')
    agent=ToolCallingAgent(tools=build_tools(events),model=model,instructions=NORMAL_INSTRUCTIONS,
                          max_steps=2,verbosity_level=-1,return_full_result=True)
    # Isolate nested dictionaries so one run cannot contaminate the control.
    agent.prompt_templates=deepcopy(agent.prompt_templates)
    if condition=='preserved':
        agent.prompt_templates['final_answer']['pre_messages']+='\n\n'+TASK_CONSTRAINTS
    return agent


class RequestLoggedModel(LoggedModel):
    """Capture sanitized message content, never client config or credentials."""
    def generate(self,messages,**kwargs):
        payload=[json.loads(m.model_dump_json()) for m in messages]
        try:return super().generate(messages,**kwargs)
        finally:self.calls[-1]['request_messages']=payload


def run_one(task,condition,trial,model_id,key):
    events=[]
    model=RequestLoggedModel(model_id=model_id,api_base='https://api.deepseek.com',api_key=key,
        temperature=0.2,top_p=1.0,max_tokens=700,client_kwargs={'timeout':45,'max_retries':0},
        extra_body={'thinking':{'type':'disabled'}})
    agent=build_agent(model,events,condition)
    started=time.perf_counter();output=None;state='error';messages=[];error_type=None
    try:
        result=agent.run(f'Calculate the total settlement amount in HKD for invoice {task["id"]}.')
        output,state,messages=result.output,result.state,result.messages
    except Exception as exc:error_type=type(exc).__name__
    invoice=FIXTURES['invoices'][task['id']]
    correct_args={'read_invoice':{'invoice_id':task['id']},'get_fx_rate':{'currency':invoice['currency']},
                  'get_fee_policy':{'fee_plan':invoice['fee_plan']}}
    observed={e['name'] for e in events if e['arguments']==correct_args[e['name']]}
    provenance=set(task['required_tools']).issubset(observed)
    return dict(task_id=task['id'],family=task['family'],condition=condition,trial=trial,
        max_steps=2,expected=task['expected'],output=output,state=state,error_type=error_type,
        seconds=time.perf_counter()-started,api_calls=len(model.calls),
        total_tokens=sum(c.get('input_tokens',0)+c.get('output_tokens',0) for c in model.calls),
        **assess(output,task['expected'],provenance,state),tool_events=events,
        model_calls=model.calls,memory=messages)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',default='results/fallback')
    parser.add_argument('--env-file')
    args=parser.parse_args()
    if args.trials<1:parser.error('Trials must be positive')
    load_dotenv(args.env_file or ROOT/'.env',override=False)
    key=os.environ.get('DEEPSEEK_API_KEY')
    if not key:parser.error('Set DEEPSEEK_API_KEY in an environment file; never in arguments.')
    out=Path(args.output)
    if (out/'runs.jsonl').exists():parser.error('Evidence exists; use a new output directory.')
    out.mkdir(parents=True,exist_ok=True)
    selected=[TASKS[2],TASKS[5]] if args.smoke else list(TASKS)
    sources=['fallback_experiment.py','fallback_evaluation.py','experiment.py','evaluation.py','data/fixtures.json','data/tasks.json']
    meta=dict(study='fallback_constraint_preservation',started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),platform=platform.platform(),provider='DeepSeek',
        base_url='https://api.deepseek.com',model_id=os.environ.get('DEEPSEEK_MODEL','deepseek-flash'),
        temperature=0.2,top_p=1.0,max_tokens=700,thinking='disabled',api_seed=None,
        trials=1 if args.smoke else args.trials,task_ids=[t['id'] for t in selected],
        task_order_seed=5660,condition_max_steps={'original':2,'preserved':2},
        treatment='append exact TASK_CONSTRAINTS to final_answer.pre_messages only',
        task_constraints=TASK_CONSTRAINTS,normal_instructions=NORMAL_INSTRUCTIONS,
        packages={p:importlib.metadata.version(p) for p in ['smolagents','openai','huggingface-hub']},
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    for filename in sources:
        snapshot=out/'source'/filename;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((ROOT/filename).read_bytes())
    rows=[]
    for trial in range(1,meta['trials']+1):
        tasks=list(selected);random.Random(5660+trial).shuffle(tasks)
        for i,task in enumerate(tasks):
            conditions=['original','preserved'] if (trial+i)%2 else ['preserved','original']
            for condition in conditions:
                row=run_one(task,condition,trial,meta['model_id'],key);rows.append(row)
                with (out/'runs.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
                print(f'{trial} {task["id"]} {condition}: json={row["json_valid"]} numeric={row["numeric_correct"]} structured={row["structured_correct"]} state={row["state"]}',flush=True)
                (out/'summary.json').write_text(json.dumps(summarize_main(rows),indent=2))
    print(json.dumps(summarize_main(rows),indent=2))

if __name__=='__main__':main()
