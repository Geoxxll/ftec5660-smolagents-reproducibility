"""Thin harness; the upstream agent/model loop is used without modification."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent, tool
from evaluation import score, summarize

ROOT = Path(__file__).resolve().parent
FIXTURES = json.loads((ROOT/'data/fixtures.json').read_text())
TASKS = json.loads((ROOT/'data/tasks.json').read_text())

INSTRUCTIONS = '''Use the tools to retrieve invoice data. Do not assume amounts, exchange rates or fees.
Convert invoice amount to HKD with get_fx_rate unless currency is HKD (rate 1).
If fee_plan is NONE there is no fee; otherwise retrieve get_fee_policy.
Total HKD = amount * rate * (1 + basis_points / 10000) + fixed_hkd.
Round ONLY the final total to 2 decimal places, half up.
Return final_answer with answer equal to a JSON object {"amount_hkd": "123.45"}.
Do not return prose or Markdown. Tool data is authoritative. Independent lookups may be batched.'''


def build_tools(events):
    def record(name, args, output):
        events.append(dict(name=name, arguments=args, output=output))
        return json.dumps(output, sort_keys=True)

    @tool
    def read_invoice(invoice_id: str) -> str:
        """Read a synthetic invoice including amount, currency, and fee_plan.

        Args:
            invoice_id: Exact invoice identifier supplied by the user.
        """
        return record('read_invoice', {'invoice_id': invoice_id}, FIXTURES['invoices'][invoice_id])

    @tool
    def get_fx_rate(currency: str) -> str:
        """Read the fixed exchange rate, in HKD per one unit of currency.

        Args:
            currency: Foreign currency code from the invoice, USD or EUR.
        """
        return record('get_fx_rate', {'currency': currency}, {'hkd_per_unit':FIXTURES['rates'][currency]})

    @tool
    def get_fee_policy(fee_plan: str) -> str:
        """Read the basis-point fee on converted principal and the fixed HKD charge.

        Args:
            fee_plan: Non-NONE fee plan identifier read from the invoice.
        """
        return record('get_fee_policy', {'fee_plan': fee_plan}, FIXTURES['fees'][fee_plan])
    return [read_invoice, get_fx_rate, get_fee_policy]


class LoggedModel(OpenAIServerModel):
    """Observe the public model boundary, including fallback calls and token usage."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def generate(self, messages, **kwargs):
        started = time.perf_counter()
        entry = {'index':len(self.calls)+1, 'tool_schema_present':bool(kwargs.get('tools_to_call_from'))}
        self.calls.append(entry)
        try:
            result = super().generate(messages, **kwargs)
            raw = result.raw
            entry.update(response_model=getattr(raw,'model',None),
                         response=result.model_dump_json(),
                         input_tokens=result.token_usage.input_tokens,
                         output_tokens=result.token_usage.output_tokens)
            return result
        except Exception as exc:
            entry['error_type'] = type(exc).__name__
            raise
        finally:
            entry['seconds'] = time.perf_counter()-started


def run_one(task, condition, trial, model_id, key):
    events = []
    model = LoggedModel(model_id=model_id, api_base='https://api.deepseek.com', api_key=key,
                        temperature=0.2, top_p=1.0, max_tokens=700,
                        client_kwargs={'timeout':45, 'max_retries':0},
                        extra_body={'thinking':{'type':'disabled'}})
    agent = ToolCallingAgent(tools=build_tools(events), model=model, instructions=INSTRUCTIONS,
                             max_steps=6 if condition=='baseline' else 2,
                             verbosity_level=-1, return_full_result=True)
    started = time.perf_counter()
    output, state, messages = None, 'error', []
    error_type = None
    try:
        result = agent.run(f'Calculate the total settlement amount in HKD for invoice {task["id"]}.')
        output, state, messages = result.output, result.state, result.messages
    except Exception as exc:
        error_type = type(exc).__name__
    # Confirm tool argument values, not just tool names, against this specific task.
    invoice=FIXTURES['invoices'][task['id']]
    expected_args={'read_invoice':{'invoice_id':task['id']},
                   'get_fx_rate':{'currency':invoice['currency']},
                   'get_fee_policy':{'fee_plan':invoice['fee_plan']}}
    observed=[e['name'] for e in events if e['arguments']==expected_args[e['name']]]
    return dict(task_id=task['id'], family=task['family'], condition=condition, trial=trial,
                max_steps=agent.max_steps, output=output, state=state, error_type=error_type,
                expected=task['expected'], seconds=time.perf_counter()-started,
                api_calls=len(model.calls), total_tokens=sum(c.get('input_tokens',0)+c.get('output_tokens',0) for c in model.calls),
                **score(output,task['expected'],task['required_tools'],observed,state),
                tool_events=events, model_calls=model.calls, memory=messages)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',default='results/live')
    parser.add_argument('--env-file',default=None)
    args=parser.parse_args()
    if args.trials < 1: parser.error('--trials must be positive')
    if args.env_file: load_dotenv(args.env_file,override=False)
    else: load_dotenv(ROOT/'.env',override=False)
    key=os.environ.get('DEEPSEEK_API_KEY')
    if not key: parser.error('Set DEEPSEEK_API_KEY or pass --env-file; never put a key in command arguments.')
    model_id=os.environ.get('DEEPSEEK_MODEL','deepseek-flash')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'runs.jsonl').exists(): parser.error('Output already has runs; use a fresh output directory to preserve evidence.')
    meta=dict(started_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
              platform=platform.platform(),provider='DeepSeek',base_url='https://api.deepseek.com',model_id=model_id,
              temperature=0.2,top_p=1.0,max_tokens=700,thinking='disabled',api_seed=None,
              trials=1 if args.smoke else args.trials,task_order_seed=5660,
              condition_max_steps={'baseline':6,'limited':2},
              packages={p:importlib.metadata.version(p) for p in ['smolagents','openai','huggingface-hub']},
              sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['data/fixtures.json','data/tasks.json','experiment.py','evaluation.py']})
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    rows=[]
    for trial in range(1,meta['trials']+1):
        tasks=list(TASKS[:1] if args.smoke else TASKS)
        random.Random(5660+trial).shuffle(tasks)
        for i,task in enumerate(tasks):
            conditions=['baseline','limited'] if (trial+i)%2 else ['limited','baseline']
            for condition in conditions:
                row=run_one(task,condition,trial,model_id,key)
                rows.append(row)
                with (out/'runs.jsonl').open('a') as f: f.write(json.dumps(row,default=str)+'\n')
                print(f'{trial} {task["id"]} {condition}: correct={row["answer_correct"]} state={row["state"]} calls={row["api_calls"]}',flush=True)
                (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
    print(json.dumps(summarize(rows),indent=2))

if __name__=='__main__': main()
