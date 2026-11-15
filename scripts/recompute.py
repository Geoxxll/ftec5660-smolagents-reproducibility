"""Recompute all published summary numbers from frozen raw runs, without API access."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation import score, summarize

p=argparse.ArgumentParser()
p.add_argument('directory',nargs='?',default='results/live')
a=p.parse_args()
root=Path(a.directory)
rows=[json.loads(s) for s in (root/'runs.jsonl').read_text().splitlines()]
keys=[(r['task_id'],r['condition'],r['trial']) for r in rows]
assert len(keys)==len(set(keys)), 'Duplicate run identifiers'
expected=json.loads((Path(__file__).resolve().parents[1]/'data/tasks.json').read_text())
meta=json.loads((root/'metadata.json').read_text())
matrix={(task['id'],condition,trial) for task in expected
        for condition in ['baseline','limited'] for trial in range(1,meta['trials']+1)}
assert set(keys)==matrix, 'Run matrix is incomplete or contains unexpected task/condition/trial IDs'
fixtures=json.loads((Path(__file__).resolve().parents[1]/'data/fixtures.json').read_text())
tasks={task['id']:task for task in expected}
for row in rows:
    task=tasks[row['task_id']]
    invoice=fixtures['invoices'][row['task_id']]
    args={'read_invoice':{'invoice_id':row['task_id']},
          'get_fx_rate':{'currency':invoice['currency']},
          'get_fee_policy':{'fee_plan':invoice['fee_plan']}}
    observed=[e['name'] for e in row['tool_events'] if e['name'] in args and e['arguments']==args[e['name']]]
    row.update(score(row['output'],task['expected'],task['required_tools'],observed,row['state']))
    row['expected']=task['expected']
    row['family']=task['family']
    row['api_calls']=len(row['model_calls'])
    row['total_tokens']=sum(c.get('input_tokens',0)+c.get('output_tokens',0) for c in row['model_calls'])
summary=summarize(rows)
(root/'summary.json').write_text(json.dumps(summary,indent=2))
by_family={family:summarize([r for r in rows if r['family']==family]) for family in ['local','fx','fee']}
(root/'by_family.json').write_text(json.dumps(by_family,indent=2))
columns=['task_id','family','condition','trial','max_steps','expected','output','answer_correct','required_tools_observed','verified_success','state','api_calls','total_tokens','seconds']
with (root/'runs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
