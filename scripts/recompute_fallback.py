"""Rebuild main-study metrics from raw messages/tool evidence, without API access."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fallback_evaluation import assess,summarize_main

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='results/fallback')
a=p.parse_args();directory=Path(a.directory)
meta=json.loads((directory/'metadata.json').read_text())
rows=[json.loads(s) for s in (directory/'runs.jsonl').read_text().splitlines()]
tasks={t['id']:t for t in json.loads((ROOT/'data/tasks.json').read_text())}
fixtures=json.loads((ROOT/'data/fixtures.json').read_text())
keys=[(r['task_id'],r['condition'],r['trial']) for r in rows]
matrix={(t,c,i) for t in meta['task_ids'] for c in ['original','preserved'] for i in range(1,meta['trials']+1)}
assert len(keys)==len(set(keys)) and set(keys)==matrix,'Main run matrix is incomplete, duplicated or unexpected'
for filename,digest in meta['sha256'].items():
    current=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()
    if filename=='fallback_evaluation.py' and current!=digest:
        correction=json.loads((directory/'analysis.json').read_text())
        assert correction['frozen_evaluator_sha256']==digest
        assert correction['analysis_evaluator_sha256']==current,'Analysis evaluator drift'
    else:
        assert current==digest,f'Frozen source/data drift: {filename}'
    snapshot=directory/'source'/filename
    if snapshot.exists():assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest,f'Snapshot drift: {filename}'
for row in rows:
    task=tasks[row['task_id']];invoice=fixtures['invoices'][row['task_id']]
    assert row['max_steps']==2,'Main study requires fixed two-step budget'
    args={'read_invoice':{'invoice_id':task['id']},'get_fx_rate':{'currency':invoice['currency']},'get_fee_policy':{'fee_plan':invoice['fee_plan']}}
    observed={e['name'] for e in row['tool_events'] if e['name'] in args and e['arguments']==args[e['name']]}
    row.update(assess(row['output'],task['expected'],set(task['required_tools']).issubset(observed),row['state']))
    row['expected']=task['expected'];row['family']=task['family']
    row['api_calls']=len(row['model_calls'])
    row['total_tokens']=sum(c.get('input_tokens',0)+c.get('output_tokens',0) for c in row['model_calls'])
    # Verify treatment at the actual request boundary, not merely a condition label.
    for call in row['model_calls']:
        if not call['tool_schema_present']:
            system=call['request_messages'][0]['content'][0]['text']
            assert (meta['task_constraints'] in system)==(row['condition']=='preserved'),'Fallback treatment mismatch'
summary=summarize_main(rows)
(directory/'summary.json').write_text(json.dumps(summary,indent=2))
by_family={name:summarize_main([r for r in rows if r['family']==name]) for name in sorted({r['family'] for r in rows})}
(directory/'by_family.json').write_text(json.dumps(by_family,indent=2))
columns=['task_id','family','condition','trial','max_steps','expected','output','json_valid','extracted_amount','numeric_parse_method','numeric_correct','structured_correct','evidence_supported_correct','state','api_calls','total_tokens','seconds']
with (directory/'runs.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
audit=[{k:r[k] for k in ['task_id','condition','trial','output','extracted_amount','numeric_parse_method','expected','numeric_correct']} for r in rows]
(directory/'numeric-audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2))
