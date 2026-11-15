import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class MainRecomputeTests(unittest.TestCase):
    def prepare(self,path):
        source=ROOT/'results/fallback-smoke'
        meta=json.loads((source/'metadata.json').read_text())
        (path/'metadata.json').write_text(json.dumps(meta))
        if (source/'analysis.json').exists():
            (path/'analysis.json').write_text((source/'analysis.json').read_text())
        rows=[json.loads(s) for s in (source/'runs.jsonl').read_text().splitlines()]
        return rows

    def invoke(self,path,rows):
        (path/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return subprocess.run([sys.executable,str(ROOT/'scripts/recompute_fallback.py'),str(path)],capture_output=True,text=True)

    def test_rejects_wrong_condition_even_when_count_is_correct(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);rows=self.prepare(path);rows[0]['condition']='other'
            result=self.invoke(path,rows)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('matrix',result.stderr.lower())

    def test_rebuilds_schema_numeric_and_usage_from_raw_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);rows=self.prepare(path)
            for r in rows:
                r.update(json_valid=False,numeric_correct=False,structured_correct=False,total_tokens=0,api_calls=0)
            result=self.invoke(path,rows)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((path/'summary.json').read_text())
            self.assertEqual(summary['preserved']['json_valid_rate'],1)
            self.assertEqual(summary['preserved']['structured_accuracy'],0.5)
            self.assertEqual(summary['original']['json_valid_rate'],0)
            self.assertEqual(summary['original']['mean_api_calls'],3)
            self.assertGreater(summary['original']['mean_total_tokens'],100)
