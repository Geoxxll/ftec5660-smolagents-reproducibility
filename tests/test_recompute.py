"""The offline rebuild must detect mislabeled runs and disregard stale score flags."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class RecomputeTests(unittest.TestCase):
    def prepare(self, directory):
        source=ROOT/'results/live'
        rows=[json.loads(s) for s in (source/'runs.jsonl').read_text().splitlines()]
        (directory/'metadata.json').write_text((source/'metadata.json').read_text())
        return rows

    def run_recompute(self,directory,rows):
        (directory/'runs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return subprocess.run([sys.executable,str(ROOT/'scripts/recompute.py'),str(directory)],capture_output=True,text=True)

    def test_equal_count_with_wrong_task_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);rows=self.prepare(path)
            rows[0]['task_id']='UNKNOWN-INVOICE'
            result=self.run_recompute(path,rows)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('run matrix',result.stderr.lower())

    def test_stale_score_flags_are_rebuilt_from_raw_answers(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);rows=self.prepare(path)
            for row in rows:
                row['answer_correct']=False
                row['verified_success']=False
            result=self.run_recompute(path,rows)
            self.assertEqual(result.returncode,0,result.stderr)
            summary=json.loads((path/'summary.json').read_text())
            self.assertAlmostEqual(summary['baseline']['answer_accuracy'],2/3)
            self.assertAlmostEqual(summary['limited']['answer_accuracy'],1/3)
