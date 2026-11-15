import json
import unittest
from decimal import Decimal as D, ROUND_HALF_UP
from pathlib import Path

class FixtureTests(unittest.TestCase):
    def test_independent_decimal_oracle_matches_every_frozen_answer(self):
        root=Path(__file__).resolve().parents[1]
        data=json.loads((root/'data/fixtures.json').read_text())
        for task in json.loads((root/'data/tasks.json').read_text()):
            invoice=data['invoices'][task['id']]
            rate=D('1') if invoice['currency']=='HKD' else D(data['rates'][invoice['currency']])
            fee=data['fees'].get(invoice['fee_plan'], {'basis_points':0,'fixed_hkd':'0'})
            total=(D(invoice['amount'])*rate*(1+D(fee['basis_points'])/10000)+D(fee['fixed_hkd'])).quantize(D('.01'),rounding=ROUND_HALF_UP)
            self.assertEqual(str(total),task['expected'],task['id'])
