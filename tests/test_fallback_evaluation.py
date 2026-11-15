import unittest
from fallback_evaluation import assess, extract_numeric, summarize_main

class FallbackEvaluationTests(unittest.TestCase):
    def test_valid_json_wrong_amount_separates_format_and_arithmetic(self):
        r=assess('{"amount_hkd":"1884.51"}','1867.87',True,'max_steps_error')
        self.assertTrue(r['json_valid'])
        self.assertFalse(r['numeric_correct'])
        self.assertFalse(r['structured_correct'])

    def test_correct_prose_number_is_not_structured_success(self):
        r=assess('Total settlement amount: **1,867.87 HKD**','1867.87',True,'max_steps_error')
        self.assertFalse(r['json_valid'])
        self.assertTrue(r['numeric_correct'])
        self.assertFalse(r['structured_correct'])

    def test_repaired_fallback_can_be_evidence_supported(self):
        r=assess({'amount_hkd':'1867.87'},'1867.87',True,'max_steps_error')
        self.assertTrue(r['evidence_supported_correct'])
        self.assertFalse(r['normal_completion'])

    def test_conflicting_finals_do_not_pick_expected_number(self):
        number,method=extract_numeric('Answer: 10.00 HKD\nFinal answer: 12.00 HKD')
        self.assertIsNone(number)
        self.assertEqual(method,'ambiguous')

    def test_intermediate_amount_and_alternative_note_are_not_final(self):
        value,method=extract_numeric('Base: 1,150.80 HKD\nAnswer: approximately 1,167.59 HKD\nIf deducted instead, 1,134.02 HKD.')
        self.assertEqual(str(value),'1167.59')
        self.assertEqual(method,'labelled')

    def test_embedded_json_fence_is_numeric_but_not_valid_contract(self):
        r=assess('```json\n{"amount_hkd":"683.96"}\n```','683.96',True,'max_steps_error')
        self.assertTrue(r['numeric_correct'])
        self.assertFalse(r['json_valid'])

    def test_nonfinite_bool_and_extra_fields_do_not_pass_schema(self):
        for value in [{'amount_hkd':True},{'amount_hkd':'NaN'},{'amount_hkd':'Infinity'},{'amount_hkd':'10','extra':1}]:
            self.assertFalse(assess(value,'10.00',True,'success')['json_valid'])

    def test_aggregation_keeps_failures_and_separates_fallback_subgroup(self):
        rows=[dict(condition='original',trial=1,state='success',api_calls=2,total_tokens=10,seconds=1,**assess({'amount_hkd':'10'},'10',True,'success')),
              dict(condition='original',trial=1,state='max_steps_error',api_calls=3,total_tokens=20,seconds=2,**assess('Answer: 10.00 HKD','10',True,'max_steps_error'))]
        r=summarize_main(rows)['original']
        self.assertEqual(r['n'],2)
        self.assertEqual(r['structured_accuracy'],0.5)
        self.assertEqual(r['numeric_accuracy'],1)
        self.assertEqual(r['fallback_n'],1)
        self.assertEqual(r['fallback_structured_accuracy'],0)

class NumericExtractionRegressionTests(unittest.TestCase):
    def test_invoice_id_in_header_is_not_a_conflicting_final_amount(self):
        value,method=extract_numeric('Invoice INV-P02 - Total Settlement Amount (HKD)\nTotal settlement amount: 1,867.87 HKD\nAnswer: 1,867.87 HKD')
        self.assertEqual(str(value),'1867.87')
        self.assertEqual(method,'labelled')

    def test_bullet_total_is_a_final_label(self):
        value,method=extract_numeric('- **Total: 93.17 x 7.81 = 727.66 HKD** (approximately)')
        self.assertEqual(str(value),'727.66')

    def test_parenthetical_working_does_not_override_declared_final_amount(self):
        value,method=extract_numeric('Total settlement amount: 1,834.75 HKD (based on 219.87 EUR at 8.42 HKD/EUR)')
        self.assertEqual(str(value),'1834.75')
