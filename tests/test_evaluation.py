"""Independent scorer checks: catching false positives and wrong denominators."""
import unittest
from evaluation import parse_amount, score, summarize

class EvaluationTests(unittest.TestCase):
    def test_exact_json_and_half_up_rounding(self):
        self.assertEqual(str(parse_amount({'amount_hkd': '1.005'})), '1.01')
        self.assertEqual(str(parse_amount('{"amount_hkd": 727.66}')), '727.66')

    def test_prose_missing_and_nonfinite_are_not_answers(self):
        for value in ['Probably 727.66', {}, {'amount_hkd': True}, {'amount_hkd': 'NaN'}, {'amount_hkd': 'Infinity'}]:
            self.assertIsNone(parse_amount(value))

    def test_correct_number_without_required_evidence_is_not_verified(self):
        result = score({'amount_hkd': '727.66'}, '727.66', ['read_invoice', 'get_fx_rate'], ['read_invoice'], 'success')
        self.assertTrue(result['answer_correct'])
        self.assertFalse(result['verified_success'])

    def test_fallback_correctness_is_separate_from_normal_completion(self):
        result = score({'amount_hkd': '727.66'}, '727.66', ['read_invoice', 'get_fx_rate'], ['read_invoice','get_fx_rate'], 'max_steps_error')
        self.assertTrue(result['answer_correct'])
        self.assertFalse(result['verified_success'])

    def test_failed_runs_remain_in_denominator_and_trials_have_sample_sd(self):
        rows = [dict(condition='baseline', trial=1, answer_correct=True, verified_success=True, state='success', api_calls=3, total_tokens=100, seconds=2),
                dict(condition='baseline', trial=2, answer_correct=False, verified_success=False, state='error', api_calls=1, total_tokens=40, seconds=1)]
        result = summarize(rows)['baseline']
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['answer_accuracy'], 0.5)
        self.assertAlmostEqual(result['trial_accuracy_sd'], 0.70710678118)

if __name__ == '__main__': unittest.main()
