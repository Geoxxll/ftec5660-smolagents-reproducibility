"""Offline evaluation: no LLM judge; all failures stay in the denominator."""
import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from statistics import mean, stdev


def parse_amount(output):
    try:
        if isinstance(output, str):
            output = json.loads(output)
        value = output['amount_hkd']
        if isinstance(value, bool):
            return None
        number = Decimal(str(value))
        if not number.is_finite():
            return None
        return number.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except (ValueError, TypeError, KeyError, InvalidOperation):
        return None


def score(output, expected, required_tools, observed_tools, state):
    correct = parse_amount(output) == Decimal(expected)
    provenance = set(required_tools).issubset(observed_tools)
    return {'answer_correct': correct, 'required_tools_observed': provenance,
            'verified_success': correct and provenance and state == 'success'}


def summarize(rows):
    groups = {}
    for condition in sorted({r['condition'] for r in rows}):
        part = [r for r in rows if r['condition'] == condition]
        trial_rates = [mean(r['answer_correct'] for r in part if r['trial'] == t)
                       for t in sorted({r['trial'] for r in part})]
        groups[condition] = dict(
            n=len(part), answer_accuracy=mean(r['answer_correct'] for r in part),
            verified_success_rate=mean(r['verified_success'] for r in part),
            fallback_rate=mean(r['state'] == 'max_steps_error' for r in part),
            error_count=sum(r['state'] == 'error' for r in part),
            mean_api_calls=mean(r['api_calls'] for r in part),
            mean_total_tokens=mean(r['total_tokens'] for r in part),
            mean_seconds=mean(r['seconds'] for r in part),
            trial_accuracies=trial_rates,
            trial_accuracy_sd=stdev(trial_rates) if len(trial_rates)>1 else 0,
            trial_accuracy_variance=stdev(trial_rates)**2 if len(trial_rates)>1 else 0)
    return groups
