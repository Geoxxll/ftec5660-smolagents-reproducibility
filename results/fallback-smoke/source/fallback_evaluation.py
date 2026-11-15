"""Predeclared main-study metrics. Numeric extraction never sees ground truth."""
import json
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from statistics import mean, stdev

NUMBER=r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
MONEY=re.compile(r'HKD\s*('+NUMBER+r')|('+NUMBER+r')\s*HKD',re.I)
LABEL=re.compile(r'\b(?:total\s+(?:settlement(?:\s+amount)?|amount|payable|HKD)|final\s+(?:answer|amount|total)|answer|settlement\s+total)\b|^\s*(?:\d+[.)]\s*)?total\s*[:=]',re.I)


def _number(value):
    if isinstance(value,bool) or not isinstance(value,(str,int,float,Decimal)):
        return None
    try:
        n=Decimal(str(value).replace(',',''))
        return n.quantize(Decimal('.01'),rounding=ROUND_HALF_UP) if n.is_finite() else None
    except (InvalidOperation,ValueError):
        return None


def _object(output):
    try:
        value=json.loads(output) if isinstance(output,str) else output
        return value if isinstance(value,dict) else None
    except (ValueError,TypeError):
        return None


def extract_numeric(output):
    """JSON first; otherwise consistent explicitly labelled final values, or bare HKD.

    Multiple different final labels are ambiguous and fail extraction. Intermediate
    amounts and unlabelled alternatives are ignored. No expected answer is an input.
    """
    obj=_object(output)
    if obj is not None and (n:=_number(obj.get('amount_hkd'))) is not None:
        return n,'json'
    if not isinstance(output,str):
        return None,'unparseable'
    text=output.replace('*','').replace('`','')
    candidates=[]
    for match in re.finditer(r'"amount_hkd"\s*:\s*"?('+NUMBER+r')"?',text):
        n=_number(match.group(1))
        if n is not None:candidates.append(n)
    for line in text.splitlines():
        if not LABEL.search(line):continue
        matches=list(MONEY.finditer(line))
        if matches:
            match=matches[-1];n=_number(match.group(1) or match.group(2))
        else:
            numbers=re.findall(NUMBER,line);n=_number(numbers[-1]) if numbers else None
        if n is not None:candidates.append(n)
    if candidates:
        return (candidates[0],'labelled') if len(set(candidates))==1 else (None,'ambiguous')
    bare=re.fullmatch(r'\s*(?:HKD\s*)?('+NUMBER+r')(?:\s*HKD)?\s*',text,re.I)
    if bare:return _number(bare.group(1)),'bare'
    return None,'unparseable'


def assess(output,expected,provenance,state):
    obj=_object(output)
    amount=_number(obj.get('amount_hkd')) if obj is not None else None
    valid=obj is not None and set(obj)=={'amount_hkd'} and amount is not None
    numeric,method=extract_numeric(output)
    correct=numeric is not None and numeric==Decimal(expected)
    structured=valid and amount==Decimal(expected)
    return dict(json_valid=valid,numeric_extracted=numeric is not None,
                extracted_amount=str(numeric) if numeric is not None else None,
                numeric_parse_method=method,numeric_correct=correct,
                structured_correct=structured,evidence_supported_correct=structured and provenance,
                required_tools_observed=provenance,normal_completion=state=='success')


def summarize_main(rows):
    groups={}
    for condition in sorted({r['condition'] for r in rows}):
        part=[r for r in rows if r['condition']==condition]
        fallback=[r for r in part if r['state']=='max_steps_error']
        trials=[mean(r['structured_correct'] for r in part if r['trial']==t)
                for t in sorted({r['trial'] for r in part})]
        rates={name:mean(r[field] for r in part) for name,field in {
            'json_valid_rate':'json_valid','numeric_extraction_rate':'numeric_extracted',
            'numeric_accuracy':'numeric_correct','structured_accuracy':'structured_correct',
            'evidence_supported_accuracy':'evidence_supported_correct',
            'normal_completion_rate':'normal_completion'}.items()}
        groups[condition]=dict(n=len(part),**rates,
            fallback_n=len(fallback),fallback_rate=len(fallback)/len(part),
            fallback_json_valid_rate=mean(r['json_valid'] for r in fallback) if fallback else None,
            fallback_structured_accuracy=mean(r['structured_correct'] for r in fallback) if fallback else None,
            fallback_numeric_accuracy=mean(r['numeric_correct'] for r in fallback) if fallback else None,
            unparseable_count=sum(not r['numeric_extracted'] for r in part),
            error_count=sum(r['state']=='error' for r in part),
            mean_api_calls=mean(r['api_calls'] for r in part),
            mean_total_tokens=mean(r['total_tokens'] for r in part),
            mean_seconds=mean(r['seconds'] for r in part),
            trial_structured_accuracies=trials,
            trial_structured_sd=stdev(trials) if len(trials)>1 else 0,
            trial_structured_variance=stdev(trials)**2 if len(trials)>1 else 0)
    return groups
