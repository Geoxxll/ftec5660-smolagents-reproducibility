# FTEC5660: task constraints in agent fallback

This experiment reproduces the max-step fallback path in Hugging Face smolagents1.20.0. The modification appends the existing task rules to the fallback system prompt. Both groups use `max_steps=2`; the normal prompt, tools, fixed invoice data and model settings are the same.

## Results

Six synthetic invoices were run three times per condition, giving36new live runs.

| Measure | Original fallback | Constraints preserved |
|---|---:|---:|
| Valid JSON |33.3%|100.0%|
| Valid JSON and correct amount |33.3%|66.7%|
| Correct amount, ignoring format |88.9%|66.7%|
| Mean model calls |2.67|2.67|
| Mean prompt + completion tokens |3877.8|3882.8|

The prompt change improved formatting, but amount accuracy fell. All six modified-condition fee calculations were wrong. The result does not establish general financial reliability.

## Install and reproduce offline

Measured environment: macOS arm64, Python3.14.7. The locked dependencies require Python>=3.11; other environments were not tested.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
python scripts/recompute_fallback.py results/fallback
python scripts/recompute.py results/live
```

Offline recomputation checks the run matrix, frozen source hashes, actual treatment messages, task-specific tool evidence and raw answers. It makes no API calls.

## Run new live trials

Copy `.env.example` to `.env` and set your own `DEEPSEEK_API_KEY`. Never commit the filled environment file.

```sh
python fallback_experiment.py --smoke --output results/new-smoke
python fallback_experiment.py --trials 3 --output results/new-fallback
python scripts/recompute_fallback.py results/new-fallback
```

The model is `deepseek-flash` at `https://api.deepseek.com`, using temperature0.2, top_p1, max_tokens700, thinking disabled, no SDK retries and no API seed. Every run starts with a fresh agent. Local seed5660+trial shuffles task order; condition order alternates. Existing evidence directories cannot be overwritten. Reruns incur API usage and the hosted alias can change over time.

## Change under test

The original agent loop is used without editing its source. The only condition-specific configuration is:

```python
if condition == 'preserved':
    agent.prompt_templates['final_answer']['pre_messages'] += '\n\n' + TASK_CONSTRAINTS
```

`TASK_CONSTRAINTS` also appears in both groups' normal instructions. Model request logs retain the exact messages so the treatment can be checked.

## Files and scoring

- `fallback_experiment.py`, `fallback_evaluation.py`:main comparison and metrics.
- `experiment.py`, `evaluation.py`:preliminary step-budget comparison.
- `data/`:fixed synthetic invoices, exchange rates, fee policies and independently checked expected values.
- `results/fallback/`:36main runs, requests/responses, tool events, frozen source and final summaries.
- `results/fallback-smoke/`:4prechecks, excluded from main statistics.
- `results/live/`:36earlier discovery runs, not reused as a control.
- `tests/`:scoring, prompt-boundary, isolation, fixture and offline-rebuild checks.

JSON validity requires one finite numeric `amount_hkd` field. Structured accuracy also requires the correct cent-rounded amount. Numeric-only scoring extracts a declared final amount without passing the expected value to the extractor. Fallback status and normal completion are reported separately. API and token counts include the extra fallback model request.

A post-run audit found three secondary numeric-extraction bugs: invoice IDs in headers, bullet-prefixed totals and parenthetical rates. These were corrected using failing regression tests. Primary JSON/structured scores, raw model outputs and ground truth did not change. The original run-time evaluator and score fields remain in `source/` and `runs.jsonl`. `analysis.json` and `analysis/` record the corrected evaluator; `runs.csv`, `numeric-audit.json` and `summary.json` contain final secondary scores.

## Upstream source

- [smolagents v1.20.0](https://github.com/huggingface/smolagents/tree/v1.20.0)
- [Agent API](https://huggingface.co/docs/smolagents/v1.20.0/en/reference/agents)
- Commit:`808e0958701b3a5b39efe7e48355bb13c19cc9dd`. The installed agents.py matched this source byte for byte.

This reproduces the documented tool loop and fallback behavior on a small custom task set. It does not reproduce CodeAgent or GAIA benchmark scores. smolagents is an external Apache-2.0 dependency. The course handout, student recording and report are submitted separately and are not included in this source repository.
