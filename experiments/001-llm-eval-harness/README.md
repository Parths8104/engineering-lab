# 001 — LLM Eval Harness

A minimal evaluation framework for testing LLM outputs across three dimensions: **accuracy**, **groundedness**, and **latency**.

## The problem

You can't `assert response == expected` on an LLM — the same prompt produces two different completions. So how do you write tests that don't flake but still catch regressions when you change a prompt?

## The approach

Three signals per test case:

1. **Accuracy** — embedding cosine similarity against a reference answer (threshold: > 0.85)
2. **Groundedness** — does the answer rely only on facts in the provided context? (LLM-as-judge)
3. **Latency** — wall-clock per call, useful for catching regressions when you swap models or add tools

Output is a structured JSON report. Commit the report alongside prompt changes to see regressions in the diff.

## Files

| File | What it does |
|---|---|
| `harness.py` | Runs all test cases in `evals/`, writes `report.json` |
| `judges.py` | LLM-as-judge for groundedness |
| `evals/example.json` | A sample test case |

## Run it

```bash
pip install openai
export OPENAI_API_KEY=sk-...
python harness.py
```

On Windows PowerShell:

```powershell
pip install openai
$env:OPENAI_API_KEY = "sk-..."
python harness.py
```

## What I learned

- `temperature=0` doesn't make outputs fully deterministic — embedding similarity is the right anchor for accuracy
- LLM-as-judge needs its own eval; judges drift across model versions, so pin them
- Measure latency separately from quality — they fail differently and need different fixes (retries vs prompt changes)
- Keeping eval cases in JSON makes diffs readable when prompts change

## Next steps

- Add p50/p95 latency aggregation across multiple runs per case
- Support batch mode for cheaper eval runs
- Add a CLI flag to compare two reports and surface regressions
