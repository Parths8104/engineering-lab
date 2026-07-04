# Evaluating LLM Outputs

## Why exact-match doesn't work

Traditional software tests assert deterministic equality: `assert func(x) == expected`. LLMs break this contract — the same prompt with the same seed can produce meaningfully different outputs depending on the model version, tokenizer nondeterminism, and floating-point ordering during inference. Tests written against exact strings become flaky within days.

## Similarity-based scoring

The standard workaround is embedding-based similarity. Embed both the generated answer and a reference answer, compute cosine similarity, and pass the case when similarity clears a threshold (typically 0.75–0.85). This is tolerant of phrasing differences that don't change meaning.

## Groundedness

For RAG systems, similarity isn't enough — an answer can be phrased correctly but factually invented. Groundedness measures whether every claim in the answer is supported by the retrieved context. LLM-as-judge is the common implementation: prompt a strong model to check the answer against the passages and return YES/NO.

## Citation coverage

Fraction of factual sentences in an answer that contain at least one citation. Sentences without citations can't be verified after the fact, so low coverage is a red flag even when similarity and groundedness look fine.

## The eval-driven workflow

Every prompt change ships alongside an updated eval report. If pass-rate drops or mean similarity regresses, the change is rejected before merge — the same pattern as unit tests gating a merge.
