# 002 — RAG Retrieval Comparison

Comparing three retrieval strategies on the same corpus and eval set to
answer: **does hybrid retrieval actually beat dense-only on the queries
we care about, and by how much?**

## The problem

Anchor's [ADR-0001](../../../anchor/docs/decisions/0001-hybrid-retrieval.md)
argues for hybrid retrieval (dense + BM25 fused via Reciprocal Rank
Fusion) over dense-only or weighted-sum approaches. The argument is
theoretical: dense embeddings miss keyword-exact queries, BM25 misses
semantic ones, RRF combines them without magic-number tuning.

This experiment runs actual numbers on that claim.

## The setup

A small technical-docs corpus (5 markdown files, ~15 chunks) and 10
questions spanning three query types:

- **Semantic queries** — paraphrased, no shared vocabulary with the source
  (dense should win)
- **Keyword queries** — exact function/class/error names
  (BM25 should win)
- **Mixed queries** — one keyword anchor plus semantic phrasing
  (hybrid should win)

Each question has a reference answer. For each retrieval strategy, we
measure **recall@k** — did the top-k retrieved chunks contain the chunk
that actually answers the question?

## The three strategies

1. **Dense-only** — OpenAI `text-embedding-3-small`, cosine similarity, top-k
2. **BM25-only** — sparse token-frequency ranking via `rank-bm25`
3. **Hybrid (RRF)** — both of the above, fused via Reciprocal Rank Fusion
   with k=60

## Files

| File | Purpose |
|---|---|
| `run.py` | Runs all three strategies against all questions, writes `report.json` |
| `data/*.md` | Technical documentation used as the corpus |
| `evals/questions.json` | 10 questions with expected-answer chunks |
| `report.json` | Latest results (regenerated on each run) |

## Run it

```bash
pip install openai chromadb rank-bm25 tiktoken numpy
export OPENAI_API_KEY=sk-...
python run.py
```

Expected output (rough shape — real numbers will vary):

```
============================================================
RETRIEVAL COMPARISON — recall@3
============================================================
strategy        semantic     keyword      mixed        overall
---------------------------------------------------------
dense-only        0.80        0.20        0.50        0.50
bm25-only         0.20        1.00        0.50        0.57
hybrid-rrf        0.80        1.00        0.75        0.85
============================================================
```

## What I learned

*(fill in after running — this is the whole point of the experiment)*

- Which query types show the biggest hybrid lift?
- Are there queries where dense-only actually beats hybrid?
- What does the failure mode look like when hybrid loses?

## Next steps

- Try weighted-sum fusion with tuned α to see if it beats RRF on this corpus
- Add cross-encoder reranking as a fourth strategy
- Grow the corpus to 100+ chunks and see if the gaps hold up
