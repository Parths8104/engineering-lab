# engineering-lab

A working notebook of experiments, small tools, and writeups from my path as a software engineer building **LLM systems** and **backend infrastructure**.

Not a portfolio. A workshop.

## Why this exists

Most engineering learning lives between the gaps — the half-broken script, the prototype that taught a lesson, the writeup that crystallized an idea. This repo is where those live.

Small commits, often. Some experiments are 30 lines. Some take a weekend. The goal is reps, not polish.

## Structure

```
engineering-lab/
├── experiments/    self-contained micro-projects, each with its own README
├── notes/          longer writeups — systems design, paper summaries, post-mortems
├── snippets/       reusable patterns I keep reaching for
└── reading/        books & papers I've worked through, with takeaways
```

## Currently exploring

- Evaluation harnesses for non-deterministic LLM outputs
- Retrieval strategies beyond naive vector search
- Agentic workflows with tool calling and graceful recovery
- Backend patterns I wish I knew sooner — idempotency, retries, rate limits

## Index

| # | Experiment | What it explores | Stack |
|---|-----------|------------------|-------|
| 001 | [llm-eval-harness](./experiments/001-llm-eval-harness) | Structured testing for non-deterministic LLM outputs | Python, OpenAI |

| 002 | [rag-retrieval-comparison](./experiments/002-rag-retrieval-comparison) | Dense vs BM25 vs hybrid RRF, measured on recall@k | Python, OpenAI, ChromaDB, rank-bm25 |

---

📫 psojitraswe@gmail.com &nbsp;·&nbsp; 🌐 [parthsojitra.dev](https://parthsojitra.dev)
