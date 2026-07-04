"""Run all three retrieval strategies against the eval questions.

Writes a JSON report with per-question and aggregated recall@k numbers
for dense-only, BM25-only, and hybrid RRF.
"""

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from openai import OpenAI
from rank_bm25 import BM25Okapi

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
EVAL_PATH = HERE / "evals" / "questions.json"
REPORT_PATH = HERE / "report.json"

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 200
CHUNK_OVERLAP = 40
TOP_K = 3
RRF_K = 60

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
encoder = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::c{self.chunk_index:03d}"


def chunk_text(doc_id: str, text: str) -> list[Chunk]:
    """Token-based sliding-window chunking."""
    tokens = encoder.encode(text)
    if not tokens:
        return []
    stride = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(tokens), stride)):
        window = tokens[start : start + CHUNK_TOKENS]
        chunks.append(Chunk(doc_id=doc_id, chunk_index=i, text=encoder.decode(window)))
        if start + CHUNK_TOKENS >= len(tokens):
            break
    return chunks


def load_corpus() -> list[Chunk]:
    """Chunk every markdown file in data/ into a flat list."""
    chunks: list[Chunk] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        chunks.extend(chunk_text(path.stem, path.read_text(encoding="utf-8")))
    return chunks


# ---------------------------------------------------------------------------
# Dense retrieval
# ---------------------------------------------------------------------------


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Batch-embed via OpenAI."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def dense_retrieve(
    query_emb: list[float],
    chunk_embs: list[list[float]],
    chunks: list[Chunk],
    k: int,
) -> list[str]:
    """Return top-k chunk_ids by cosine similarity."""
    scored = [(cosine(query_emb, ce), c.chunk_id) for c, ce in zip(chunks, chunk_embs)]
    scored.sort(reverse=True)
    return [cid for _, cid in scored[:k]]


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_retrieve(query: str, bm25: BM25Okapi, chunks: list[Chunk], k: int) -> list[str]:
    """Return top-k chunk_ids by BM25 score."""
    scores = bm25.get_scores(tokenize(query))
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    # Drop zero-score matches so we don't accidentally count noise
    top = [(i, s) for i, s in indexed[:k] if s > 0]
    return [chunks[i].chunk_id for i, _ in top]


# ---------------------------------------------------------------------------
# Hybrid retrieval (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------


def hybrid_retrieve(dense_ids: list[str], bm25_ids: list[str], k: int) -> list[str]:
    """Fuse two ranked lists via RRF and return top-k chunk_ids."""
    scores: dict[str, float] = defaultdict(float)
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] += 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] += 1.0 / (RRF_K + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:k]]


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], gold_ids: list[str]) -> float:
    """Fraction of gold chunks that appear anywhere in the retrieved list."""
    if not gold_ids:
        return 0.0
    hits = sum(1 for g in gold_ids if g in retrieved)
    return hits / len(gold_ids)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY environment variable.")
        return 1

    print("Loading corpus...")
    chunks = load_corpus()
    print(f"  {len(chunks)} chunks across {len({c.doc_id for c in chunks})} documents")

    print("Embedding corpus...")
    # Batch to a single API call for a small corpus
    chunk_embs = embed_batch([c.text for c in chunks])

    print("Building BM25 index...")
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    print("Loading eval questions...")
    questions = json.loads(EVAL_PATH.read_text())
    print(f"  {len(questions)} questions")

    # Embed all questions in one call to save API round-trips
    question_embs = embed_batch([q["question"] for q in questions])

    per_strategy_scores: dict[str, list[dict]] = {
        "dense-only": [],
        "bm25-only": [],
        "hybrid-rrf": [],
    }

    for q, q_emb in zip(questions, question_embs):
        gold = q["expected_chunk_ids"]

        dense_ids = dense_retrieve(q_emb, chunk_embs, chunks, k=TOP_K)
        bm25_ids = bm25_retrieve(q["question"], bm25, chunks, k=TOP_K)
        # Hybrid gets the full top-10 of each ranker before fusing,
        # matching Anchor's production behavior.
        dense_wide = dense_retrieve(q_emb, chunk_embs, chunks, k=10)
        bm25_wide = bm25_retrieve(q["question"], bm25, chunks, k=10)
        hybrid_ids = hybrid_retrieve(dense_wide, bm25_wide, k=TOP_K)

        for name, ids in [
            ("dense-only", dense_ids),
            ("bm25-only", bm25_ids),
            ("hybrid-rrf", hybrid_ids),
        ]:
            per_strategy_scores[name].append(
                {
                    "question_id": q["id"],
                    "type": q["type"],
                    "recall_at_k": recall_at_k(ids, gold),
                    "retrieved": ids,
                    "gold": gold,
                }
            )

    # -------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------
    def by_type(rows: list[dict]) -> dict[str, float]:
        buckets: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            buckets[r["type"]].append(r["recall_at_k"])
        return {t: sum(v) / len(v) for t, v in buckets.items()}

    summary = {}
    for name, rows in per_strategy_scores.items():
        summary[name] = {
            "overall": round(sum(r["recall_at_k"] for r in rows) / len(rows), 3),
            "by_type": {t: round(v, 3) for t, v in by_type(rows).items()},
        }

    # -------------------------------------------------------------------
    # Print + write report
    # -------------------------------------------------------------------
    types = sorted({q["type"] for q in questions})
    print("\n" + "=" * 64)
    print(f"RETRIEVAL COMPARISON — recall@{TOP_K}")
    print("=" * 64)
    header = f"{'strategy':<14}" + "".join(f"{t:<12}" for t in types) + f"{'overall':<10}"
    print(header)
    print("-" * len(header))
    for name in ("dense-only", "bm25-only", "hybrid-rrf"):
        row = f"{name:<14}"
        for t in types:
            row += f"{summary[name]['by_type'].get(t, 0.0):<12.2f}"
        row += f"{summary[name]['overall']:<10.2f}"
        print(row)
    print("=" * 64)

    report = {
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "chunk_tokens": CHUNK_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "rrf_k": RRF_K,
            "corpus_chunks": len(chunks),
            "questions": len(questions),
        },
        "summary": summary,
        "per_question": per_strategy_scores,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
