"""
Minimal LLM evaluation harness.

Runs each test case in evals/, scores accuracy via embedding similarity,
groundedness via LLM-as-judge, and latency via wall-clock timing.
Writes a structured report to report.json.

Usage:
    export OPENAI_API_KEY=sk-...
    python harness.py
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from judges import judge_grounded

client = OpenAI()

EVALS_DIR = Path(__file__).parent / "evals"
REPORT_PATH = Path(__file__).parent / "report.json"

ACCURACY_THRESHOLD = 0.85


@dataclass
class EvalResult:
    test_id: str
    accuracy: float
    accuracy_passed: bool
    grounded: bool
    latency_ms: float
    response: str


def embed(text: str) -> list[float]:
    """Return an embedding vector for the given text."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)


def run_case(case: dict) -> EvalResult:
    """Run a single eval case and return scored result."""
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=case["model"],
        messages=case["messages"],
        temperature=0,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    answer = resp.choices[0].message.content

    accuracy = cosine_similarity(embed(answer), embed(case["expected"]))
    grounded = judge_grounded(answer, case.get("context", ""))

    return EvalResult(
        test_id=case["id"],
        accuracy=round(accuracy, 4),
        accuracy_passed=accuracy >= ACCURACY_THRESHOLD,
        grounded=grounded,
        latency_ms=round(latency_ms, 2),
        response=answer,
    )


def main() -> None:
    cases = sorted(EVALS_DIR.glob("*.json"))
    if not cases:
        print(f"No eval cases found in {EVALS_DIR}")
        return

    results = []
    for path in cases:
        case = json.loads(path.read_text())
        print(f"Running {case['id']}...")
        result = run_case(case)
        results.append(asdict(result))
        status = "PASS" if result.accuracy_passed and result.grounded else "FAIL"
        print(f"  [{status}] acc={result.accuracy} grounded={result.grounded} latency={result.latency_ms}ms")

    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nRan {len(results)} cases. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
