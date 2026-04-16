"""Evaluate retrieval quality and end-to-end latency for the parking chatbot.

Metrics:
- **Recall@K**     — fraction of questions for which at least one retrieved
                     chunk has a topic in the gold-topic set.
- **Precision@K**  — fraction of retrieved chunks (across all questions) whose
                     topic is in the gold-topic set, averaged per question.
- **MRR@K**        — mean reciprocal rank of the first relevant chunk.
- **Retrieval latency** — wall-clock per retrieval call (p50/p95).
- **End-to-end latency** — wall-clock per ``chat()`` call (p50/p95), only
                           when ``--end-to-end`` is passed (uses Anthropic API).

Usage::

    python -m eval.evaluate                 # retrieval metrics only
    python -m eval.evaluate --end-to-end    # also run full chatbot turns
    python -m eval.evaluate --k 6           # change top-k

Writes a markdown report to ``eval/results/report.md``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.config import settings
from src.retriever import get_retriever


HERE = Path(__file__).resolve().parent
QUESTIONS_PATH = HERE / "questions.json"
RESULTS_DIR = HERE / "results"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100) * (len(s) - 1)))))
    return s[k]


def evaluate_retrieval(k: int) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text())
    retriever = get_retriever(k=k)

    recall_hits = 0
    precision_per_q: list[float] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    per_question: list[dict] = []

    for item in questions:
        q = item["q"]
        gold = set(item["gold_topics"])

        t0 = time.perf_counter()
        docs = retriever.invoke(q)
        latencies.append(time.perf_counter() - t0)

        topics = [d.metadata.get("topic", "") for d in docs]
        rel_flags = [t in gold for t in topics]

        hit = any(rel_flags)
        if hit:
            recall_hits += 1
        precision_per_q.append(sum(rel_flags) / max(len(rel_flags), 1))
        rr = 0.0
        for rank, flag in enumerate(rel_flags, 1):
            if flag:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        per_question.append({
            "q": q,
            "gold_topics": sorted(gold),
            "retrieved_topics": topics,
            "hit": hit,
            "precision": precision_per_q[-1],
            "rr": rr,
            "latency_s": latencies[-1],
        })

    n = len(questions)
    return {
        "k": k,
        "n": n,
        "recall_at_k": recall_hits / n,
        "precision_at_k": statistics.mean(precision_per_q),
        "mrr_at_k": statistics.mean(reciprocal_ranks),
        "retrieval_latency": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "mean": statistics.mean(latencies),
        },
        "per_question": per_question,
    }


def evaluate_end_to_end() -> dict:
    """Run the full chatbot turn for each question and time it."""
    from src.chatbot import chat  # imported lazily so retrieval-only runs cheap

    questions = json.loads(QUESTIONS_PATH.read_text())
    latencies: list[float] = []
    transcripts: list[dict] = []
    for i, item in enumerate(questions):
        q = item["q"]
        t0 = time.perf_counter()
        result = chat(q, thread_id=f"eval-{i}")
        latencies.append(time.perf_counter() - t0)
        transcripts.append({
            "q": q,
            "reply": result["reply"],
            "latency_s": latencies[-1],
            "guardrail_findings": (
                result["input_findings"] + result["output_findings"]
            ),
        })
    return {
        "n": len(questions),
        "e2e_latency": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "mean": statistics.mean(latencies),
        },
        "transcripts": transcripts,
    }


def render_report(retrieval: dict, e2e: dict | None) -> str:
    lines = [
        "# Parking Chatbot — Evaluation Report",
        "",
        f"- Questions: **{retrieval['n']}**",
        f"- Top-K: **{retrieval['k']}**",
        f"- Embedding model: `{settings.embedding_model}`",
        f"- LLM: `{settings.openai_model}`",
        "",
        "## Retrieval quality",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Recall@{retrieval['k']}    | **{retrieval['recall_at_k']:.3f}** |",
        f"| Precision@{retrieval['k']} | **{retrieval['precision_at_k']:.3f}** |",
        f"| MRR@{retrieval['k']}       | **{retrieval['mrr_at_k']:.3f}** |",
        "",
        "## Retrieval latency (seconds)",
        "",
        f"| p50 | p95 | mean |",
        f"|---|---|---|",
        f"| {retrieval['retrieval_latency']['p50']:.3f} "
        f"| {retrieval['retrieval_latency']['p95']:.3f} "
        f"| {retrieval['retrieval_latency']['mean']:.3f} |",
        "",
    ]
    if e2e:
        lines += [
            "## End-to-end latency (seconds)",
            "",
            f"| p50 | p95 | mean |",
            f"|---|---|---|",
            f"| {e2e['e2e_latency']['p50']:.3f} "
            f"| {e2e['e2e_latency']['p95']:.3f} "
            f"| {e2e['e2e_latency']['mean']:.3f} |",
            "",
        ]

    lines += ["## Per-question retrieval results", ""]
    for row in retrieval["per_question"]:
        mark = "✓" if row["hit"] else "✗"
        lines.append(
            f"- {mark} **{row['q']}** "
            f"— gold={row['gold_topics']} retrieved={row['retrieved_topics']} "
            f"(P={row['precision']:.2f}, RR={row['rr']:.2f}, "
            f"{row['latency_s']*1000:.0f} ms)"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--end-to-end", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    retrieval = evaluate_retrieval(k=args.k)
    e2e = evaluate_end_to_end() if args.end_to_end else None

    report = render_report(retrieval, e2e)
    out_md = RESULTS_DIR / "report.md"
    out_md.write_text(report, encoding="utf-8")

    out_json = RESULTS_DIR / "report.json"
    out_json.write_text(
        json.dumps({"retrieval": retrieval, "end_to_end": e2e}, indent=2),
        encoding="utf-8",
    )

    print(report)
    print(f"\nWrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
