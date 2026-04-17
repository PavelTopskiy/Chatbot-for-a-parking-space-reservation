"""Stage 4 — load-testing harness for each component of the pipeline.

Run one scenario at a time::

    python3 -m scripts.load_test db          # concurrent DB approvals
    python3 -m scripts.load_test mcp         # concurrent MCP writes
    python3 -m scripts.load_test orch        # concurrent orchestrator threads
    python3 -m scripts.load_test all         # run all of the above

Each scenario prints p50 / p95 / p99 latency, throughput, and error
count. The harness uses ``concurrent.futures.ThreadPoolExecutor`` so it
exercises real lock contention (SQLite, fcntl on the reservations file,
LangGraph MemorySaver).

Scenarios that hit external services (``mcp``) require that service to
be running. ``db`` and ``orch`` are self-contained.

This is deliberately *not* a pytest test — load tests are too slow and
too environment-sensitive to gate CI on. Run it on demand.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Callable, List


# Satisfy config-load for src modules even without real credentials.
os.environ.setdefault("OPENAI_API_KEY", "load-test-dummy")


# ---------- shared helpers ----------

def _percentiles(samples_ms: List[float]) -> dict:
    if not samples_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    samples_ms = sorted(samples_ms)
    n = len(samples_ms)
    def _pct(p):
        i = min(int(round((p / 100.0) * (n - 1))), n - 1)
        return samples_ms[i]
    return {
        "p50": _pct(50),
        "p95": _pct(95),
        "p99": _pct(99),
        "max": samples_ms[-1],
        "mean": statistics.mean(samples_ms),
    }


def _run_workers(fn: Callable[[int], float], total: int, workers: int,
                 label: str) -> None:
    """Submit ``total`` calls to ``fn(i)`` across ``workers`` threads.

    Each call must return elapsed milliseconds (or raise).
    """
    print(f"\n=== {label} ===")
    print(f"  total={total}  workers={workers}")

    latencies: list[float] = []
    errors: list[str] = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, i) for i in range(total)]
        for f in cf.as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception as exc:  # noqa: BLE001 — load test, we log
                errors.append(repr(exc))
    wall_s = time.perf_counter() - t0

    p = _percentiles(latencies)
    tput = len(latencies) / wall_s if wall_s > 0 else 0.0
    print(f"  ok={len(latencies)}  errors={len(errors)}")
    print(f"  wall={wall_s:.2f}s  throughput={tput:.1f} req/s")
    print(
        f"  latency ms: mean={p['mean']:.1f}  p50={p['p50']:.1f}  "
        f"p95={p['p95']:.1f}  p99={p['p99']:.1f}  max={p['max']:.1f}"
    )
    if errors:
        print("  first 3 errors:")
        for e in errors[:3]:
            print(f"    - {e}")


# ---------- scenario: concurrent DB approvals ----------

def scenario_db(total: int = 500, workers: int = 50) -> None:
    """Stage N pending bookings, then race ``approve_booking`` from
    ``workers`` threads. Validates SQLite concurrency for the admin path.
    """
    from src import db as db_mod

    fd, path = tempfile.mkstemp(suffix="-loadtest.db")
    os.close(fd)
    db_mod.settings = SimpleNamespace(db_path=path)
    db_mod.init_db()

    ids = [
        db_mod.create_booking(
            first_name=f"User{i}",
            last_name="Load",
            car_plate=f"LD-{i:04d}",
            start_ts="2026-06-01T09:00",
            end_ts="2026-06-01T18:00",
        )
        for i in range(total)
    ]

    def _approve(i: int) -> float:
        t = time.perf_counter()
        db_mod.approve_booking(ids[i], admin_notes="load test")
        return (time.perf_counter() - t) * 1000

    try:
        _run_workers(_approve, total, workers, "DB approvals (SQLite)")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------- scenario: concurrent MCP writes ----------

def scenario_mcp(total: int = 200, workers: int = 20) -> None:
    """Hammer the MCP server with parallel write_confirmed_reservation
    calls. Validates bearer auth + ``fcntl`` file locking under
    contention.

    Prereq: ``python3 -m src.mcp_server`` must be running.
    """
    from src.mcp_client import write_confirmed_reservation

    def _write(i: int) -> float:
        booking = {
            "id": 10000 + i,
            "first_name": "Load",
            "last_name": f"Tester{i}",
            "car_plate": f"LT-{i:04d}",
            "start_ts": "2026-06-01T09:00",
            "end_ts": "2026-06-01T18:00",
            "reviewed_at": "2026-06-01T08:30Z",
        }
        t = time.perf_counter()
        res = write_confirmed_reservation(booking)
        elapsed = (time.perf_counter() - t) * 1000
        if res.lower().startswith("mcp write failed") or "error" in res.lower()[:10]:
            raise RuntimeError(res)
        return elapsed

    _run_workers(_write, total, workers, "MCP writes (HTTP + fcntl)")


# ---------- scenario: concurrent orchestrator threads ----------

def scenario_orch(total: int = 100, workers: int = 20) -> None:
    """Spin up ``total`` parallel orchestrator threads, each staging a
    booking and approving it. Uses stubbed user_chat + mcp_write so no
    external services are hit — this measures LangGraph + checkpointer
    + SQLite overhead.
    """
    from src import db as db_mod
    from src import orchestrator

    fd, path = tempfile.mkstemp(suffix="-orchload.db")
    os.close(fd)
    db_mod.settings = SimpleNamespace(db_path=path)
    db_mod.init_db()

    def _fake_chat(message, thread_id="default"):
        bid = db_mod.create_booking(
            first_name="Orch",
            last_name=thread_id,
            car_plate=f"OR-{thread_id[-6:]}",
            start_ts="2026-06-01T09:00",
            end_ts="2026-06-01T18:00",
        )
        return {
            "reply": f"Reservation #{bid} staged.",
            "input_findings": [],
            "output_findings": [],
            "blocked": False,
        }

    orchestrator.user_chat = _fake_chat          # stub Stage 1
    orchestrator.mcp_write = lambda b: "ok"      # stub Stage 3

    graph = orchestrator.build_orchestrator()

    def _run(i: int) -> float:
        tid = f"load-{i:04d}"
        t = time.perf_counter()
        r = orchestrator.start_turn(graph, "reserve", tid)
        if not r["interrupted"]:
            raise RuntimeError("expected interrupt")
        f = orchestrator.resume_with_decision(graph, tid, "approve", "ok")
        if f["final_status"] != "confirmed":
            raise RuntimeError(f"unexpected status {f['final_status']}")
        return (time.perf_counter() - t) * 1000

    try:
        _run_workers(_run, total, workers, "Orchestrator threads (LangGraph)")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------- entrypoint ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 4 load test")
    parser.add_argument(
        "scenario",
        choices=["db", "mcp", "orch", "all"],
        help="Which scenario to run.",
    )
    parser.add_argument("--total", type=int, default=0,
                        help="Total requests (0 = scenario default).")
    parser.add_argument("--workers", type=int, default=0,
                        help="Concurrent workers (0 = scenario default).")
    args = parser.parse_args(argv)

    def _opts(default_total, default_workers):
        return (
            args.total or default_total,
            args.workers or default_workers,
        )

    if args.scenario in {"db", "all"}:
        scenario_db(*_opts(500, 50))
    if args.scenario in {"mcp", "all"}:
        scenario_mcp(*_opts(200, 20))
    if args.scenario in {"orch", "all"}:
        scenario_orch(*_opts(100, 20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
