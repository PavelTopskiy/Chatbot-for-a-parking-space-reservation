"""Integration tests for the Stage 4 LangGraph orchestrator.

These tests exercise the real compiled graph — user_node, approval_node
(with ``interrupt``), decision_node, and recording_node — but stub out
the two external dependencies:

- the OpenAI-backed Stage 1 chatbot (``orchestrator.user_chat``), and
- the Stage 3 MCP client (``orchestrator.mcp_write``),

so the tests run offline and deterministically.

The database is redirected to a per-test temp file via monkeypatching
``src.db.settings.db_path``.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest


# Satisfy config load-time checks without calling OpenAI.
os.environ.setdefault("OPENAI_API_KEY", "test-dummy")


# ---------- fixtures ----------

@pytest.fixture
def temp_db(monkeypatch):
    """Point the DB at a temp SQLite file and initialise its schema."""
    from src import db as db_mod

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setattr(
        db_mod,
        "settings",
        SimpleNamespace(db_path=path),
    )
    db_mod.init_db()
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _make_fake_chat_staging_reservation(first_name="Test", last_name="User",
                                         plate="TEST-1"):
    """Return a chatbot stub that stages a new booking on every call."""
    def _fake_chat(message, thread_id="default"):
        from src import db
        bid = db.create_booking(
            first_name=first_name,
            last_name=last_name,
            car_plate=plate,
            start_ts="2026-05-01T09:00",
            end_ts="2026-05-01T18:00",
        )
        return {
            "reply": f"Reservation #{bid} staged as PENDING.",
            "input_findings": [],
            "output_findings": [],
            "blocked": False,
        }
    return _fake_chat


def _fake_chat_no_reservation(message, thread_id="default"):
    return {
        "reply": "Our working hours are 24/7.",
        "input_findings": [],
        "output_findings": [],
        "blocked": False,
    }


# ---------- tests ----------

def test_no_reservation_completes_without_interrupt(temp_db, monkeypatch):
    """A plain info question should end at END, no approval flow."""
    from src import orchestrator

    monkeypatch.setattr(orchestrator, "user_chat", _fake_chat_no_reservation)

    graph = orchestrator.build_orchestrator()
    result = orchestrator.start_turn(graph, "what are your hours?", "t-info")

    assert result["interrupted"] is False
    assert "24/7" in result["assistant_reply"]
    assert result.get("final_status") == "conversation"


def test_reservation_triggers_approval_interrupt(temp_db, monkeypatch):
    """Staging a booking should pause the graph with an approval payload."""
    from src import orchestrator

    monkeypatch.setattr(
        orchestrator, "user_chat",
        _make_fake_chat_staging_reservation(),
    )

    graph = orchestrator.build_orchestrator()
    result = orchestrator.start_turn(graph, "reserve a spot", "t-stage")

    assert result["interrupted"] is True, result
    payload = result["payload"]
    assert payload["type"] == "approval_required"
    assert payload["booking"]["first_name"] == "Test"
    assert payload["booking"]["status"] == "pending"


def test_approve_flow_writes_to_mcp(temp_db, monkeypatch):
    """approve decision → booking confirmed → mcp_write called once."""
    from src import orchestrator

    mcp_calls = []

    def fake_mcp(booking):
        mcp_calls.append(dict(booking))
        return f"ok: wrote #{booking['id']}"

    monkeypatch.setattr(
        orchestrator, "user_chat",
        _make_fake_chat_staging_reservation(first_name="Alice", plate="A-1"),
    )
    monkeypatch.setattr(orchestrator, "mcp_write", fake_mcp)

    graph = orchestrator.build_orchestrator()
    r = orchestrator.start_turn(graph, "reserve", "t-approve")
    assert r["interrupted"]

    final = orchestrator.resume_with_decision(
        graph, "t-approve", "approve", "looks good",
    )

    assert final["final_status"] == "confirmed"
    assert final["booking"]["status"] == "confirmed"
    assert final["booking"]["admin_notes"] == "looks good"
    assert len(mcp_calls) == 1
    assert mcp_calls[0]["id"] == final["booking"]["id"]
    assert final["mcp_result"].startswith("ok:")


def test_reject_flow_skips_mcp(temp_db, monkeypatch):
    """reject decision → booking rejected → mcp_write NOT called."""
    from src import orchestrator

    mcp_calls = []

    def fake_mcp(booking):
        mcp_calls.append(booking)
        return "should_not_happen"

    monkeypatch.setattr(
        orchestrator, "user_chat",
        _make_fake_chat_staging_reservation(),
    )
    monkeypatch.setattr(orchestrator, "mcp_write", fake_mcp)

    graph = orchestrator.build_orchestrator()
    r = orchestrator.start_turn(graph, "reserve", "t-reject")
    assert r["interrupted"]

    final = orchestrator.resume_with_decision(
        graph, "t-reject", "reject", "garage full on that date",
    )

    assert final["final_status"] == "rejected"
    assert final["booking"]["status"] == "rejected"
    assert final["booking"]["admin_notes"] == "garage full on that date"
    assert mcp_calls == []
    assert final.get("mcp_result") is None


def test_mcp_failure_does_not_break_graph(temp_db, monkeypatch):
    """If the MCP call raises, the graph still completes, booking stays confirmed."""
    from src import orchestrator

    def boom(booking):
        raise RuntimeError("mcp server unreachable")

    monkeypatch.setattr(
        orchestrator, "user_chat",
        _make_fake_chat_staging_reservation(),
    )
    monkeypatch.setattr(orchestrator, "mcp_write", boom)

    graph = orchestrator.build_orchestrator()
    r = orchestrator.start_turn(graph, "reserve", "t-mcp-fail")
    assert r["interrupted"]

    final = orchestrator.resume_with_decision(graph, "t-mcp-fail", "approve")

    assert final["final_status"] == "confirmed"
    assert final["booking"]["status"] == "confirmed"
    assert "error" in (final["mcp_result"] or "").lower()


def test_concurrent_thread_ids_are_isolated(temp_db, monkeypatch):
    """Two parallel threads in the same graph do not see each other's state."""
    from src import orchestrator

    monkeypatch.setattr(
        orchestrator, "user_chat",
        _make_fake_chat_staging_reservation(),
    )
    monkeypatch.setattr(orchestrator, "mcp_write", lambda b: "ok")

    graph = orchestrator.build_orchestrator()
    r1 = orchestrator.start_turn(graph, "reserve", "t-A")
    r2 = orchestrator.start_turn(graph, "reserve", "t-B")

    assert r1["payload"]["booking"]["id"] != r2["payload"]["booking"]["id"]

    # Approve A, reject B.
    fa = orchestrator.resume_with_decision(graph, "t-A", "approve")
    fb = orchestrator.resume_with_decision(graph, "t-B", "reject", "nope")

    assert fa["final_status"] == "confirmed"
    assert fb["final_status"] == "rejected"
    assert fa["booking"]["id"] != fb["booking"]["id"]


def test_parallel_orchestrator_runs_do_not_cross_bookings(temp_db, monkeypatch):
    """Regression: under real thread-pool concurrency, each orchestrator
    run must pick up its own booking — not another thread's. Earlier the
    user_node used a pending-ID-diff strategy that would race (thread A
    could see thread B's newly-staged booking in its ``after`` set).
    The fix parses the booking ID from the chatbot's reply, which is
    concurrency-safe.
    """
    import concurrent.futures as cf
    from src import db, orchestrator

    monkeypatch.setattr(
        orchestrator, "user_chat",
        _make_fake_chat_staging_reservation(),
    )
    monkeypatch.setattr(orchestrator, "mcp_write", lambda b: "ok")

    graph = orchestrator.build_orchestrator()
    N = 40

    def _run(i: int) -> tuple[int, str]:
        tid = f"par-{i:04d}"
        r = orchestrator.start_turn(graph, "reserve", tid)
        assert r["interrupted"]
        f = orchestrator.resume_with_decision(graph, tid, "approve")
        return r["payload"]["booking"]["id"], f["final_status"]

    with cf.ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_run, range(N)))

    ids = [bid for bid, _ in results]
    statuses = [s for _, s in results]

    assert all(s == "confirmed" for s in statuses), statuses
    assert len(set(ids)) == N, "each run must claim a distinct booking id"

    # Every booking in the DB should be confirmed.
    pending = db.list_bookings(status="pending")
    assert pending == [], f"no booking should be left pending, got: {pending}"
