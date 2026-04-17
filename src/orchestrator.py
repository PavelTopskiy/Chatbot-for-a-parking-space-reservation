"""Stage 4 — full-pipeline orchestrator using LangGraph.

This module wires the Stage 1 user chatbot, the Stage 2 admin approval
step, and the Stage 3 MCP file recorder into a single LangGraph
``StateGraph``. It uses LangGraph's native human-in-the-loop primitive
(``interrupt`` + ``Command``) so the graph pauses when an admin decision
is needed and resumes deterministically once the admin responds.

The orchestrator *complements* the existing async path (user CLI + admin
dashboard + MCP server). The graph is useful for:

- Integration-testing the whole pipeline end-to-end in one process
- Live demos that show all three components interact via LangGraph
- Embedding the full workflow inside another service or notebook

Graph shape::

             ┌──────────────┐
    START ──▶│  user_node   │  ── Stage 1 chatbot (RAG + SQL tools)
             └──────┬───────┘
                    │  (conditional: did a reservation get staged?)
            ┌───────┴────────┐
            │                │
           END        ┌─────────────────┐
                      │ approval_node   │  ── interrupt(): pause for admin
                      └──────┬──────────┘
                             │
                      ┌──────▼──────────┐
                      │  decision_node  │  ── apply approve/reject to DB
                      └──────┬──────────┘
                             │ (confirmed?)
                      ┌──────┴──────────┐
                      │                 │
          ┌────────────────────┐       END
          │  recording_node    │  ── Stage 3 MCP write
          └──────┬─────────────┘
                 │
                END

Public API
----------
- ``build_orchestrator()`` — compile and return the graph
- ``start_turn(graph, user_message, thread_id)`` — run one user turn
- ``resume_with_decision(graph, thread_id, action, notes)`` — resume a
  paused graph with an admin approve/reject decision
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import db
from .chatbot import chat as user_chat
from .mcp_client import write_confirmed_reservation as mcp_write

log = logging.getLogger(__name__)

# Matches the reply text produced by ``chatbot.create_reservation``:
#   "Reservation #<id> staged as PENDING. ..."
# Using a regex on the reply is concurrency-safe — each thread's
# ``user_chat`` call returns its own reply referencing its own booking.
_RESERVATION_ID_RE = re.compile(r"Reservation\s+#(\d+)")


class OrchestratorState(TypedDict, total=False):
    """State flowing through the orchestrator graph.

    Only the keys a given node writes to are returned by that node;
    LangGraph merges them into the state dict.
    """
    user_message: str
    assistant_reply: str
    thread_id: str
    booking_id: Optional[int]
    booking: Optional[dict]
    decision: Optional[dict]       # {"action": "approve"|"reject", "notes": str}
    mcp_result: Optional[str]
    final_status: str              # "confirmed" | "rejected" | "conversation" | ...


# ---------- nodes ----------

def user_node(state: OrchestratorState) -> dict:
    """Run one user turn through the Stage 1 chatbot agent.

    Detects whether the turn staged a pending booking by regex-matching
    the chatbot's reply — the ``create_reservation`` tool returns
    ``"Reservation #<id> staged as PENDING..."`` which this parses to
    recover the new booking's ID. This is concurrency-safe because each
    thread receives its own reply for its own booking.
    """
    result = user_chat(
        state["user_message"],
        thread_id=state.get("thread_id", "default"),
    )
    reply = result.get("reply") or ""
    update: dict[str, Any] = {"assistant_reply": reply}

    match = _RESERVATION_ID_RE.search(reply)
    if match:
        booking_id = int(match.group(1))
        booking = db.get_booking(booking_id)
        if booking and booking.get("status") == "pending":
            update["booking_id"] = booking_id
            update["booking"] = booking
            log.info("user_node staged booking #%s", booking_id)
            return update

    update["final_status"] = "conversation"
    return update


def route_after_user(state: OrchestratorState) -> str:
    """Branch: if a booking was staged, go to approval; else end."""
    if state.get("booking_id"):
        return "approval_node"
    return END


def approval_node(state: OrchestratorState) -> dict:
    """Pause the graph and wait for the admin decision.

    ``interrupt`` emits a payload to the caller and suspends the graph.
    Execution resumes here when the caller sends
    ``Command(resume={"action": "approve"|"reject", "notes": str})``.
    The resume value becomes the return value of ``interrupt``.

    This node has no side effects before ``interrupt``, so it is safe
    for LangGraph's replay-on-resume semantics.
    """
    payload = {
        "type": "approval_required",
        "booking_id": state.get("booking_id"),
        "booking": state.get("booking"),
        "prompt": "Admin decision required: approve or reject this reservation.",
    }
    decision = interrupt(payload)
    return {"decision": decision}


def decision_node(state: OrchestratorState) -> dict:
    """Apply the admin's approve/reject decision to the database."""
    booking_id = state["booking_id"]
    decision = state.get("decision") or {}
    action = (decision.get("action") or "").lower()
    notes = decision.get("notes", "") or ""

    if action == "approve":
        ok = db.approve_booking(booking_id, admin_notes=notes)
        status = "confirmed" if ok else "failed_to_approve"
    elif action == "reject":
        ok = db.reject_booking(booking_id, admin_notes=notes)
        status = "rejected" if ok else "failed_to_reject"
    else:
        status = f"unknown_action:{action!r}"

    refreshed = db.get_booking(booking_id)
    log.info("decision_node booking #%s -> %s", booking_id, status)
    return {"final_status": status, "booking": refreshed}


def route_after_decision(state: OrchestratorState) -> str:
    if state.get("final_status") == "confirmed":
        return "recording_node"
    return END


def recording_node(state: OrchestratorState) -> dict:
    """Call the Stage 3 MCP server to persist the confirmed reservation.

    Failures here do not halt the graph — the booking is already
    confirmed in SQLite. The MCP result string is stored in state so
    callers can report or retry.
    """
    booking = state.get("booking") or {}
    try:
        result = mcp_write(booking)
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        log.warning("recording_node MCP call failed: %s", exc)
        result = f"error: {exc}"
    return {"mcp_result": result}


# ---------- graph factory ----------

def build_orchestrator():
    """Build and compile the full-pipeline orchestrator graph.

    Uses an in-memory checkpointer so each ``thread_id`` has its own
    isolated state. For multi-process deployments, swap ``MemorySaver``
    for ``SqliteSaver`` or ``PostgresSaver``.
    """
    g = StateGraph(OrchestratorState)
    g.add_node("user_node", user_node)
    g.add_node("approval_node", approval_node)
    g.add_node("decision_node", decision_node)
    g.add_node("recording_node", recording_node)

    g.add_edge(START, "user_node")
    g.add_conditional_edges("user_node", route_after_user)
    g.add_edge("approval_node", "decision_node")
    g.add_conditional_edges("decision_node", route_after_decision)
    g.add_edge("recording_node", END)

    return g.compile(checkpointer=MemorySaver())


# ---------- convenience API ----------

def _interrupt_payload(snapshot) -> Optional[dict]:
    """Extract the pending interrupt payload, if any, from a state snapshot."""
    for task in snapshot.tasks or ():
        for intr in getattr(task, "interrupts", ()) or ():
            return intr.value
    return None


def start_turn(graph, user_message: str, thread_id: str) -> dict:
    """Run one user turn.

    If no reservation is staged, the graph completes and the result is
    returned with ``interrupted=False``. If a reservation is staged, the
    graph pauses at ``approval_node`` and the result carries
    ``interrupted=True`` and the approval payload.
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(
        {"user_message": user_message, "thread_id": thread_id},
        config=config,
    )
    snapshot = graph.get_state(config)
    payload = _interrupt_payload(snapshot)
    if payload is not None:
        return {
            "interrupted": True,
            "thread_id": thread_id,
            "payload": payload,
            "assistant_reply": snapshot.values.get("assistant_reply"),
        }
    return {
        "interrupted": False,
        "thread_id": thread_id,
        "assistant_reply": snapshot.values.get("assistant_reply"),
        "final_status": snapshot.values.get("final_status", "conversation"),
    }


def resume_with_decision(
    graph,
    thread_id: str,
    action: str,
    notes: str = "",
) -> dict:
    """Resume a paused graph with the admin's decision.

    Returns the post-run state including ``final_status`` and, if
    approved, ``mcp_result`` from the recording node.
    """
    config = {"configurable": {"thread_id": thread_id}}
    decision = {"action": action, "notes": notes}
    graph.invoke(Command(resume=decision), config=config)
    snapshot = graph.get_state(config)
    return {
        "thread_id": thread_id,
        "final_status": snapshot.values.get("final_status"),
        "booking": snapshot.values.get("booking"),
        "mcp_result": snapshot.values.get("mcp_result"),
    }
