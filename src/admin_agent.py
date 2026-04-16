"""Admin-facing LangGraph agent for reviewing and managing reservations.

This is the second agent in the system. While the user-facing chatbot
(``chatbot.py``) collects reservation details and stages them as *pending*,
the admin agent helps the administrator:

- List and inspect pending reservations
- Check availability to validate feasibility
- Approve or reject reservations (with optional notes)
- View notification history

The admin agent is integrated into the FastAPI server (``server.py``) and
can also be used from a standalone admin CLI.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from . import db
from . import notifications as notif
from .config import settings


ADMIN_SYSTEM_PROMPT = """\
You are the SkyPark Central admin assistant. You help the parking administrator
manage reservations. Your capabilities:

  • List all pending reservations that need review.
  • Inspect a specific reservation by ID.
  • Check current parking availability to decide if a booking is feasible.
  • Approve a reservation (optionally with notes).
  • Reject a reservation (with a reason).
  • View recent notifications.

Decision guidelines:
  - Before approving, check that enough spaces are available for the requested
    period. If the garage is nearly full, warn the admin.
  - When rejecting, always include a clear reason so the guest understands.
  - You may recommend an action, but the final approve/reject must be an
    explicit admin instruction — never auto-approve.
  - Keep responses clear and structured.
"""


# ---------- admin tools ----------

@tool
def list_pending_reservations() -> str:
    """List all reservations with status 'pending' that need admin review."""
    rows = db.list_bookings(status="pending")
    if not rows:
        return "No pending reservations."
    parts = []
    for r in rows:
        parts.append(
            f"#{r['id']} | {r['first_name']} {r['last_name']} | "
            f"{r['car_plate']} | {r['start_ts']} → {r['end_ts']} | "
            f"created {r['created_at']}"
        )
    return f"{len(rows)} pending reservation(s):\n" + "\n".join(parts)


@tool
def inspect_reservation(booking_id: int) -> str:
    """Get full details of a specific reservation by ID."""
    b = db.get_booking(booking_id)
    if not b:
        return f"Reservation #{booking_id} not found."
    lines = [f"Reservation #{b['id']}"]
    for k in ("first_name", "last_name", "car_plate", "start_ts", "end_ts",
              "status", "admin_notes", "reviewed_at", "created_at"):
        lines.append(f"  {k}: {b.get(k, 'N/A')}")
    return "\n".join(lines)


@tool
def admin_check_availability(zone: Optional[str] = None) -> str:
    """Check current parking-space availability per zone.

    Args:
        zone: Optional zone filter ('L1', 'L2', 'L3', 'Rooftop').
    """
    rows = db.get_availability(zone=zone)
    if not rows:
        return f"No data for zone={zone!r}"
    return "\n".join(
        f"{r['zone']}: {r['available']}/{r['total']} free"
        for r in rows
    )


@tool
def approve_reservation(booking_id: int, notes: Optional[str] = None) -> str:
    """Approve a pending reservation.

    Args:
        booking_id: The reservation ID to approve.
        notes: Optional admin notes to attach.
    """
    ok = db.approve_booking(booking_id, admin_notes=notes or "")
    if not ok:
        return f"Could not approve #{booking_id} — it may not exist or is not pending."
    booking = db.get_booking(booking_id)
    notif.notify_booking_confirmed(booking)
    return f"Reservation #{booking_id} is now CONFIRMED."


@tool
def reject_reservation(booking_id: int, reason: str) -> str:
    """Reject a pending reservation with a reason.

    Args:
        booking_id: The reservation ID to reject.
        reason: Explanation for the guest.
    """
    if not reason.strip():
        return "A reason is required when rejecting a reservation."
    ok = db.reject_booking(booking_id, admin_notes=reason)
    if not ok:
        return f"Could not reject #{booking_id} — it may not exist or is not pending."
    booking = db.get_booking(booking_id)
    notif.notify_booking_rejected(booking)
    return f"Reservation #{booking_id} REJECTED. Reason: {reason}"


@tool
def view_notifications(unread_only: bool = False) -> str:
    """View admin notifications.

    Args:
        unread_only: If True, show only unread notifications.
    """
    items = notif.get_notifications(unread_only=unread_only)
    if not items:
        return "No notifications."
    parts = []
    for n in items:
        read_mark = " " if n["read"] else "*"
        parts.append(f"[{read_mark}] #{n['id']} ({n['kind']}) booking={n['booking_id']} — {n['subject']}")
    return f"{len(items)} notification(s):\n" + "\n".join(parts)


ADMIN_TOOLS = [
    list_pending_reservations,
    inspect_reservation,
    admin_check_availability,
    approve_reservation,
    reject_reservation,
    view_notifications,
]


# ---------- agent factory ----------

@lru_cache(maxsize=1)
def build_admin_agent():
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    db.init_db()
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    return create_react_agent(
        llm,
        tools=ADMIN_TOOLS,
        checkpointer=MemorySaver(),
    )


def admin_chat(message: str, thread_id: str = "admin-default") -> str:
    """Run one admin turn through the admin agent. Returns the reply text."""
    agent = build_admin_agent()
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {
            "messages": [
                SystemMessage(content=ADMIN_SYSTEM_PROMPT),
                HumanMessage(content=message),
            ]
        },
        config=config,
    )
    final = result["messages"][-1].content
    if isinstance(final, list):
        final = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in final
        )
    return final
