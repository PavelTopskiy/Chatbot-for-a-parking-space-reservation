"""Stage 4 — single-process demo of the full orchestrator pipeline.

Walks through: user message → (if reservation staged) admin prompt in
the same terminal → MCP write. Useful for demoing the whole pipeline
without spinning up three separate servers.

For production-style deployment, keep using the 3-terminal flow
(mcp_server + server + cli). This CLI is for exercising the LangGraph
orchestrator end-to-end in one process.
"""
from __future__ import annotations

import uuid

from .orchestrator import build_orchestrator, resume_with_decision, start_turn


BANNER = """\
SkyPark orchestrator demo — Stage 4
Type a message to the parking chatbot. If it stages a reservation,
this same terminal will prompt you for an admin approve/reject
decision. Type 'exit' to quit.
"""


def _print_booking_card(booking: dict) -> None:
    print()
    print("=" * 60)
    print("ADMIN APPROVAL REQUIRED")
    print(
        f"  #{booking.get('id')} — "
        f"{booking.get('first_name')} {booking.get('last_name')}"
    )
    print(f"  plate : {booking.get('car_plate')}")
    print(f"  period: {booking.get('start_ts')}  →  {booking.get('end_ts')}")
    print("=" * 60)


def _prompt_decision() -> tuple[str, str]:
    action = ""
    while action not in {"approve", "reject"}:
        action = input("admin action [approve/reject] > ").strip().lower()
    notes = input("notes (optional) > ").strip()
    return action, notes


def run() -> None:
    graph = build_orchestrator()
    thread_id = f"orch-{uuid.uuid4().hex[:8]}"
    print(BANNER)
    print(f"(thread_id: {thread_id})\n")

    while True:
        try:
            msg = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not msg:
            continue
        if msg.lower() in {"exit", "quit"}:
            return

        result = start_turn(graph, msg, thread_id)
        reply = result.get("assistant_reply") or ""
        print(f"bot > {reply}")

        if not result.get("interrupted"):
            continue

        payload = result.get("payload") or {}
        booking = payload.get("booking") or {}
        _print_booking_card(booking)
        action, notes = _prompt_decision()

        final = resume_with_decision(graph, thread_id, action, notes)
        print(f"\n  final status : {final.get('final_status')}")
        if final.get("mcp_result"):
            print(f"  mcp result   : {final['mcp_result']}")
        print()


if __name__ == "__main__":
    run()
