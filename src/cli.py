"""Interactive CLI for the parking chatbot.

Run with::

    python -m src.cli

Type ``/quit`` to exit, ``/new`` to start a fresh conversation thread.
"""
from __future__ import annotations

import sys
import uuid

from .chatbot import chat
from .db import init_db


BANNER = """\
SkyPark Central — Parking Chatbot (Stage 1)
Type your question, or '/quit' to exit, '/new' for a new conversation.
"""


def main() -> int:
    init_db()
    print(BANNER)
    thread_id = uuid.uuid4().hex[:8]
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue
        if user == "/quit":
            return 0
        if user == "/new":
            thread_id = uuid.uuid4().hex[:8]
            print(f"(new conversation: {thread_id})")
            continue

        result = chat(user, thread_id=thread_id)
        print(f"bot > {result['reply']}")
        findings = result["input_findings"] + result["output_findings"]
        if findings:
            print(f"      [guardrails: {', '.join(sorted(set(findings)))}]")


if __name__ == "__main__":
    sys.exit(main())
