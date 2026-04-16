"""Standalone admin CLI for reviewing reservations.

Run with::

    python3 -m src.admin_cli

Commands:
    (free text)  — talk to the admin AI assistant
    /pending     — list pending reservations
    /approve ID  — approve a reservation
    /reject ID   — reject a reservation (will prompt for reason)
    /status ID   — inspect a reservation
    /quit        — exit
"""
from __future__ import annotations

import sys

from . import db
from . import notifications as notif
from .admin_agent import admin_chat


BANNER = """\
SkyPark Central — Admin Console (Stage 2)
Commands: /pending, /approve ID, /reject ID, /status ID, /quit
Or type a free-text question for the AI assistant.
"""


def main() -> int:
    db.init_db()
    print(BANNER)
    while True:
        try:
            line = input("admin > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line == "/quit":
            return 0

        if line == "/pending":
            rows = db.list_bookings(status="pending")
            if not rows:
                print("  No pending reservations.")
            for r in rows:
                print(
                    f"  #{r['id']} | {r['first_name']} {r['last_name']} | "
                    f"{r['car_plate']} | {r['start_ts']} -> {r['end_ts']}"
                )
            continue

        if line.startswith("/approve "):
            try:
                bid = int(line.split()[1])
            except (ValueError, IndexError):
                print("  Usage: /approve ID")
                continue
            notes = input("  Notes (optional): ").strip()
            ok = db.approve_booking(bid, admin_notes=notes)
            if ok:
                booking = db.get_booking(bid)
                notif.notify_booking_confirmed(booking)
                print(f"  Reservation #{bid} CONFIRMED.")
            else:
                print(f"  Could not approve #{bid} (not found or not pending).")
            continue

        if line.startswith("/reject "):
            try:
                bid = int(line.split()[1])
            except (ValueError, IndexError):
                print("  Usage: /reject ID")
                continue
            reason = input("  Reason (required): ").strip()
            if not reason:
                print("  A reason is required.")
                continue
            ok = db.reject_booking(bid, admin_notes=reason)
            if ok:
                booking = db.get_booking(bid)
                notif.notify_booking_rejected(booking)
                print(f"  Reservation #{bid} REJECTED.")
            else:
                print(f"  Could not reject #{bid} (not found or not pending).")
            continue

        if line.startswith("/status "):
            try:
                bid = int(line.split()[1])
            except (ValueError, IndexError):
                print("  Usage: /status ID")
                continue
            b = db.get_booking(bid)
            if not b:
                print(f"  Reservation #{bid} not found.")
            else:
                for k, v in b.items():
                    print(f"  {k}: {v}")
            continue

        # Free-text goes to the admin AI agent
        reply = admin_chat(line)
        print(f"assistant > {reply}")


if __name__ == "__main__":
    sys.exit(main())
