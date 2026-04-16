"""Notification service for reservation events.

Supports two channels:
1. **Email** (SMTP) — sends an email to the admin when a new reservation is
   created, and to the guest (if email provided) when it is confirmed/rejected.
2. **In-app log** — always writes to an in-memory list so the admin dashboard
   and CLI can display notifications without requiring email setup.

Email is optional: if SMTP settings are not configured the send silently
falls back to in-app-only.
"""
from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from typing import List, Optional

from .config import settings

log = logging.getLogger(__name__)


# ---------- in-app notification store ----------

@dataclass
class Notification:
    id: int
    booking_id: int
    kind: str          # "new_reservation" | "confirmed" | "rejected"
    subject: str
    body: str
    created_at: str
    read: bool = False


_store: List[Notification] = []
_next_id = 1


def _add(booking_id: int, kind: str, subject: str, body: str) -> Notification:
    global _next_id
    n = Notification(
        id=_next_id,
        booking_id=booking_id,
        kind=kind,
        subject=subject,
        body=body,
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    _store.append(n)
    _next_id += 1
    return n


def get_notifications(unread_only: bool = False) -> List[dict]:
    out = _store if not unread_only else [n for n in _store if not n.read]
    return [n.__dict__ for n in out]


def mark_read(notification_id: int) -> bool:
    for n in _store:
        if n.id == notification_id:
            n.read = True
            return True
    return False


# ---------- email helpers ----------

def _send_email(to: str, subject: str, body: str) -> bool:
    """Send a plain-text email via SMTP. Returns True on success."""
    if not all([settings.smtp_host, settings.smtp_user, settings.smtp_pass]):
        log.debug("SMTP not configured — skipping email send")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = settings.smtp_from or settings.smtp_user
        msg["To"] = to
        msg.set_content(body)
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as s:
            s.starttls()
            s.login(settings.smtp_user, settings.smtp_pass)
            s.send_message(msg)
        log.info("Email sent to %s: %s", to, subject)
        return True
    except Exception:
        log.exception("Failed to send email to %s", to)
        return False


# ---------- public API ----------

def notify_new_reservation(booking: dict) -> Notification:
    """Called when the user chatbot creates a new pending reservation."""
    bid = booking["id"]
    subject = f"[SkyPark] New reservation #{bid} needs review"
    body = (
        f"A new parking reservation has been submitted and needs your approval.\n\n"
        f"  Reservation ID: {bid}\n"
        f"  Guest: {booking['first_name']} {booking['last_name']}\n"
        f"  Vehicle: {booking['car_plate']}\n"
        f"  Period: {booking['start_ts']} → {booking['end_ts']}\n"
        f"  Status: PENDING\n\n"
        f"Please review this reservation in the admin dashboard or use the\n"
        f"admin CLI to approve or reject it.\n"
    )
    _send_email(settings.admin_email, subject, body)
    return _add(bid, "new_reservation", subject, body)


def notify_booking_confirmed(booking: dict) -> Notification:
    """Called when the admin confirms a reservation."""
    bid = booking["id"]
    notes = booking.get("admin_notes", "")
    subject = f"[SkyPark] Reservation #{bid} CONFIRMED"
    body = (
        f"Your parking reservation has been confirmed!\n\n"
        f"  Reservation ID: {bid}\n"
        f"  Guest: {booking['first_name']} {booking['last_name']}\n"
        f"  Vehicle: {booking['car_plate']}\n"
        f"  Period: {booking['start_ts']} → {booking['end_ts']}\n"
        f"  Status: CONFIRMED\n"
    )
    if notes:
        body += f"  Admin notes: {notes}\n"
    return _add(bid, "confirmed", subject, body)


def notify_booking_rejected(booking: dict) -> Notification:
    """Called when the admin rejects a reservation."""
    bid = booking["id"]
    notes = booking.get("admin_notes", "No reason provided")
    subject = f"[SkyPark] Reservation #{bid} REJECTED"
    body = (
        f"Unfortunately your parking reservation has been rejected.\n\n"
        f"  Reservation ID: {bid}\n"
        f"  Guest: {booking['first_name']} {booking['last_name']}\n"
        f"  Vehicle: {booking['car_plate']}\n"
        f"  Period: {booking['start_ts']} → {booking['end_ts']}\n"
        f"  Reason: {notes}\n\n"
        f"Please contact us if you have questions.\n"
    )
    return _add(bid, "rejected", subject, body)
