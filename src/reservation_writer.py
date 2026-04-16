"""Pure file-writing logic for confirmed reservations.

Kept in its own module with zero external dependencies so it can be:
  - called directly by the MCP tool (mcp_server.py)
  - tested without installing the mcp or uvicorn packages
  - reused by any future integration

Format written:  Name | Car Number | Reservation Period | Approval Time
"""
from __future__ import annotations

import fcntl
from pathlib import Path

from .config import settings


def _sanitise(value: str) -> str:
    """Remove pipe chars and control characters to preserve the log format."""
    return value.replace("|", "/").replace("\n", " ").replace("\r", "").strip()


def write_reservation_entry(
    booking_id: int,
    name: str,
    car_number: str,
    period_start: str,
    period_end: str,
    approval_time: str,
    out_path: str | None = None,
) -> str:
    """Append one confirmed-reservation line to the log file.

    Args:
        booking_id:    Internal booking ID.
        name:          Guest full name.
        car_number:    Vehicle licence plate.
        period_start:  ISO-8601 reservation start.
        period_end:    ISO-8601 reservation end.
        approval_time: ISO-8601 timestamp of admin approval.
        out_path:      Override output file path (default: settings.reservations_file).

    Returns:
        The entry string that was written (without the trailing newline).

    Raises:
        ValueError: if any field is empty after sanitisation.
    """
    s_name     = _sanitise(name)
    s_plate    = _sanitise(car_number)
    s_start    = _sanitise(period_start)
    s_end      = _sanitise(period_end)
    s_approved = _sanitise(approval_time)

    if not all([s_name, s_plate, s_start, s_end, s_approved]):
        raise ValueError("All fields are required and must be non-empty.")

    entry = f"{s_name} | {s_plate} | {s_start} - {s_end} | {s_approved}"

    target = Path(out_path or settings.reservations_file)
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(entry + "\n")
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

    return entry
