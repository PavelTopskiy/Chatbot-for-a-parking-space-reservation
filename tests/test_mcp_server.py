"""Unit tests for the reservation writer (no network / no mcp package required).

Tests target src.reservation_writer which contains the pure file I/O logic
used by the MCP tool. The MCP transport layer is tested end-to-end manually.
"""
from __future__ import annotations

from src.reservation_writer import write_reservation_entry


def test_write_creates_file(tmp_path):
    """write_reservation_entry creates the file and writes a formatted line."""
    out = tmp_path / "reservations.txt"
    result = write_reservation_entry(
        booking_id=42,
        name="John Doe",
        car_number="AA-1234-BB",
        period_start="2026-04-20T09:00",
        period_end="2026-04-20T18:00",
        approval_time="2026-04-20T08:30:00Z",
        out_path=str(out),
    )

    assert "John Doe" in result   # entry string contains the name
    assert out.exists()
    content = out.read_text()
    assert "John Doe" in content
    assert "AA-1234-BB" in content
    assert "2026-04-20T09:00" in content
    assert "2026-04-20T18:00" in content
    assert "2026-04-20T08:30:00Z" in content
    # Verify pipe-separated format: Name | Plate | Period | Approval
    parts = content.strip().split(" | ")
    assert len(parts) == 4


def test_write_sanitises_pipes(tmp_path):
    """Pipe characters in input are replaced with '/' to preserve format."""
    out = tmp_path / "reservations.txt"
    write_reservation_entry(
        booking_id=1,
        name="John | Doe",
        car_number="AA|123",
        period_start="2026-05-01T10:00",
        period_end="2026-05-01T12:00",
        approval_time="2026-05-01T09:00Z",
        out_path=str(out),
    )
    content = out.read_text()
    # Exactly 3 delimiter pipes (separating 4 fields)
    assert content.count(" | ") == 3
    # Injected pipes replaced with /
    assert "John / Doe" in content
    assert "AA/123" in content


def test_write_multiple_appends(tmp_path):
    """Multiple calls append separate lines — one per booking."""
    out = tmp_path / "reservations.txt"
    for i in range(3):
        write_reservation_entry(
            booking_id=i,
            name=f"Guest {i}",
            car_number=f"PLATE-{i}",
            period_start="2026-06-01T10:00",
            period_end="2026-06-01T11:00",
            approval_time="2026-06-01T09:00Z",
            out_path=str(out),
        )
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    assert "Guest 0" in lines[0]
    assert "Guest 2" in lines[2]


def test_write_raises_on_empty_field(tmp_path):
    """ValueError is raised when a required field is blank."""
    import pytest
    out = tmp_path / "reservations.txt"
    with pytest.raises(ValueError):
        write_reservation_entry(
            booking_id=1,
            name="",          # empty — should raise
            car_number="XY-999",
            period_start="2026-06-01T10:00",
            period_end="2026-06-01T12:00",
            approval_time="2026-06-01T09:00Z",
            out_path=str(out),
        )
