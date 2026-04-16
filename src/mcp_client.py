"""MCP client — calls the SkyPark MCP server after a reservation is confirmed.

``write_confirmed_reservation`` is the single public function. It is
synchronous so it can be called from FastAPI route handlers, the admin CLI,
and the admin LangGraph agent tool — all without changing their signatures.

Flow:
  1. Admin approves a booking (via dashboard, admin CLI, or admin agent).
  2. The approval path calls ``write_confirmed_reservation(booking)``.
  3. This module opens an SSE connection to the MCP server, initialises a
     ClientSession, and invokes the ``write_confirmed_reservation`` tool.
  4. The MCP server appends the entry to ``confirmed_reservations.txt``.
  5. If the MCP server is unreachable, a warning is logged and the function
     returns an error string — the approval itself is NOT rolled back.

Security: the Bearer token (``MCP_SECRET``) is sent in the Authorization
header on every request.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

from .config import settings

log = logging.getLogger(__name__)


async def _call_mcp_tool(booking: dict) -> str:
    """Async implementation — opens SSE transport and calls the tool."""
    headers = {"Authorization": f"Bearer {settings.mcp_secret}"}

    async with sse_client(url=settings.mcp_url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "write_confirmed_reservation",
                arguments={
                    "booking_id":    booking["id"],
                    "name":          f"{booking['first_name']} {booking['last_name']}",
                    "car_number":    booking["car_plate"],
                    "period_start":  booking["start_ts"],
                    "period_end":    booking["end_ts"],
                    "approval_time": booking.get("reviewed_at", ""),
                },
            )

    # result.content is a list of TextContent / other content blocks.
    text = "\n".join(
        c.text for c in result.content if hasattr(c, "text")
    )
    return text or "MCP tool returned no text"


def write_confirmed_reservation(booking: dict) -> str:
    """Synchronous entry point used by all approval paths.

    Runs the async MCP call in a new event loop. Safe to call from sync
    contexts (FastAPI route handlers run in a thread pool by default when
    not declared ``async``).

    Returns the MCP server reply string, or an error message if the server
    is unreachable (so callers can log it without crashing the approval flow).
    """
    try:
        return asyncio.run(_call_mcp_tool(booking))
    except Exception as exc:
        msg = f"MCP write failed (booking #{booking.get('id')}): {exc}"
        log.warning(msg)
        return msg
