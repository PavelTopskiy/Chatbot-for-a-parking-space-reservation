"""MCP client — calls the SkyPark MCP server after a reservation is confirmed.

This is a plain-``httpx`` implementation of an MCP JSON-RPC client. It does
not use the external ``mcp`` SDK (which requires Python 3.10+), so it runs
on Python 3.9 alongside the matching ``mcp_server.py``.

``write_confirmed_reservation`` is the single public function. It is
synchronous so it can be called from FastAPI route handlers, the admin CLI,
and the admin LangGraph agent tool — all without changing their signatures.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from .config import settings

log = logging.getLogger(__name__)


def _rpc_call(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a single JSON-RPC 2.0 request and return the parsed result."""
    headers = {
        "Authorization": f"Bearer {settings.mcp_secret}",
        "Content-Type": "application/json",
    }
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }
    with httpx.Client(timeout=10.0) as client:
        response = client.post(settings.mcp_url, json=payload, headers=headers)
        response.raise_for_status()
        body = response.json()

    if "error" in body:
        err = body["error"]
        raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
    return body.get("result", {})


def write_confirmed_reservation(booking: dict) -> str:
    """Persist a confirmed booking by calling the MCP server's write tool.

    Returns the MCP server reply string, or an error message if the server
    is unreachable (so callers can log it without crashing the approval flow).
    """
    try:
        result = _rpc_call(
            "tools/call",
            {
                "name": "write_confirmed_reservation",
                "arguments": {
                    "booking_id":    booking["id"],
                    "name":          f"{booking['first_name']} {booking['last_name']}",
                    "car_number":    booking["car_plate"],
                    "period_start":  booking["start_ts"],
                    "period_end":    booking["end_ts"],
                    "approval_time": booking.get("reviewed_at", ""),
                },
            },
        )
    except Exception as exc:
        msg = f"MCP write failed (booking #{booking.get('id')}): {exc}"
        log.warning(msg)
        return msg

    if result.get("isError"):
        content = result.get("content", [])
        text = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        msg = f"MCP tool error: {text}"
        log.warning(msg)
        return msg

    content = result.get("content", [])
    return "\n".join(c.get("text", "") for c in content if isinstance(c, dict)) or "OK"
