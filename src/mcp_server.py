"""SkyPark MCP Server — Stage 3.

Exposes one MCP tool:
  write_confirmed_reservation(booking_id, name, car_number,
                               period_start, period_end, approval_time)

The tool appends a line to ``data/confirmed_reservations.txt`` in the format:
  Name | Car Number | Reservation Period | Approval Time

Security:
  - Every HTTP request must carry ``Authorization: Bearer <MCP_SECRET>``.
  - The secret is loaded from the environment variable ``MCP_SECRET``.
  - Requests without a valid token receive HTTP 401.
  - All input fields are sanitised (pipe characters stripped) before writing.

Transport: SSE (Server-Sent Events) on port 8001 by default.

Usage::

    python3 -m src.mcp_server          # start server
    python3 -m src.mcp_server --help   # show options
"""
from __future__ import annotations

import argparse
import logging

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .config import settings
from .reservation_writer import write_reservation_entry

log = logging.getLogger(__name__)

# ---------- MCP app ----------

mcp = FastMCP(
    "skypark-reservation-writer",
    instructions=(
        "SkyPark Central MCP server. "
        "Use write_confirmed_reservation to persist an approved booking."
    ),
)


@mcp.tool()
def write_confirmed_reservation(
    booking_id: int,
    name: str,
    car_number: str,
    period_start: str,
    period_end: str,
    approval_time: str,
) -> str:
    """Write a confirmed reservation to the persistent log file.

    File format (one line per reservation):
        Name | Car Number | Reservation Period | Approval Time

    Args:
        booking_id:    Internal booking ID.
        name:          Guest full name (first + last).
        car_number:    Vehicle licence plate.
        period_start:  ISO-8601 reservation start timestamp.
        period_end:    ISO-8601 reservation end timestamp.
        approval_time: ISO-8601 timestamp when administrator approved.

    Returns:
        Confirmation string with the written entry.
    """
    entry = write_reservation_entry(
        booking_id=booking_id,
        name=name,
        car_number=car_number,
        period_start=period_start,
        period_end=period_end,
        approval_time=approval_time,
    )
    log.info("MCP: wrote reservation #%s", booking_id)
    return f"OK — reservation #{booking_id} written: {entry}"


# ---------- auth middleware ----------

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Authorization header does not match MCP_SECRET."""

    async def dispatch(self, request: Request, call_next):
        # Allow health-check without token
        if request.url.path == "/health":
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        token = auth.removeprefix("Bearer ").strip()

        if not settings.mcp_secret:
            log.warning("MCP_SECRET is not set — running without auth (unsafe)")
        elif token != settings.mcp_secret:
            log.warning(
                "MCP: rejected request from %s — invalid token",
                request.client.host if request.client else "unknown",
            )
            return Response("Unauthorized", status_code=401)

        return await call_next(request)


# ---------- ASGI app (for direct use by uvicorn / tests) ----------

def build_app():
    """Return the Starlette ASGI app wrapped with auth middleware."""
    base = mcp.sse_app()
    base.add_middleware(BearerAuthMiddleware)
    return base


# ---------- health check ----------

from starlette.routing import Route  # noqa: E402


async def health(request: Request):
    return Response('{"status":"ok","service":"skypark-mcp"}', media_type="application/json")


# ---------- entrypoint ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="SkyPark MCP server")
    parser.add_argument("--host", default=settings.mcp_host)
    parser.add_argument("--port", type=int, default=settings.mcp_port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log.info("Starting SkyPark MCP server on %s:%s", args.host, args.port)
    log.info("Reservations file: %s", settings.reservations_file)
    log.info("Auth: %s", "enabled" if settings.mcp_secret else "DISABLED")

    uvicorn.run(
        "src.mcp_server:build_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
