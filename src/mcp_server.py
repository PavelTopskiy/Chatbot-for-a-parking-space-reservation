"""SkyPark MCP Server — Stage 3 (FastAPI-native implementation).

Implements the Model Context Protocol (JSON-RPC 2.0 over HTTP) without
depending on the external ``mcp`` SDK package, which requires Python 3.10+.
This implementation runs on Python 3.9+ and uses only FastAPI + pydantic.

Protocol:
  Single endpoint:  POST /mcp
  Body:             JSON-RPC 2.0 request
  Methods:
    - initialize                → server info + capabilities
    - tools/list                → list of exposed tools
    - tools/call                → invoke a tool

Exposed tool:
  write_confirmed_reservation(booking_id, name, car_number,
                               period_start, period_end, approval_time)
  → appends one line to ``data/confirmed_reservations.txt``
     in the format:  Name | Car Number | Reservation Period | Approval Time

Security:
  - Every request must carry ``Authorization: Bearer <MCP_SECRET>``.
  - Invalid tokens → HTTP 401.
  - Inputs are sanitised (pipe characters stripped) before writing.
  - Writes use fcntl exclusive file locking.

Usage::

    python3 -m src.mcp_server
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import settings
from .reservation_writer import write_reservation_entry

log = logging.getLogger(__name__)

app = FastAPI(title="SkyPark MCP Server", version="1.0")


# ---------- MCP tool registry ----------

TOOLS = [
    {
        "name": "write_confirmed_reservation",
        "description": (
            "Append a confirmed reservation to the persistent log file. "
            "Format: 'Name | Car Number | Reservation Period | Approval Time'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "booking_id":    {"type": "integer", "description": "Internal booking ID"},
                "name":          {"type": "string",  "description": "Guest full name"},
                "car_number":    {"type": "string",  "description": "Vehicle licence plate"},
                "period_start":  {"type": "string",  "description": "ISO-8601 reservation start"},
                "period_end":    {"type": "string",  "description": "ISO-8601 reservation end"},
                "approval_time": {"type": "string",  "description": "ISO-8601 approval timestamp"},
            },
            "required": [
                "booking_id", "name", "car_number",
                "period_start", "period_end", "approval_time",
            ],
        },
    },
]


def _call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call. Returns an MCP tool-result dict."""
    if name != "write_confirmed_reservation":
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True,
        }
    try:
        entry = write_reservation_entry(**arguments)
        booking_id = arguments.get("booking_id", "?")
        log.info("MCP: wrote reservation #%s", booking_id)
        return {
            "content": [{"type": "text", "text": f"OK — reservation #{booking_id} written: {entry}"}],
            "isError": False,
        }
    except TypeError as exc:
        return {
            "content": [{"type": "text", "text": f"Invalid arguments: {exc}"}],
            "isError": True,
        }
    except ValueError as exc:
        return {
            "content": [{"type": "text", "text": f"Validation error: {exc}"}],
            "isError": True,
        }
    except Exception as exc:
        log.exception("MCP tool call failed")
        return {
            "content": [{"type": "text", "text": f"Internal error: {exc}"}],
            "isError": True,
        }


# ---------- auth ----------

def _check_auth(authorization: Optional[str]) -> None:
    if not settings.mcp_secret:
        # No secret configured — allow anonymous access (dev only).
        log.warning("MCP_SECRET is not set; running without authentication")
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[len("Bearer "):].strip()
    if token != settings.mcp_secret:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


# ---------- JSON-RPC endpoint ----------

def _rpc_error(code: int, message: str, request_id: Any) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "error": {"code": code, "message": message},
        "id": request_id,
    }


def _rpc_result(result: Any, request_id: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "result": result, "id": request_id}


@app.post("/mcp")
async def mcp_rpc(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_rpc_error(-32700, "Parse error", None), status_code=400)

    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}

    if method == "initialize":
        return _rpc_result(
            {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "skypark-reservation-writer",
                    "version": "1.0.0",
                },
                "capabilities": {"tools": {}},
            },
            request_id,
        )

    if method == "tools/list":
        return _rpc_result({"tools": TOOLS}, request_id)

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        if not tool_name:
            return _rpc_error(-32602, "Missing tool name", request_id)
        return _rpc_result(_call_tool(tool_name, arguments), request_id)

    return _rpc_error(-32601, f"Method not found: {method}", request_id)


# ---------- health check ----------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "skypark-mcp", "tools": [t["name"] for t in TOOLS]}


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
        "src.mcp_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
