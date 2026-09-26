"""Shared stdio loop for the demo downstream servers."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable

from atg.gateway.protocol import (
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    UNAUTHORIZED,
    RpcError,
    error_response,
    parse_request,
    response,
    text_result,
)

Handler = Callable[[dict[str, Any]], str]


def serve(name: str, tools: list[dict[str, Any]], handlers: dict[str, Handler], *, require_auth: bool = False) -> int:
    """Read JSON-RPC lines on stdin, write results on stdout, until EOF.

    Environment switches exist so the gateway's degradation tests can make a
    real subprocess behave badly rather than mocking the failure:
    ATG_FAIL_MODE=crash exits on the first call, =error returns an RPC error,
    =hang sleeps, and ATG_SLOW_MS delays every call.
    """
    fail_mode = os.environ.get("ATG_FAIL_MODE", "")
    slow_ms = float(os.environ.get("ATG_SLOW_MS", "0"))
    token = os.environ.get("MCP_AUTH_TOKEN")

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = parse_request(raw)
        except RpcError as exc:
            sys.stdout.write(error_response(None, exc) + "\n")
            sys.stdout.flush()
            continue

        try:
            if request.method == "initialize":
                if require_auth and not token:
                    raise RpcError(UNAUTHORIZED, f"{name}: MCP_AUTH_TOKEN not supplied")
                result: Any = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": {"name": name, "version": "0.1.0"},
                    "capabilities": {"tools": {}},
                }
            elif request.method == "tools/list":
                result = {"tools": tools}
            elif request.method == "tools/call":
                if slow_ms:
                    time.sleep(slow_ms / 1000.0)
                if fail_mode == "crash":
                    sys.exit(1)
                if fail_mode == "hang":
                    time.sleep(3600)
                if fail_mode == "error":
                    raise RpcError(-32000, f"{name}: simulated downstream failure")
                tool = str(request.params.get("name", ""))
                handler = handlers.get(tool)
                if handler is None:
                    raise RpcError(METHOD_NOT_FOUND, f"{name} has no tool {tool!r}")
                args = request.params.get("arguments") or {}
                result = text_result(handler(args))
            elif request.method in ("notifications/initialized", "ping"):
                result = {}
            else:
                raise RpcError(METHOD_NOT_FOUND, f"unsupported method {request.method!r}")
            sys.stdout.write(response(request.id, result) + "\n")
        except RpcError as exc:
            sys.stdout.write(error_response(request.id, exc) + "\n")
        except Exception as exc:
            sys.stdout.write(error_response(request.id, RpcError(-32603, f"{name}: {exc}")) + "\n")
        sys.stdout.flush()
    return 0


def string_schema(**props: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
        "required": list(props),
        "additionalProperties": False,
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True)
