"""Just enough MCP: JSON-RPC 2.0 framed as newline-delimited JSON over a pipe.

The real protocol is larger than this. What the gateway needs is `initialize`,
`tools/list` and `tools/call`, plus error objects, and implementing that directly
keeps the demo runnable with no dependencies and no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, BinaryIO

PROTOCOL_VERSION = "2025-06-18"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UPSTREAM_UNAVAILABLE = -32010
RATE_LIMITED = -32011
UNAUTHORIZED = -32012


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_obj(self) -> dict[str, Any]:
        err: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            err["data"] = self.data
        return err


@dataclass(frozen=True, slots=True)
class Request:
    id: Any
    method: str
    params: dict[str, Any]

    @property
    def is_notification(self) -> bool:
        return self.id is None


def parse_request(line: str) -> Request:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise RpcError(PARSE_ERROR, f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict) or obj.get("jsonrpc") != "2.0" or "method" not in obj:
        raise RpcError(INVALID_REQUEST, "not a JSON-RPC 2.0 request")
    # JSON-RPC allows positional params; MCP never uses them. Check the value as
    # given, since `or {}` would let an empty array through as an empty object.
    raw_params = obj.get("params")
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise RpcError(INVALID_PARAMS, "params must be an object")
    return Request(obj.get("id"), str(obj["method"]), raw_params)


def response(request_id: Any, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}, separators=(",", ":"))


def error_response(request_id: Any, error: RpcError) -> str:
    return json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "error": error.to_obj()}, separators=(",", ":")
    )


def request_line(request_id: Any, method: str, params: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        payload["params"] = params
    return json.dumps(payload, separators=(",", ":"))


def write_line(stream: BinaryIO, line: str) -> None:
    stream.write((line + "\n").encode())
    stream.flush()


def text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    """The `tools/call` result shape."""
    return {"content": [{"type": "text", "text": text}], "isError": is_error}
