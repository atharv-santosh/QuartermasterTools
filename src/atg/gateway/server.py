"""The gateway: several downstream MCP servers behind one endpoint, with retrieval.

A client that speaks plain MCP sees a small tool surface. It gets a fixed set of
gateway tools, and `find_tools` returns the handful of downstream tools that match
a request, which the client can then call through `tools/call` as if they were
local. The retrieval layer from `atg.retrieval` decides which handful.

The alternative, exposing everything downstream in `tools/list`, is available with
`--no-retrieval`, so the difference in tool surface is something you can see
rather than a claim.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from atg.catalog import Catalog, Param, Tool
from atg.events import Trajectory
from atg.gateway.downstream import DownstreamServer, ServerConfig, default_servers
from atg.gateway.protocol import (
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    PROTOCOL_VERSION,
    RpcError,
    error_response,
    parse_request,
    response,
    text_result,
)
from atg.render import Format, render
from atg.retrieval.bm25 import BM25Index
from atg.retrieval.embed import EmbeddingIndex, fit_hashing_embedder
from atg.retrieval.strategies import RerankStrategy

GATEWAY_NAME = "atg-gateway"
DEFAULT_SHORTLIST = 5

_JSON_TO_KIND = {
    "integer": "int",
    "number": "money",
    "boolean": "bool",
    "string": "text",
}


def _param_from_schema(name: str, spec: dict[str, Any], required: bool) -> Param:
    kind = "enum" if spec.get("enum") else _JSON_TO_KIND.get(spec.get("type", "string"), "text")
    fmt = spec.get("format")
    if fmt == "email":
        kind = "email"
    elif fmt == "uri":
        kind = "url"
    elif fmt == "date-time":
        kind = "date"
    return Param(name, kind, required, spec.get("description", ""), tuple(spec.get("enum", ())))


def catalog_from_servers(servers: Sequence[DownstreamServer]) -> tuple[Catalog, dict[str, str]]:
    """Build a retrieval catalog from what the downstream servers advertise.

    Returns the catalog and a map from qualified tool name to owning server. Names
    are qualified with the server so two servers exposing `send_email` do not
    collide, which is the first thing that happens in a real estate.
    """
    tools: list[Tool] = []
    owner: dict[str, str] = {}
    for server in servers:
        for advertised in server.tools:
            schema = advertised.get("inputSchema") or {}
            props: dict[str, Any] = schema.get("properties") or {}
            required = set(schema.get("required") or ())
            qualified = f"{server.config.name}.{advertised['name']}"
            owner[qualified] = server.config.name
            description = advertised.get("description", "")
            tools.append(
                Tool(
                    name=qualified,
                    server=server.config.name,
                    domain=server.config.name,
                    resource=advertised["name"],
                    verb=advertised["name"].split("_")[0],
                    summary=description.split(".")[0] + "." if description else advertised["name"],
                    description=description,
                    params=tuple(
                        _param_from_schema(n, s, n in required) for n, s in props.items()
                    ),
                    family=f"{server.config.name}.{advertised['name'].split('_')[-1]}",
                    tags=(server.config.name,),
                )
            )
    return Catalog(tools=tuple(tools), seed=0), owner


# --------------------------------------------------------------------------


@dataclass(slots=True)
class Gateway:
    """Owns the downstream connections, the retrieval index and the request loop."""

    configs: tuple[ServerConfig, ...]
    shortlist: int = DEFAULT_SHORTLIST
    fmt: Format = "signature"
    use_retrieval: bool = True
    trajectory: Trajectory | None = None
    servers: list[DownstreamServer] = field(default_factory=list)
    catalog: Catalog | None = None
    owner: dict[str, str] = field(default_factory=dict)
    strategy: RerankStrategy | None = None
    started_at: float = field(default_factory=time.time)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        for config in self.configs:
            server = DownstreamServer(config)
            try:
                server.start()
            except Exception as exc:
                # A downstream server that will not come up must not stop the
                # gateway. Its tools are simply absent and its health says why.
                server.last_error = str(exc)
                server.errors += 1
                server.breaker.record_failure()
            self.servers.append(server)
        self._reindex()

    def _reindex(self) -> None:
        self.catalog, self.owner = catalog_from_servers(self.servers)
        if len(self.catalog) == 0:
            self.strategy = None
            return
        embedder = fit_hashing_embedder(self.catalog)
        self.strategy = RerankStrategy(
            self.catalog,
            BM25Index(self.catalog),
            EmbeddingIndex(self.catalog, embedder),
            fmt=self.fmt,
        )

    def stop(self) -> None:
        for server in self.servers:
            server.stop()

    def server_by_name(self, name: str) -> DownstreamServer | None:
        return next((s for s in self.servers if s.config.name == name), None)

    # -- the tool surface --------------------------------------------------

    def gateway_tools(self) -> list[dict[str, Any]]:
        tools = [
            {
                "name": "find_tools",
                "description": (
                    "Find the downstream tools that fit a request. Returns their names and "
                    "signatures, which can then be called with call_tool. Call this first."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "request": {"type": "string", "description": "What you are trying to do."},
                        "limit": {"type": "integer", "description": "How many tools to return."},
                    },
                    "required": ["request"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "describe_tool",
                "description": "Full JSON Schema for one downstream tool, by qualified name.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "call_tool",
                "description": "Call a downstream tool by qualified name with its arguments.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Qualified name, server.tool."},
                        "arguments": {"type": "object", "description": "Arguments for the tool."},
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "gateway_health",
                "description": "Per-server liveness, cache, rate limit and circuit breaker state.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        ]
        if not self.use_retrieval:
            assert self.catalog is not None
            tools.extend(
                {
                    "name": tool.name,
                    "description": tool.full_description(),
                    "inputSchema": tool.json_schema(),
                }
                for tool in self.catalog.tools
            )
        return tools

    # -- gateway tool implementations --------------------------------------

    def find_tools(self, request: str, limit: int | None = None) -> dict[str, Any]:
        if self.strategy is None or self.catalog is None:
            return {"tools": [], "note": "no downstream tools are reachable"}
        k = limit or self.shortlist
        result = self.strategy.retrieve(request, k)
        rows = [
            {
                "name": name,
                "server": self.owner.get(name, "?"),
                "signature": render(self.catalog.get(name), self.fmt),
                "reachable": self._reachable(name),
            }
            for name in result.candidates
        ]
        if self.trajectory is not None:
            self.trajectory.emit(
                "decision",
                kind="find_tools",
                request=request,
                k=k,
                candidates=[r["name"] for r in rows],
                catalog_size=len(self.catalog),
                prompt_tokens=result.prompt_tokens,
            )
        return {"tools": rows, "catalog_size": len(self.catalog), "shortlist": k}

    def _reachable(self, qualified: str) -> bool:
        server = self.server_by_name(self.owner.get(qualified, ""))
        return bool(server and server.alive and server.breaker.allow())

    def describe_tool(self, name: str) -> dict[str, Any]:
        if self.catalog is None or name not in self.catalog.by_name:
            raise RpcError(INVALID_PARAMS, f"no such tool {name!r}")
        tool = self.catalog.get(name)
        return {"name": name, "server": tool.server, **tool.as_mcp_tool()}

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.catalog is None or name not in self.catalog.by_name:
            raise RpcError(INVALID_PARAMS, f"no such tool {name!r}")
        server_name = self.owner[name]
        server = self.server_by_name(server_name)
        if server is None:
            raise RpcError(INVALID_PARAMS, f"no server {server_name!r}")
        downstream_tool = self.catalog.get(name).resource
        if self.trajectory is not None:
            self.trajectory.emit(
                "tool_call", tool=name, server=server_name, args=arguments
            )
        result, how = server.call(downstream_tool, arguments)
        if self.trajectory is not None:
            self.trajectory.emit("tool_result", tool=name, server=server_name, via=how, ok=True)
        payload = dict(result)
        payload["_via"] = how
        return payload

    def health(self) -> dict[str, Any]:
        return {
            "gateway": GATEWAY_NAME,
            "uptime_s": round(time.time() - self.started_at, 2),
            "retrieval": self.use_retrieval,
            "downstream_tools": len(self.catalog or ()),
            "exposed_tools": len(self.gateway_tools()),
            "servers": [s.health() for s in self.servers],
        }

    # -- request dispatch --------------------------------------------------

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "serverInfo": {"name": GATEWAY_NAME, "version": "0.1.0"},
                "capabilities": {"tools": {}},
            }
        if method in ("notifications/initialized", "ping"):
            return {}
        if method == "tools/list":
            return {"tools": self.gateway_tools()}
        if method != "tools/call":
            raise RpcError(METHOD_NOT_FOUND, f"unsupported method {method!r}")

        name = str(params.get("name", ""))
        arguments: dict[str, Any] = params.get("arguments") or {}

        if name == "find_tools":
            request = str(arguments.get("request", ""))
            if not request:
                raise RpcError(INVALID_PARAMS, "find_tools needs a request")
            limit = arguments.get("limit")
            return text_result(json.dumps(self.find_tools(request, int(limit) if limit else None), indent=2))
        if name == "describe_tool":
            return text_result(json.dumps(self.describe_tool(str(arguments.get("name", ""))), indent=2))
        if name == "gateway_health":
            return text_result(json.dumps(self.health(), indent=2))
        if name == "call_tool":
            target = str(arguments.get("name", ""))
            return text_result(json.dumps(self.call_tool(target, arguments.get("arguments") or {})))
        # Direct passthrough, only reachable with retrieval disabled.
        if self.catalog is not None and name in self.catalog.by_name:
            return text_result(json.dumps(self.call_tool(name, arguments)))
        raise RpcError(METHOD_NOT_FOUND, f"gateway has no tool {name!r}")

    def serve(self, stdin: TextIO, stdout: TextIO) -> None:
        for raw in stdin:
            raw = raw.strip()
            if not raw:
                continue
            request_id: Any = None
            try:
                request = parse_request(raw)
                request_id = request.id
                result = self.handle(request.method, request.params)
                if request.is_notification:
                    continue
                stdout.write(response(request_id, result) + "\n")
            except RpcError as exc:
                if self.trajectory is not None:
                    self.trajectory.emit("error", code=exc.code, message=exc.message, data=exc.data)
                stdout.write(error_response(request_id, exc) + "\n")
            except Exception as exc:
                stdout.write(error_response(request_id, RpcError(-32603, f"gateway: {exc}")) + "\n")
            stdout.flush()


# --------------------------------------------------------------------------


@dataclass(slots=True)
class InProcessClient:
    """Speaks to a Gateway without a pipe, for tests and the demo script."""

    gateway: Gateway
    _id: int = 0

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._id += 1
        return self.gateway.handle(method, params or {})

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = self.call("tools/call", {"name": name, "arguments": arguments or {}})
        text = result["content"][0]["text"]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def list_tools(self) -> list[str]:
        return [t["name"] for t in self.call("tools/list")["tools"]]


def build_gateway(
    configs: Sequence[ServerConfig] | None = None,
    *,
    shortlist: int = DEFAULT_SHORTLIST,
    use_retrieval: bool = True,
    trajectory: Trajectory | None = None,
) -> Gateway:
    gateway = Gateway(
        configs=tuple(configs if configs is not None else default_servers()),
        shortlist=shortlist,
        use_retrieval=use_retrieval,
        trajectory=trajectory,
    )
    gateway.start()
    return gateway


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the MCP gateway on stdio.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--shortlist", type=int, default=DEFAULT_SHORTLIST)
    parser.add_argument("--no-retrieval", action="store_true")
    args = parser.parse_args(argv)

    configs = None
    if args.config is not None:
        from atg.gateway.downstream import load_config

        configs = load_config(args.config)

    gateway = build_gateway(configs, shortlist=args.shortlist, use_retrieval=not args.no_retrieval)
    try:
        gateway.serve(sys.stdin, sys.stdout)
    finally:
        gateway.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
