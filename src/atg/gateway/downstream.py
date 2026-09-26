"""Downstream MCP servers: subprocess transport, auth, cache, rate limit, breaker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from atg.gateway.protocol import (
    PROTOCOL_VERSION,
    RATE_LIMITED,
    UNAUTHORIZED,
    UPSTREAM_UNAVAILABLE,
    RpcError,
    request_line,
    write_line,
)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """One downstream server, as the gateway is configured to reach it."""

    name: str
    command: tuple[str, ...]
    auth_env: str | None = None
    auth_header: str = "Authorization"
    rate_limit_per_min: int = 120
    cache_ttl_s: float = 30.0
    timeout_s: float = 10.0
    cacheable_tools: tuple[str, ...] = ()
    cwd: str | None = None

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "ServerConfig":
        return cls(
            name=obj["name"],
            command=tuple(obj["command"]),
            auth_env=obj.get("auth_env"),
            auth_header=obj.get("auth_header", "Authorization"),
            rate_limit_per_min=int(obj.get("rate_limit_per_min", 120)),
            cache_ttl_s=float(obj.get("cache_ttl_s", 30.0)),
            timeout_s=float(obj.get("timeout_s", 10.0)),
            cacheable_tools=tuple(obj.get("cacheable_tools", ())),
            cwd=obj.get("cwd"),
        )


def load_config(path: Path) -> tuple[ServerConfig, ...]:
    payload = json.loads(path.read_text())
    return tuple(ServerConfig.from_dict(s) for s in payload["servers"])


# --------------------------------------------------------------------------


@dataclass(slots=True)
class TokenBucket:
    """Per-server rate limit. A burst of the full minute's allowance is fine;
    a sustained rate above it is not."""

    capacity: float
    refill_per_s: float
    tokens: float = 0.0
    last: float = field(default_factory=time.monotonic)
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last = self.clock()

    def take(self, n: float = 1.0) -> bool:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_s)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False

    def retry_after_s(self, n: float = 1.0) -> float:
        if self.refill_per_s <= 0:
            return float("inf")
        return max(0.0, (n - self.tokens) / self.refill_per_s)


@dataclass(slots=True)
class TtlCache:
    """Bounded LRU with per-entry expiry, for read-only downstream calls."""

    ttl_s: float
    max_entries: int = 512
    clock: Callable[[], float] = time.monotonic
    _data: "OrderedDict[str, tuple[float, Any]]" = field(default_factory=OrderedDict)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires, value = entry
        if self.clock() >= expires:
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        self._data[key] = (self.clock() + self.ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


@dataclass(slots=True)
class CircuitBreaker:
    """Stops hammering a downstream server that is failing.

    Closed, it passes calls and counts consecutive failures. At the threshold it
    opens for `cooldown_s`, failing fast with a clear error rather than making the
    client wait out a timeout per call. After the cooldown one trial call decides
    whether to close again.
    """

    threshold: int = 3
    cooldown_s: float = 15.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    trips: int = 0

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "open" if self.clock() - self.opened_at < self.cooldown_s else "half_open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold and self.opened_at is None:
            self.opened_at = self.clock()
            self.trips += 1
        elif self.state == "half_open":
            self.opened_at = self.clock()


# --------------------------------------------------------------------------


class DownstreamServer:
    """One downstream MCP server behind a subprocess pipe.

    Auth is resolved here and never forwarded from the client: the gateway holds
    each server's credential, so a client that can reach the gateway still cannot
    read the token for any server behind it.
    """

    def __init__(self, config: ServerConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self.clock = clock
        self.proc: subprocess.Popen[bytes] | None = None
        self.tools: tuple[dict[str, Any], ...] = ()
        self.bucket = TokenBucket(
            capacity=float(config.rate_limit_per_min),
            refill_per_s=config.rate_limit_per_min / 60.0,
            clock=clock,
        )
        self.cache = TtlCache(ttl_s=config.cache_ttl_s, clock=clock)
        self.breaker = CircuitBreaker(clock=clock)
        self.calls = 0
        self.errors = 0
        self.last_error: str | None = None
        self._id = 0
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _auth_token(self) -> str | None:
        if self.config.auth_env is None:
            return None
        token = os.environ.get(self.config.auth_env)
        if not token:
            raise RpcError(
                UNAUTHORIZED,
                f"server {self.config.name!r} needs {self.config.auth_env} in the gateway environment",
            )
        return token

    def start(self) -> None:
        env = dict(os.environ)
        token = self._auth_token()
        if token is not None:
            env["MCP_AUTH_TOKEN"] = token
            env["MCP_AUTH_HEADER"] = self.config.auth_header
        self.proc = subprocess.Popen(
            list(self.config.command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            cwd=self.config.cwd,
        )
        self._rpc("initialize", {"protocolVersion": PROTOCOL_VERSION, "clientInfo": {"name": "atg-gateway"}})
        listed = self._rpc("tools/list", {})
        self.tools = tuple(listed.get("tools", ()))

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
        finally:
            self.proc = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -- transport ---------------------------------------------------------

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RpcError(UPSTREAM_UNAVAILABLE, f"server {self.config.name!r} is not running")
        with self._lock:
            self._id += 1
            write_line(proc.stdin, request_line(self._id, method, params))
            line = proc.stdout.readline()
        if not line:
            raise RpcError(UPSTREAM_UNAVAILABLE, f"server {self.config.name!r} closed the connection")
        obj = json.loads(line.decode())
        if "error" in obj:
            err = obj["error"]
            raise RpcError(int(err.get("code", -32603)), str(err.get("message", "upstream error")))
        return obj.get("result") or {}

    # -- the guarded call path --------------------------------------------

    def cacheable(self, tool: str) -> bool:
        return tool in self.config.cacheable_tools

    def call(self, tool: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Call a downstream tool. Returns (result, how) where how names the path taken."""
        key = json.dumps({"t": tool, "a": arguments}, sort_keys=True)
        if self.cacheable(tool):
            hit = self.cache.get(key)
            if hit is not None:
                return hit, "cache"

        if not self.breaker.allow():
            raise RpcError(
                UPSTREAM_UNAVAILABLE,
                f"server {self.config.name!r} circuit is open after {self.breaker.failures} failures",
                {"server": self.config.name, "state": "open", "trips": self.breaker.trips},
            )
        if not self.bucket.take():
            raise RpcError(
                RATE_LIMITED,
                f"server {self.config.name!r} rate limit of {self.config.rate_limit_per_min}/min exceeded",
                {"server": self.config.name, "retry_after_s": round(self.bucket.retry_after_s(), 3)},
            )

        self._auth_token()
        self.calls += 1
        try:
            result = self._rpc("tools/call", {"name": tool, "arguments": arguments})
        except RpcError as exc:
            self.errors += 1
            self.last_error = exc.message
            self.breaker.record_failure()
            raise
        except Exception as exc:
            self.errors += 1
            self.last_error = str(exc)
            self.breaker.record_failure()
            raise RpcError(UPSTREAM_UNAVAILABLE, f"{self.config.name}: {exc}") from exc

        self.breaker.record_success()
        if self.cacheable(tool):
            self.cache.put(key, result)
        return result, "upstream"

    def health(self) -> dict[str, Any]:
        return {
            "server": self.config.name,
            "alive": self.alive,
            "tools": len(self.tools),
            "calls": self.calls,
            "errors": self.errors,
            "breaker": self.breaker.state,
            "breaker_trips": self.breaker.trips,
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "rate_tokens_left": round(self.bucket.tokens, 2),
            "last_error": self.last_error,
        }


def local_server_command(module: str) -> tuple[str, ...]:
    return (sys.executable, "-m", module)


def default_servers() -> tuple[ServerConfig, ...]:
    """The two demo servers, one of which requires a credential."""
    return (
        ServerConfig(
            name="mailroom",
            command=local_server_command("atg.gateway.demo_servers.mailroom"),
            auth_env="ATG_MAILROOM_TOKEN",
            rate_limit_per_min=60,
            cache_ttl_s=20.0,
            cacheable_tools=("list_mailboxes", "get_message"),
        ),
        ServerConfig(
            name="warehouse",
            command=local_server_command("atg.gateway.demo_servers.warehouse"),
            rate_limit_per_min=120,
            cache_ttl_s=30.0,
            cacheable_tools=("get_stock_level", "list_warehouses", "get_carrier_rate"),
        ),
    )


def tool_names(servers: Sequence[DownstreamServer]) -> list[str]:
    return [t["name"] for s in servers for t in s.tools]
