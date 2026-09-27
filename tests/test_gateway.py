"""Gateway tests. Every downstream server here is a real subprocess."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from atg.gateway.downstream import (
    CircuitBreaker,
    ServerConfig,
    TokenBucket,
    TtlCache,
    default_servers,
    local_server_command,
)
from atg.gateway.protocol import (
    METHOD_NOT_FOUND,
    RATE_LIMITED,
    UNAUTHORIZED,
    UPSTREAM_UNAVAILABLE,
    RpcError,
    parse_request,
)
from atg.gateway.server import InProcessClient, build_gateway

TOKEN = "test-token-not-a-real-credential"


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("ATG_MAILROOM_TOKEN", TOKEN)
    gw = build_gateway()
    yield gw
    gw.stop()


@pytest.fixture
def client(gateway):
    return InProcessClient(gateway)


# -- protocol --------------------------------------------------------------


def test_parse_request_rejects_non_jsonrpc() -> None:
    with pytest.raises(RpcError):
        parse_request("{}")
    with pytest.raises(RpcError):
        parse_request("not json")
    with pytest.raises(RpcError):
        parse_request(json.dumps({"jsonrpc": "2.0", "method": "x", "params": []}))


def test_parse_request_accepts_a_notification() -> None:
    request = parse_request(json.dumps({"jsonrpc": "2.0", "method": "ping"}))
    assert request.is_notification


# -- multiplexing ----------------------------------------------------------


def test_both_downstream_servers_come_up_and_advertise_tools(gateway) -> None:
    assert len(gateway.servers) == 2
    assert all(s.alive for s in gateway.servers)
    assert {s.config.name for s in gateway.servers} == {"mailroom", "warehouse"}
    assert len(gateway.catalog or ()) == 9


def test_the_client_surface_is_smaller_than_the_estate(client, gateway) -> None:
    exposed = client.list_tools()
    assert exposed == ["find_tools", "describe_tool", "call_tool", "gateway_health"]
    assert len(exposed) < len(gateway.catalog or ())


def test_disabling_retrieval_exposes_everything_instead(monkeypatch) -> None:
    monkeypatch.setenv("ATG_MAILROOM_TOKEN", TOKEN)
    gw = build_gateway(use_retrieval=False)
    try:
        assert len(InProcessClient(gw).list_tools()) == 4 + len(gw.catalog or ())
    finally:
        gw.stop()


def test_tool_names_are_qualified_by_server(gateway) -> None:
    assert gateway.catalog is not None
    for name in gateway.catalog.by_name:
        server, _, rest = name.partition(".")
        assert server in {"mailroom", "warehouse"}
        assert rest


def test_find_tools_returns_the_relevant_server(client) -> None:
    found = client.call_tool("find_tools", {"request": "how much stock is left of SKU-1187", "limit": 2})
    assert found["tools"][0]["name"] == "warehouse.get_stock_level"


def test_find_tools_distinguishes_drafting_from_sending(client) -> None:
    found = client.call_tool(
        "find_tools", {"request": "draft an email to the vendor but do not send it yet", "limit": 3}
    )
    names = [row["name"] for row in found["tools"]]
    assert "mailroom.email_draft_create" in names


def test_find_tools_respects_its_limit(client) -> None:
    assert len(client.call_tool("find_tools", {"request": "anything", "limit": 2})["tools"]) == 2


def test_find_tools_needs_a_request(client) -> None:
    with pytest.raises(RpcError):
        client.call_tool("find_tools", {})


def test_describe_tool_returns_a_full_schema(client) -> None:
    described = client.call_tool("describe_tool", {"name": "warehouse.get_stock_level"})
    assert described["inputSchema"]["required"] == ["sku"]
    assert described["server"] == "warehouse"


def test_describe_tool_rejects_an_unknown_name(client) -> None:
    with pytest.raises(RpcError):
        client.call_tool("describe_tool", {"name": "nope.nope"})


def test_calling_through_reaches_the_downstream_process(client) -> None:
    result = client.call_tool(
        "call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-1187"}}
    )
    payload = json.loads(result["content"][0]["text"])
    assert payload["available"] == 412 - 60


def test_an_unknown_gateway_tool_is_method_not_found(client) -> None:
    with pytest.raises(RpcError) as exc:
        client.call_tool("no_such_tool", {})
    assert exc.value.code == METHOD_NOT_FOUND


# -- auth ------------------------------------------------------------------


def test_the_gateway_holds_the_credential_not_the_client(gateway, client) -> None:
    mailroom = gateway.server_by_name("mailroom")
    assert mailroom is not None and mailroom.config.auth_env == "ATG_MAILROOM_TOKEN"
    warehouse = gateway.server_by_name("warehouse")
    assert warehouse is not None and warehouse.config.auth_env is None
    surface = json.dumps(client.call("tools/list"))
    assert TOKEN not in surface


def test_a_missing_credential_fails_that_server_and_only_that_server(gateway, client, monkeypatch) -> None:
    monkeypatch.delenv("ATG_MAILROOM_TOKEN")
    with pytest.raises(RpcError) as exc:
        client.call_tool("call_tool", {"name": "mailroom.list_mailboxes", "arguments": {}})
    assert exc.value.code == UNAUTHORIZED
    ok = client.call_tool("call_tool", {"name": "warehouse.list_warehouses", "arguments": {}})
    assert not ok["isError"]


def test_a_server_that_will_not_start_does_not_take_the_gateway_down(monkeypatch) -> None:
    monkeypatch.delenv("ATG_MAILROOM_TOKEN", raising=False)
    gw = build_gateway()
    try:
        mailroom = gw.server_by_name("mailroom")
        assert mailroom is not None
        assert mailroom.last_error is not None
        assert mailroom.tools == ()
        warehouse = gw.server_by_name("warehouse")
        assert warehouse is not None and warehouse.alive
        found = InProcessClient(gw).call_tool("find_tools", {"request": "stock level", "limit": 2})
        assert all(row["server"] == "warehouse" for row in found["tools"])
    finally:
        gw.stop()


# -- cache -----------------------------------------------------------------


def test_read_only_calls_are_cached_and_mutating_ones_are_not(client) -> None:
    first = client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-1187"}})
    second = client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-1187"}})
    assert first["_via"] == "upstream"
    assert second["_via"] == "cache"
    a = client.call_tool("call_tool", {"name": "warehouse.adjust_stock_level", "arguments": {"sku": "SKU-1187", "delta": "1"}})
    b = client.call_tool("call_tool", {"name": "warehouse.adjust_stock_level", "arguments": {"sku": "SKU-1187", "delta": "1"}})
    assert a["_via"] == b["_via"] == "upstream"


def test_different_arguments_are_different_cache_entries(client) -> None:
    client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-1187"}})
    other = client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-3341"}})
    assert other["_via"] == "upstream"


def test_ttl_cache_expires() -> None:
    now = {"t": 0.0}
    cache = TtlCache(ttl_s=5.0, clock=lambda: now["t"])
    cache.put("k", 1)
    assert cache.get("k") == 1
    now["t"] = 5.1
    assert cache.get("k") is None


def test_ttl_cache_evicts_the_least_recently_used() -> None:
    cache = TtlCache(ttl_s=100.0, max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1


# -- rate limiting ---------------------------------------------------------


def test_token_bucket_allows_a_burst_then_refills() -> None:
    now = {"t": 0.0}
    bucket = TokenBucket(capacity=3, refill_per_s=1.0, clock=lambda: now["t"])
    assert all(bucket.take() for _ in range(3))
    assert not bucket.take()
    assert bucket.retry_after_s() == pytest.approx(1.0)
    now["t"] = 2.0
    assert bucket.take() and bucket.take()
    assert not bucket.take()


def test_the_gateway_rate_limits_per_server_and_says_when_to_retry(gateway, client) -> None:
    warehouse = gateway.server_by_name("warehouse")
    assert warehouse is not None
    warehouse.bucket.tokens = 2
    calls = 0
    with pytest.raises(RpcError) as exc:
        for i in range(5):
            client.call_tool(
                "call_tool", {"name": "warehouse.adjust_stock_level", "arguments": {"sku": "S", "delta": str(i)}}
            )
            calls += 1
    assert calls == 2
    assert exc.value.code == RATE_LIMITED
    assert exc.value.data and exc.value.data["retry_after_s"] >= 0
    ok = client.call_tool("call_tool", {"name": "mailroom.list_mailboxes", "arguments": {}})
    assert not ok["isError"]


def test_a_cache_hit_does_not_spend_a_rate_limit_token(gateway, client) -> None:
    warehouse = gateway.server_by_name("warehouse")
    assert warehouse is not None
    client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-2290"}})
    before = warehouse.bucket.tokens
    client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-2290"}})
    assert warehouse.bucket.tokens == before


# -- degradation -----------------------------------------------------------


def test_circuit_breaker_opens_then_half_opens() -> None:
    now = {"t": 0.0}
    breaker = CircuitBreaker(threshold=2, cooldown_s=10.0, clock=lambda: now["t"])
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.allow()
    breaker.record_failure()
    assert not breaker.allow() and breaker.state == "open"
    now["t"] = 10.5
    assert breaker.state == "half_open" and breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed" and breaker.trips == 1


def test_a_killed_downstream_server_degrades_rather_than_hanging(gateway, client) -> None:
    mailroom = gateway.server_by_name("mailroom")
    assert mailroom is not None and mailroom.proc is not None
    mailroom.proc.kill()
    mailroom.proc.wait()
    codes = []
    for _ in range(4):
        with pytest.raises(RpcError) as exc:
            client.call_tool("call_tool", {"name": "mailroom.list_mailboxes", "arguments": {}})
        codes.append(exc.value.code)
    assert all(c == UPSTREAM_UNAVAILABLE for c in codes)
    assert mailroom.breaker.state == "open"
    assert mailroom.breaker.trips == 1
    still = client.call_tool("call_tool", {"name": "warehouse.list_warehouses", "arguments": {}})
    assert not still["isError"]


def test_find_tools_marks_a_dead_server_unreachable_instead_of_hiding_it(gateway, client) -> None:
    mailroom = gateway.server_by_name("mailroom")
    assert mailroom is not None and mailroom.proc is not None
    mailroom.proc.kill()
    mailroom.proc.wait()
    found = client.call_tool("find_tools", {"request": "read the vendor message about the renewal", "limit": 5})
    rows = {row["name"]: row["reachable"] for row in found["tools"]}
    assert any(name.startswith("mailroom.") for name in rows)
    assert all(not ok for name, ok in rows.items() if name.startswith("mailroom."))


def test_a_downstream_error_is_surfaced_not_swallowed(monkeypatch) -> None:
    monkeypatch.setenv("ATG_FAIL_MODE", "error")
    config = replace(
        next(c for c in default_servers() if c.name == "warehouse"), rate_limit_per_min=60
    )
    gw = build_gateway([config])
    try:
        with pytest.raises(RpcError) as exc:
            InProcessClient(gw).call_tool(
                "call_tool", {"name": "warehouse.list_warehouses", "arguments": {}}
            )
        assert "simulated downstream failure" in exc.value.message
    finally:
        gw.stop()


def test_health_reports_everything_an_operator_needs(gateway, client) -> None:
    client.call_tool("call_tool", {"name": "warehouse.list_warehouses", "arguments": {}})
    health = client.call_tool("gateway_health")
    assert health["downstream_tools"] == 9
    assert health["exposed_tools"] == 4
    per_server = {s["server"]: s for s in health["servers"]}
    assert per_server["warehouse"]["calls"] >= 1
    assert per_server["warehouse"]["breaker"] == "closed"
    for row in per_server.values():
        assert set(row) >= {"alive", "calls", "errors", "breaker", "cache_hits", "rate_tokens_left"}


def test_a_server_with_no_tools_leaves_the_gateway_usable(monkeypatch) -> None:
    monkeypatch.setenv("ATG_MAILROOM_TOKEN", TOKEN)
    broken = ServerConfig(name="broken", command=local_server_command("atg.gateway.demo_servers.nope"))
    gw = build_gateway([broken])
    try:
        assert len(gw.catalog or ()) == 0
        found = InProcessClient(gw).call_tool("find_tools", {"request": "anything"})
        assert found["tools"] == []
    finally:
        gw.stop()
