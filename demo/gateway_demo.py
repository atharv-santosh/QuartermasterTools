"""Walk the gateway through the things it claims to do, against real subprocesses.

Nothing here is mocked. Both downstream servers are separate OS processes over
pipes, the auth token is a real environment variable the gateway holds and the
client never sees, and the server that goes down is killed rather than stubbed.

    python demo/gateway_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from atg.events import Trajectory, fixed_clock
from atg.gateway.protocol import RpcError
from atg.gateway.server import InProcessClient, build_gateway

OUT = Path(__file__).resolve().parent.parent / "out"


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 62 - len(title)))


def main() -> int:
    os.environ.setdefault("ATG_MAILROOM_TOKEN", "demo-token-not-a-real-credential")
    trajectory = Trajectory(run_id="gateway-demo", clock=fixed_clock())
    trajectory.emit("run_start", kind="gateway_demo")

    gateway = build_gateway(trajectory=trajectory)
    client = InProcessClient(gateway)

    rule("tool surface")
    exposed = client.list_tools()
    downstream = len(gateway.catalog or ())
    print(f"downstream tools behind the gateway: {downstream}")
    print(f"tools the client actually sees:      {len(exposed)}  {exposed}")

    rule("retrieval picks the shortlist")
    for request in (
        "how much stock is left of SKU-1187",
        "draft an email to the vendor about the renewal, do not send it yet",
        "what does express shipping cost out of the eu-west warehouse",
    ):
        found = client.call_tool("find_tools", {"request": request, "limit": 3})
        print(f"\n  {request}")
        for row in found["tools"]:
            print(f"    {row['name']:<34} {row['signature'][:64]}")

    rule("calling through, and the response cache")
    first = client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-1187"}})
    second = client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-1187"}})
    print(f"  first  call via {first['_via']}")
    print(f"  second call via {second['_via']}")
    mutating = client.call_tool(
        "call_tool", {"name": "warehouse.create_purchase_order", "arguments": {"sku": "SKU-1187", "quantity": "24"}}
    )
    print(f"  mutating call via {mutating['_via']} (never cached)")

    rule("per-server auth is held by the gateway")
    mailroom = gateway.server_by_name("mailroom")
    assert mailroom is not None
    print(f"  mailroom requires {mailroom.config.auth_env}, gateway has it: {bool(os.environ.get('ATG_MAILROOM_TOKEN'))}")
    print("  the client never supplies or sees it; warehouse needs no credential at all")
    saved = os.environ.pop("ATG_MAILROOM_TOKEN")
    try:
        client.call_tool("call_tool", {"name": "mailroom.get_message", "arguments": {"message_id": "m_1001"}})
        print("  unexpected: call succeeded without the token")
    except RpcError as exc:
        print(f"  with the token removed: [{exc.code}] {exc.message}")
    finally:
        os.environ["ATG_MAILROOM_TOKEN"] = saved

    rule("rate limiting")
    warehouse = gateway.server_by_name("warehouse")
    assert warehouse is not None
    warehouse.cache.clear()
    warehouse.bucket.tokens = 3
    allowed = 0
    limited = None
    for i in range(6):
        try:
            client.call_tool(
                "call_tool", {"name": "warehouse.adjust_stock_level", "arguments": {"sku": "SKU-1187", "delta": str(i)}}
            )
            allowed += 1
        except RpcError as exc:
            limited = exc
            break
    print(f"  {allowed} calls allowed, then [{limited.code if limited else '-'}] {limited.message if limited else 'no limit hit'}")
    if limited and limited.data:
        print(f"  the error tells the client when to come back: {limited.data}")
    warehouse.bucket.tokens = warehouse.bucket.capacity

    rule("graceful degradation when a downstream server dies")
    assert mailroom.proc is not None
    mailroom.proc.kill()
    mailroom.proc.wait()
    print(f"  killed the mailroom process, alive={mailroom.alive}")
    failures = 0
    for _ in range(4):
        try:
            client.call_tool("call_tool", {"name": "mailroom.list_mailboxes", "arguments": {}})
        except RpcError as exc:
            failures += 1
            last = exc
    print(f"  {failures} calls failed; breaker is now {mailroom.breaker.state} after {mailroom.breaker.trips} trip(s)")
    print(f"  last error: [{last.code}] {last.message}")

    still = client.call_tool("call_tool", {"name": "warehouse.get_carrier_rate", "arguments": {"warehouse": "eu-west", "service": "express"}})
    print(f"  warehouse still serving: {still['content'][0]['text']}")
    found = client.call_tool("find_tools", {"request": "read the vendor's message about the renewal", "limit": 4})
    print("  find_tools keeps returning mailroom tools but marks them unreachable:")
    for row in found["tools"]:
        print(f"    {row['name']:<34} reachable={row['reachable']}")

    rule("health")
    print(json.dumps(gateway.health(), indent=2))

    gateway.stop()
    trajectory.emit("run_end", status="ok")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gateway-demo.jsonl"
    trajectory.write(path)
    print(f"\ntrajectory: {path}  ({len(trajectory.events)} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
