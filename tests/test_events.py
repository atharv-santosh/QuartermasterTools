from __future__ import annotations

import json

import pytest

from atg.events import EVENT_TYPES, Event, Trajectory, fixed_clock, read_trajectory


def test_a_line_has_exactly_the_five_shared_keys() -> None:
    t = Trajectory(run_id="r", clock=fixed_clock())
    event = t.emit("run_start", kind="x")
    assert json.loads(event.to_json()).keys() == {"run_id", "seq", "ts", "type", "payload"}


def test_round_trip() -> None:
    t = Trajectory(run_id="r", clock=fixed_clock())
    original = t.emit("tool_call", tool="send_email", args={"to": "a@b.example"})
    assert Event.from_json(original.to_json()) == original


def test_sequence_numbers_are_dense_and_start_at_zero() -> None:
    t = Trajectory(run_id="r", clock=fixed_clock())
    for i in range(6):
        t.emit("decision", i=i)
    assert [e.seq for e in t] == list(range(6))


def test_the_clock_is_monotonic() -> None:
    t = Trajectory(run_id="r", clock=fixed_clock())
    for _ in range(5):
        t.emit("decision")
    stamps = [e.ts for e in t]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_an_unknown_type_is_rejected() -> None:
    line = json.dumps({"run_id": "r", "seq": 0, "ts": 1.0, "type": "invented", "payload": {}})
    with pytest.raises(ValueError):
        Event.from_json(line)


def test_a_missing_or_extra_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        Event.from_json(json.dumps({"run_id": "r", "seq": 0, "ts": 1.0, "type": "error"}))
    with pytest.raises(ValueError):
        Event.from_json(
            json.dumps({"run_id": "r", "seq": 0, "ts": 1.0, "type": "error", "payload": {}, "extra": 1})
        )


def test_the_nine_types_are_the_nine_documented_types() -> None:
    assert set(EVENT_TYPES) == {
        "run_start",
        "llm_call",
        "llm_result",
        "tool_call",
        "tool_result",
        "decision",
        "human_gate",
        "error",
        "run_end",
    }


def test_writing_and_reading_a_file(tmp_path) -> None:
    t = Trajectory(run_id="r", clock=fixed_clock())
    t.emit("run_start", n=1)
    t.emit("llm_call", n_candidates=8)
    t.emit("run_end", status="ok")
    path = tmp_path / "nested" / "run.jsonl"
    t.write(path)
    back = read_trajectory(path)
    assert [e.type for e in back] == ["run_start", "llm_call", "run_end"]
    assert back == t.events


def test_a_truncated_trajectory_is_still_readable(tmp_path) -> None:
    """A run killed mid-flight has no run_end, and that is valid input."""
    t = Trajectory(run_id="r", clock=fixed_clock())
    t.emit("run_start")
    t.emit("tool_call", tool="x")
    path = tmp_path / "partial.jsonl"
    t.write(path)
    back = read_trajectory(path)
    assert back[0].type == "run_start"
    assert not any(e.type == "run_end" for e in back)


def test_the_fixed_clock_makes_two_runs_byte_identical(tmp_path) -> None:
    def build(name: str):
        t = Trajectory(run_id="r", clock=fixed_clock())
        t.emit("run_start")
        t.emit("decision", kind="retrieval", candidates=["a", "b"])
        t.emit("run_end", status="ok")
        path = tmp_path / name
        t.write(path)
        return path.read_bytes()

    assert build("a.jsonl") == build("b.jsonl")


def test_the_gateway_demo_trajectory_matches_the_schema(monkeypatch, tmp_path) -> None:
    from atg.gateway.server import InProcessClient, build_gateway

    monkeypatch.setenv("ATG_MAILROOM_TOKEN", "t")
    trajectory = Trajectory(run_id="gw", clock=fixed_clock())
    trajectory.emit("run_start", kind="gateway_demo")
    gw = build_gateway(trajectory=trajectory)
    try:
        client = InProcessClient(gw)
        client.call_tool("find_tools", {"request": "stock level for SKU-1187"})
        client.call_tool("call_tool", {"name": "warehouse.get_stock_level", "arguments": {"sku": "SKU-1187"}})
    finally:
        gw.stop()
    trajectory.emit("run_end", status="ok")
    path = tmp_path / "gw.jsonl"
    trajectory.write(path)

    events = read_trajectory(path)
    types = [e.type for e in events]
    assert types[0] == "run_start" and types[-1] == "run_end"
    assert "decision" in types and "tool_call" in types and "tool_result" in types
    for event in events:
        assert event.type in EVENT_TYPES
    decision = next(e for e in events if e.type == "decision")
    assert decision.payload["kind"] == "find_tools"
    result = next(e for e in events if e.type == "tool_result")
    assert result.payload["via"] in ("upstream", "cache")
