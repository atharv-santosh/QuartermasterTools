"""The shared trajectory event schema. See docs/trajectory-schema.md."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

EventType = Literal[
    "run_start",
    "llm_call",
    "llm_result",
    "tool_call",
    "tool_result",
    "decision",
    "human_gate",
    "error",
    "run_end",
]

EVENT_TYPES: tuple[EventType, ...] = (
    "run_start",
    "llm_call",
    "llm_result",
    "tool_call",
    "tool_result",
    "decision",
    "human_gate",
    "error",
    "run_end",
)


@dataclass(frozen=True, slots=True)
class Event:
    run_id: str
    seq: int
    ts: float
    type: EventType
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {"run_id": self.run_id, "seq": self.seq, "ts": self.ts, "type": self.type, "payload": self.payload},
            separators=(",", ":"),
            sort_keys=False,
        )

    @classmethod
    def from_json(cls, line: str) -> "Event":
        obj = json.loads(line)
        if set(obj) != {"run_id", "seq", "ts", "type", "payload"}:
            raise ValueError(f"trajectory line has keys {sorted(obj)}")
        if obj["type"] not in EVENT_TYPES:
            raise ValueError(f"unknown event type {obj['type']!r}")
        return cls(obj["run_id"], obj["seq"], obj["ts"], obj["type"], obj["payload"])


@dataclass(slots=True)
class Trajectory:
    """An append-only event list for one run, writable as JSONL."""

    run_id: str
    events: list[Event] = field(default_factory=list)
    clock: Any = time.time

    def emit(self, type: EventType, **payload: Any) -> Event:
        event = Event(self.run_id, len(self.events), self.clock(), type, payload)
        self.events.append(event)
        return event

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(e.to_json() for e in self.events) + "\n")

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)


def read_trajectory(path: Path) -> list[Event]:
    return [Event.from_json(line) for line in path.read_text().splitlines() if line.strip()]


def fixed_clock(start: float = 1_790_000_000.0, step: float = 0.05) -> Any:
    """A monotonic fake clock, so recorded trajectories diff cleanly."""
    state = {"t": start}

    def now() -> float:
        state["t"] += step
        return round(state["t"], 6)

    return now
