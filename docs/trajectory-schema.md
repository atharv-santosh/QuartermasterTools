# Trajectory schema (shared with sibling projects)

This project is one of several sharing a common trajectory event vocabulary, so
that any of them can produce and consume the others' execution logs. This
document specifies exactly what `atg` writes.

## Wire format

One JSON object per line, UTF-8, newline terminated:

```json
{"run_id":"sweep-offline","seq":4,"ts":1790000000.25,"type":"tool_call","payload":{"task_id":"t0007","tool":"refund_charge_partial","args":{"charge_id":"c_4820113","amount":"48.50"}}}
```

Every line has exactly these five top-level keys, no more and no fewer:

| key | type | meaning |
|---|---|---|
| `run_id` | string | the run this event belongs to |
| `seq` | int | per-run sequence number, starting at 0, strictly increasing |
| `ts` | float | UNIX timestamp in seconds, fractional |
| `type` | string | one of the nine canonical types |
| `payload` | object | event specific, freely nested JSON |

## Canonical types

```
run_start | llm_call | llm_result | tool_call | tool_result | decision | human_gate | error | run_end
```

`atg.events.EVENT_TYPES` is the authority, and `Event.from_json` rejects any
line whose key set differs or whose type is not in that list, so a malformed
trajectory fails loudly at the boundary rather than half-parsing.

## What this project emits

`atg` has no event vocabulary of its own, so there is no mapping table to
maintain: it emits the nine shared types directly. Two places produce
trajectories.

**The measurement harness** (`atg bench --trajectory`, one run covering the
whole sweep):

| type | when | notable payload keys |
|---|---|---|
| `run_start` | once, first line | `catalog_size`, `catalog_fingerprint`, `n_tasks`, `k` |
| `decision` | a retrieval step resolved a shortlist | `strategy`, `candidates`, `gold_rank`, `prompt_tokens`, `stages` |
| `llm_call` | the model is about to be asked to choose | `n_candidates`, `fmt`, `prompt_tokens` |
| `llm_result` | what it chose | `action` as `{kind, tool, args}` or `{kind: "none"}` |
| `tool_call` | the chosen tool, with arguments | `tool`, `args` |
| `tool_result` | the grader's verdict | `ok`, `outcome` |
| `human_gate` | the agent asked for a value rather than inventing one | `reason`, `tool`, `missing` |
| `error` | the model produced no action at all | `reason` |
| `run_end` | once, last line | `status`, `n_attempts` |

**The gateway** (`demo/gateway_demo.py`), where `decision` carries a
`find_tools` retrieval, `tool_call` and `tool_result` bracket a downstream call
with `server` and `via` (`upstream` or `cache`), and `error` carries the JSON-RPC
`code` for a rate limit, an auth failure or an open circuit.

A `decision` payload always has a `kind` naming which kind of decision it was
(`retrieval` or `find_tools`), so a consumer can filter without guessing.

## Guarantees

* Lines appear in strictly increasing `seq` order within a `run_id`, with no gaps.
* `seq` starts at 0 and equals the event's index in the file.
* The first line of a complete run is `run_start`; the last is `run_end`.
* A truncated file is still valid input. A run killed part way through simply
  has no `run_end`, and that is a legitimate mid-flight export rather than a
  corrupt one.
* `payload` is not schema-locked beyond the table above. Consumers should treat
  unknown payload keys as forward-compatible additions, not errors.

## Determinism

`atg.events.fixed_clock` supplies a monotonic fake clock so that a recorded
trajectory diffs cleanly against a later run of the same code. The offline sweep
and the gateway demo both use it, which is why two runs of `atg bench
--trajectory` produce byte-identical JSONL. Use `time.time` for anything whose
wall-clock timing matters.

## Reading one back

```python
from pathlib import Path
from atg.events import read_trajectory

events = read_trajectory(Path("out/sweep.jsonl"))
misses = [e for e in events if e.type == "decision" and e.payload.get("gold_rank") is None]
```
