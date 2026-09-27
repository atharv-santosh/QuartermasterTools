"""Run the benchmark against real Gemini and record a replayable fixture.

Deliberately small. Vertex quota on this project is shared and has been hit
before, so the script takes an explicit task budget, sleeps between calls, and
writes the fixture incrementally so a quota failure part way through still
leaves usable recorded data rather than nothing.

    python demo/run_gemini.py --n 40 --sizes 25,1236 --strategy rerank
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from atg.agent import run_attempt
from atg.bench import CatalogWindows, summarize
from atg.catalog import build_catalog
from atg.gemini_adapter import (
    DEFAULT_MODEL,
    CallRecorder,
    build_client,
    credentials_available,
    make_gemini_decider,
)
from atg.llm_sim import select_shortlist
from atg.retrieval.strategies import Indexes, build_strategies
from atg.tasks import build_tasks

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="tasks to sample")
    parser.add_argument("--sizes", default="25,1236")
    parser.add_argument("--strategy", default="rerank")
    parser.add_argument("--fmt", default="json_schema")
    parser.add_argument("--convention", default="native", choices=("native", "proxy"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse calls already recorded in the output fixture instead of re-paying for them",
    )
    args = parser.parse_args(argv)

    if not credentials_available():
        print("no ADC credentials; run: gcloud auth application-default login", file=sys.stderr)
        return 2

    catalog = build_catalog()
    tasks = list(build_tasks(catalog))
    random.Random(args.seed).shuffle(tasks)
    sample = tasks[: args.n]
    sizes = [int(s) for s in args.sizes.split(",")]

    indexes = Indexes.build(catalog)
    windows = CatalogWindows(catalog, sample)
    registry = build_strategies(indexes, select_shortlist, fmt=args.fmt)
    strategy = registry[args.strategy]

    tag = f"{args.strategy}-{args.fmt}-{args.convention}"
    out = Path(args.out) if args.out else FIXTURES / f"gemini_{tag}.json"

    recorder = CallRecorder(model=args.model, convention=args.convention)
    cache: dict[str, dict[str, object]] = {}
    if args.resume and out.exists():
        # Quota on this project runs out mid-run. Reloading what a previous
        # attempt already recorded means a resumed run only pays for the calls
        # that are actually missing, and the fixture keeps every call either
        # attempt made rather than being overwritten by a shorter one.
        previous = json.loads(out.read_text())
        for record in previous.get("records", []):
            recorder.add(record)
            cache[record["key"]] = {"name": record.get("name"), "args": record.get("args") or {}}
        usage = previous.get("usage", {})
        recorder.usage.calls = int(usage.get("calls", 0))
        recorder.usage.retries = int(usage.get("retries", 0))
        recorder.usage.prompt_tokens = int(usage.get("prompt_tokens", 0))
        recorder.usage.output_tokens = int(usage.get("output_tokens", 0))
        recorder.usage.total_tokens = int(usage.get("total_tokens", 0))
        print(f"resuming from {len(cache)} recorded calls in {out}", flush=True)

    decider = make_gemini_decider(
        client=build_client(),
        model=args.model,
        convention=args.convention,
        recorder=recorder,
        cache=cache,
    )

    rows: list[dict[str, object]] = []
    try:
        for size in sizes:
            attempts = []
            for i, task in enumerate(sample):
                attempts.append(
                    run_attempt(
                        task,
                        strategy,
                        catalog,
                        decider,
                        k=args.k,
                        allowed=windows.allowed(task, size),
                        catalog_size=size,
                    )
                )
                if args.sleep:
                    time.sleep(args.sleep)
                if (i + 1) % 10 == 0:
                    recorder.write(out)
                    print(f"  size={size} {i + 1}/{len(sample)} calls={recorder.usage.calls}", flush=True)
            s = summarize(attempts)
            rows.append(
                {
                    "size": size,
                    "n": s.n,
                    "task_success": round(s.task_success, 4),
                    "recall_at_k": round(s.recall_at_k, 4),
                    "failures": s.failures,
                    "recall_hit_task_fail": s.recall_hit_task_fail,
                    "recall_miss_task_pass": s.recall_miss_task_pass,
                    "outcomes": dict(Counter(a.outcome for a in attempts)),
                }
            )
            print(
                f"{tag} size={size:<5} n={s.n} success={s.task_success:.3f} "
                f"recall@{args.k}={s.recall_at_k:.3f}",
                flush=True,
            )
    except Exception as exc:
        print(f"stopped early: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        recorder.write(out)
        summary = {
            "tag": tag,
            "model": args.model,
            "convention": args.convention,
            "fmt": args.fmt,
            "strategy": args.strategy,
            "k": args.k,
            "n_sampled": len(sample),
            "sample_seed": args.seed,
            "task_ids": [t.task_id for t in sample],
            "rows": rows,
            "usage": recorder.usage.to_dict(args.model),
        }
        summary_path = out.with_name(out.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["usage"], indent=2))
        print(f"fixture: {out}\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
