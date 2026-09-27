"""Simulator against real Gemini on exactly the same tasks, offline from the fixture.

The recorded fixture is the model here, so this needs no credentials and gives
the same answer every time. It also checks that the fixture replays: if the
recorded run cannot be reproduced from it, the comparison says so rather than
quietly scoring a partial replay.

    python demo/compare_sim_vs_real.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from atg.agent import run_attempt
from atg.bench import CatalogWindows, summarize
from atg.catalog import build_catalog
from atg.gemini_adapter import make_replay_decider
from atg.llm_sim import decide as sim_decide
from atg.llm_sim import select_shortlist
from atg.retrieval.strategies import Indexes, build_strategies
from atg.tasks import build_tasks

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=str(FIXTURES / "gemini_headline.json"))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    fixture = Path(args.fixture)
    summary_path = fixture.with_name(fixture.stem + "_summary.json")
    if not fixture.exists() or not summary_path.exists():
        print(f"no recorded run at {fixture}", file=sys.stderr)
        return 2
    meta = load_summary(summary_path)

    catalog = build_catalog()
    tasks = list(build_tasks(catalog))
    random.Random(meta["sample_seed"]).shuffle(tasks)
    sample = tasks[: meta["n_sampled"]]
    assert [t.task_id for t in sample] == meta["task_ids"], "task suite moved since the recording"

    indexes = Indexes.build(catalog)
    windows = CatalogWindows(catalog, sample)
    strategy = build_strategies(indexes, select_shortlist, fmt=meta["fmt"])[meta["strategy"]]
    replay = make_replay_decider(fixture, convention=meta["convention"])
    k = meta["k"]

    rows = []
    print(f"{meta['n_sampled']} tasks, {meta['strategy']}, k={k}, {meta['model']}\n")
    header = f"{'size':>6}  {'recall@k':>9}  {'sim success':>12}  {meta['model'] + ' success':>30}  {'gap':>7}"
    print(header)
    print("-" * len(header))
    for row in meta["rows"]:
        size = row["size"]
        allowed = {t.task_id: windows.allowed(t, size) for t in sample}
        sim = summarize(
            [
                run_attempt(t, strategy, catalog, sim_decide, k=k, allowed=allowed[t.task_id], catalog_size=size)
                for t in sample
            ]
        )
        real = summarize(
            [
                run_attempt(t, strategy, catalog, replay, k=k, allowed=allowed[t.task_id], catalog_size=size)
                for t in sample
            ]
        )
        assert real.recall_at_k == sim.recall_at_k, "retrieval is model independent; this should match"
        if round(real.task_success, 4) != row["task_success"]:
            print(
                f"  replay mismatch at size {size}: fixture recorded {row['task_success']:.3f}, "
                f"replay scores {real.task_success:.3f}",
                file=sys.stderr,
            )
        rows.append(
            {
                "size": size,
                "n": sim.n,
                "recall_at_k": round(sim.recall_at_k, 4),
                "sim_task_success": round(sim.task_success, 4),
                "real_task_success": round(real.task_success, 4),
                "gap": round(real.task_success - sim.task_success, 4),
                "sim_failures": sim.failures,
                "real_failures": real.failures,
                "sim_recall_hit_task_fail": sim.recall_hit_task_fail,
                "real_recall_hit_task_fail": real.recall_hit_task_fail,
            }
        )
        print(
            f"{size:>6}  {sim.recall_at_k:>9.3f}  {sim.task_success:>12.3f}  "
            f"{real.task_success:>30.3f}  {real.task_success - sim.task_success:>+7.3f}"
        )

    first, last = rows[0], rows[-1]
    print(
        f"\nover {first['size']} to {last['size']} tools, recall falls "
        f"{first['recall_at_k'] - last['recall_at_k']:.3f}, "
        f"simulator task success falls {first['sim_task_success'] - last['sim_task_success']:.3f}, "
        f"{meta['model']} task success falls {first['real_task_success'] - last['real_task_success']:.3f}"
    )
    payload = {
        "model": meta["model"],
        "convention": meta["convention"],
        "strategy": meta["strategy"],
        "fmt": meta["fmt"],
        "k": k,
        "n": meta["n_sampled"],
        "sample_seed": meta["sample_seed"],
        "usage": meta["usage"],
        "rows": rows,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
