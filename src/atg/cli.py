"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

from atg.bench import DEFAULT_K, DEFAULT_SIZES, full_report, run_compression, run_sweep, taxonomy
from atg.catalog import build_catalog
from atg.render import FORMATS, render
from atg.retrieval.strategies import Indexes
from atg.tasks import build_tasks

OUT = Path("out")


def _sizes(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_SIZES
    return tuple(int(x) for x in raw.split(","))


def cmd_catalog(args: argparse.Namespace) -> int:
    catalog = build_catalog(seed=args.seed)
    if args.show:
        for tool in catalog.tools[: args.show]:
            print(render(tool, args.fmt))
        return 0
    domains: dict[str, int] = {}
    servers: dict[str, int] = {}
    for tool in catalog.tools:
        domains[tool.domain] = domains.get(tool.domain, 0) + 1
        servers[tool.server] = servers.get(tool.server, 0) + 1
    print(f"tools          {len(catalog)}")
    print(f"fingerprint    {catalog.fingerprint()}")
    print(f"seed           {catalog.seed}")
    print(f"domains        {len(domains)}")
    print(f"servers        {len(servers)}")
    print(f"families       {len(catalog.families)}")
    print(f"aliases        {sum(1 for t in catalog.tools if t.alias_of)}")
    print(f"preconditioned {sum(1 for t in catalog.tools if t.precondition)}")
    print(f"mean params    {statistics.mean(len(t.params) for t in catalog.tools):.2f}")
    for fmt in FORMATS:
        from atg.render import block_tokens

        print(f"whole catalog as {fmt:<13} {block_tokens(list(catalog.tools), fmt):>9,} tokens")
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    catalog = build_catalog()
    tasks = build_tasks(catalog, n=args.n)
    kinds: dict[str, int] = {}
    for task in tasks:
        kinds[task.kind] = kinds.get(task.kind, 0) + 1
    print(f"{len(tasks)} tasks  {kinds}")
    for task in tasks[: args.show]:
        print(f"\n[{task.kind}] {task.gold_tool}")
        print(f"  {task.query}")
        print(f"  gold args: {task.slot_map}")
        if task.confusable_with:
            print(f"  near duplicates: {', '.join(task.confusable_with)}")
        if task.missing_param:
            print(f"  deliberately absent from the query: {task.missing_param}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    payload = full_report(
        sizes=_sizes(args.sizes),
        k=args.k,
        out=Path(args.out) if args.out else OUT / "results.json",
        progress=None if args.quiet else lambda line: print(line, flush=True),
        trajectory_out=OUT / "sweep.jsonl" if args.trajectory else None,
    )
    print()
    _print_matrix(payload)
    if args.report:
        from atg.report import write_report

        path = write_report(payload, Path(args.report))
        print(f"\nhtml report: {path}")
    return 0


def _print_matrix(payload: dict) -> None:
    sweep = payload["sweep"]
    sizes = sweep["sizes"]
    strategies = sorted({s["strategy"] for s in sweep["summaries"]})
    index = {(s["strategy"], s["catalog_size"]): s for s in sweep["summaries"]}
    for label, key in (("task success", "task_success"), (f"recall@{sweep['k']}", "recall_at_k")):
        print(f"{label}")
        print("  " + "strategy".ljust(13) + "".join(f"{s:>9}" for s in sizes))
        for strategy in strategies:
            cells = "".join(
                f"{index[(strategy, size)][key]:>9.3f}" if (strategy, size) in index else f"{'-':>9}"
                for size in sizes
            )
            print("  " + strategy.ljust(13) + cells)
        print()
    print("mean prompt tokens per query")
    print("  " + "strategy".ljust(13) + "".join(f"{s:>9}" for s in sizes))
    for strategy in strategies:
        cells = "".join(
            f"{index[(strategy, size)]['mean_prompt_tokens']:>9,.0f}" if (strategy, size) in index else f"{'-':>9}"
            for size in sizes
        )
        print("  " + strategy.ljust(13) + cells)
    dis = payload["disagreement_full_catalog"]
    print(
        f"\nat the full catalog: recall hit but task failed {dis['recall_hit_task_fail']}"
        f" ({dis['recall_hit_task_fail_share']:.1%}), recall missed but task passed"
        f" {dis['recall_miss_task_pass']} ({dis['recall_miss_task_pass_share']:.1%})"
    )


def cmd_compress(args: argparse.Namespace) -> int:
    catalog = build_catalog()
    tasks = build_tasks(catalog)
    rows = run_compression(
        catalog=catalog,
        tasks=tasks,
        k=args.k,
        size=args.size,
        strategy=args.strategy,
        indexes=Indexes.build(catalog),
    )
    header = f"{'format':<14}{'tok/query':>11}{'tok/tool':>10}{'whole catalog':>15}{'success':>9}{'saving':>9}{'delta':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.fmt:<14}{row.mean_prompt_tokens:>11,.0f}{row.tokens_per_tool:>10,.0f}"
            f"{row.full_catalog_tokens:>15,}{row.task_success:>9.3f}"
            f"{row.savings_vs_json:>9.1%}{row.success_delta_vs_json:>+8.3f}"
        )
    return 0


def cmd_taxonomy(args: argparse.Namespace) -> int:
    catalog = build_catalog()
    tasks = build_tasks(catalog)
    sweep = run_sweep(
        catalog=catalog, tasks=tasks, sizes=_sizes(args.sizes), k=args.k, indexes=Indexes.build(catalog)
    )
    report = taxonomy(sweep.attempts)
    print(f"{report['total_failures']} failures across {len(sweep.attempts)} attempts\n")
    print(f"{'failure mode':<28}{'count':>8}{'share':>9}")
    print("-" * 45)
    for mode, count in report["overall"].items():
        print(f"{mode:<28}{count:>8}{report['overall_share'][mode]:>9.1%}")
    print("\nby strategy")
    for strategy, modes in report["by_strategy"].items():
        total = sum(modes.values())
        top = sorted(modes.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {strategy:<13} {total:>5} failures   " + ", ".join(f"{m} {c}" for m, c in top))
    print("\nby task kind")
    for kind, modes in report["by_task_kind"].items():
        total = sum(modes.values())
        top = sorted(modes.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {kind:<15} {total:>5} failures   " + ", ".join(f"{m} {c}" for m, c in top))
    return 0


def cmd_gateway(args: argparse.Namespace) -> int:
    from atg.gateway.server import main as gateway_main

    argv = []
    if args.config:
        argv += ["--config", args.config]
    if args.shortlist:
        argv += ["--shortlist", str(args.shortlist)]
    if args.no_retrieval:
        argv += ["--no-retrieval"]
    return gateway_main(argv)


def cmd_calibrate(args: argparse.Namespace) -> int:
    from atg.calibrate import report_calibration

    path = Path(args.fixture)
    if not path.exists():
        print(f"no calibration fixture at {path}", file=sys.stderr)
        return 2
    result = report_calibration(path)
    print(json.dumps(result, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from atg.report import write_report

    payload = json.loads(Path(args.results).read_text())
    path = write_report(payload, Path(args.out))
    print(f"html report: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atg", description="Tool retrieval and routing at scale.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("catalog", help="describe the synthetic tool catalog")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--show", type=int, default=0, help="print this many rendered tools")
    p.add_argument("--fmt", default="signature", choices=FORMATS)
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("tasks", help="describe the benchmark task suite")
    p.add_argument("--n", type=int, default=240)
    p.add_argument("--show", type=int, default=5)
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("bench", help="run the full measurement")
    p.add_argument("--sizes", default=None)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--out", default=None)
    p.add_argument("--report", default=None, help="also write a self-contained HTML report here")
    p.add_argument("--trajectory", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("compress", help="token cost and accuracy of each schema rendering")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--strategy", default="rerank")
    p.set_defaults(func=cmd_compress)

    p = sub.add_parser("taxonomy", help="distribution of retrieval and argument failures")
    p.add_argument("--sizes", default=None)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.set_defaults(func=cmd_taxonomy)

    p = sub.add_parser("gateway", help="run the MCP gateway on stdio")
    p.add_argument("--config", default=None)
    p.add_argument("--shortlist", type=int, default=None)
    p.add_argument("--no-retrieval", action="store_true")
    p.set_defaults(func=cmd_gateway)

    p = sub.add_parser("calibrate", help="token estimator error against the recorded fixture")
    p.add_argument("--fixture", default="tests/fixtures/token_calibration.json")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("report", help="build the HTML report from a results file")
    p.add_argument("--results", default="out/results.json")
    p.add_argument("--out", default="out/report.html")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
