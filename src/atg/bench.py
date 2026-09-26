"""The measurement harness: size sweep, compression sweep, failure taxonomy."""

from __future__ import annotations

import hashlib
import json
import statistics
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from atg.agent import Attempt, Decider, run_attempt
from atg.catalog import Catalog, build_catalog
from atg.events import Trajectory, fixed_clock
from atg.llm_sim import decide as sim_decide
from atg.llm_sim import select_shortlist as sim_select
from atg.render import FORMATS, Format, block_tokens
from atg.retrieval.strategies import Indexes, Strategy, build_strategies
from atg.tasks import FAILURE_ORDER, Task, build_tasks

DEFAULT_SIZES: tuple[int, ...] = (10, 25, 50, 100, 200, 400, 800, 1236)
DEFAULT_K = 8


def _perm_key(task_id: str, name: str) -> int:
    digest = hashlib.blake2b(f"{task_id}\x1f{name}".encode(), digest_size=8).digest()
    return struct.unpack("<Q", digest)[0]


class CatalogWindows:
    """Per-query catalogs, nested across sizes.

    Comparing strategies at catalog size 10 and size 1236 only means something if
    the query is answerable at both, so each task gets its own deterministic
    ordering of the catalog with the tools it needs (the gold tool, its
    acceptable alternatives, its near duplicates) pinned to the front. Size n is
    the first n of that ordering, which makes every size a superset of the one
    below it.
    """

    def __init__(self, catalog: Catalog, tasks: Sequence[Task]) -> None:
        self.catalog = catalog
        self._orders: dict[str, tuple[str, ...]] = {}
        all_names = [t.name for t in catalog.tools]
        for task in tasks:
            pinned = [n for n in task.required_tools() if n in catalog.by_name]
            rest = sorted((n for n in all_names if n not in set(pinned)), key=lambda n: _perm_key(task.task_id, n))
            self._orders[task.task_id] = tuple(pinned + rest)

    def allowed(self, task: Task, size: int) -> frozenset[str]:
        order = self._orders[task.task_id]
        return frozenset(order[: max(size, len(task.required_tools()))])

    def min_size(self, task: Task) -> int:
        return len(task.required_tools())


@dataclass(slots=True)
class Summary:
    strategy: str
    catalog_size: int
    n: int
    task_success: float
    recall_at_k: float
    mean_prompt_tokens: float
    median_prompt_tokens: float
    stages: int
    recall_hit_task_fail: int
    recall_miss_task_pass: int
    failures: dict[str, int] = field(default_factory=dict)


def summarize(attempts: Sequence[Attempt]) -> Summary:
    n = len(attempts)
    assert n, "no attempts to summarize"
    successes = sum(a.success for a in attempts)
    recalls = sum(a.recall_hit for a in attempts)
    failures: dict[str, int] = {k: 0 for k in FAILURE_ORDER}
    for a in attempts:
        if not a.success:
            failures[a.outcome] = failures.get(a.outcome, 0) + 1
    return Summary(
        strategy=attempts[0].strategy,
        catalog_size=attempts[0].catalog_size,
        n=n,
        task_success=successes / n,
        recall_at_k=recalls / n,
        mean_prompt_tokens=statistics.mean(a.prompt_tokens for a in attempts),
        median_prompt_tokens=statistics.median(a.prompt_tokens for a in attempts),
        stages=attempts[0].stages,
        recall_hit_task_fail=sum(1 for a in attempts if a.recall_hit and not a.success),
        recall_miss_task_pass=sum(1 for a in attempts if not a.recall_hit and a.success),
        failures={k: v for k, v in failures.items() if v},
    )


@dataclass(slots=True)
class SweepResult:
    catalog_seed: int
    catalog_size_total: int
    catalog_fingerprint: str
    task_seed: int
    n_tasks: int
    k: int
    sizes: tuple[int, ...]
    fmt: Format
    embedder: str
    decider: str
    summaries: list[Summary] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    elapsed_s: float = 0.0

    def by_strategy(self, strategy: str) -> list[Summary]:
        return [s for s in self.summaries if s.strategy == strategy]

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_seed": self.catalog_seed,
            "catalog_size_total": self.catalog_size_total,
            "catalog_fingerprint": self.catalog_fingerprint,
            "task_seed": self.task_seed,
            "n_tasks": self.n_tasks,
            "k": self.k,
            "sizes": list(self.sizes),
            "fmt": self.fmt,
            "embedder": self.embedder,
            "decider": self.decider,
            "elapsed_s": round(self.elapsed_s, 3),
            "summaries": [asdict(s) for s in self.summaries],
        }


def run_sweep(
    *,
    catalog: Catalog | None = None,
    tasks: Sequence[Task] | None = None,
    sizes: Iterable[int] = DEFAULT_SIZES,
    k: int = DEFAULT_K,
    fmt: Format = "json_schema",
    decider: Decider = sim_decide,
    decider_name: str = "sim",
    strategies: Sequence[str] | None = None,
    indexes: Indexes | None = None,
    trajectory: Trajectory | None = None,
    keep_attempts: bool = True,
    progress: Callable[[str], None] | None = None,
) -> SweepResult:
    """Accuracy and token cost for every strategy at every catalog size."""
    catalog = catalog or build_catalog()
    tasks = tasks or build_tasks(catalog)
    indexes = indexes or Indexes.build(catalog)
    windows = CatalogWindows(catalog, tasks)
    registry = build_strategies(indexes, sim_select, fmt=fmt)
    wanted = list(strategies) if strategies else list(registry)
    size_list = tuple(sorted(set(sizes)))

    started = time.perf_counter()
    result = SweepResult(
        catalog_seed=catalog.seed,
        catalog_size_total=len(catalog),
        catalog_fingerprint=catalog.fingerprint(),
        task_seed=424242,
        n_tasks=len(tasks),
        k=k,
        sizes=size_list,
        fmt=fmt,
        embedder=indexes.embedder_name,
        decider=decider_name,
    )

    for name in wanted:
        strategy: Strategy = registry[name]
        for size in size_list:
            attempts: list[Attempt] = []
            for task in tasks:
                allowed = windows.allowed(task, size)
                attempts.append(
                    run_attempt(
                        task,
                        strategy,
                        catalog,
                        decider,
                        k=k,
                        allowed=allowed,
                        catalog_size=len(allowed),
                        trajectory=trajectory,
                    )
                )
            result.summaries.append(summarize(attempts))
            if keep_attempts:
                result.attempts.extend(attempts)
            if progress:
                s = result.summaries[-1]
                progress(
                    f"{name:<12} size={size:<5} success={s.task_success:.3f} "
                    f"recall@{k}={s.recall_at_k:.3f} tokens={s.mean_prompt_tokens:,.0f}"
                )
    result.elapsed_s = time.perf_counter() - started
    return result


# --------------------------------------------------------------------------
# Compression


@dataclass(slots=True)
class CompressionRow:
    fmt: Format
    strategy: str
    catalog_size: int
    mean_prompt_tokens: float
    tokens_per_tool: float
    task_success: float
    recall_at_k: float
    full_catalog_tokens: int
    savings_vs_json: float = 0.0
    success_delta_vs_json: float = 0.0


def run_compression(
    *,
    catalog: Catalog | None = None,
    tasks: Sequence[Task] | None = None,
    k: int = DEFAULT_K,
    size: int | None = None,
    strategy: str = "rerank",
    decider: Decider = sim_decide,
    indexes: Indexes | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[CompressionRow]:
    """Token cost and accuracy of each schema rendering, everything else fixed."""
    catalog = catalog or build_catalog()
    tasks = tasks or build_tasks(catalog)
    indexes = indexes or Indexes.build(catalog)
    windows = CatalogWindows(catalog, tasks)
    size = size or len(catalog)

    rows: list[CompressionRow] = []
    for fmt in FORMATS:
        registry = build_strategies(indexes, sim_select, fmt=fmt)
        strat = registry[strategy]
        attempts = [
            run_attempt(
                task,
                strat,
                catalog,
                decider,
                k=k,
                allowed=windows.allowed(task, size),
                catalog_size=size,
            )
            for task in tasks
        ]
        summary = summarize(attempts)
        rows.append(
            CompressionRow(
                fmt=fmt,
                strategy=strategy,
                catalog_size=size,
                mean_prompt_tokens=summary.mean_prompt_tokens,
                tokens_per_tool=summary.mean_prompt_tokens / k,
                task_success=summary.task_success,
                recall_at_k=summary.recall_at_k,
                full_catalog_tokens=block_tokens(list(catalog.tools), fmt),
            )
        )
        if progress:
            r = rows[-1]
            progress(
                f"{fmt:<13} tokens/query={r.mean_prompt_tokens:>8,.0f} "
                f"whole-catalog={r.full_catalog_tokens:>9,} success={r.task_success:.3f}"
            )

    base = next(r for r in rows if r.fmt == "json_schema")
    for row in rows:
        row.savings_vs_json = 1.0 - row.mean_prompt_tokens / base.mean_prompt_tokens
        row.success_delta_vs_json = row.task_success - base.task_success
    return rows


# --------------------------------------------------------------------------
# Failure taxonomy


def taxonomy(attempts: Sequence[Attempt], *, sizes: Iterable[int] | None = None) -> dict[str, Any]:
    """Distribution of failure modes, overall and split by strategy and task kind."""
    keep = set(sizes) if sizes else None
    rows = [a for a in attempts if not a.success and (keep is None or a.catalog_size in keep)]
    total = len(rows)
    overall: dict[str, int] = {}
    by_strategy: dict[str, dict[str, int]] = {}
    by_kind: dict[str, dict[str, int]] = {}
    for a in rows:
        overall[a.outcome] = overall.get(a.outcome, 0) + 1
        by_strategy.setdefault(a.strategy, {}).setdefault(a.outcome, 0)
        by_strategy[a.strategy][a.outcome] += 1
        by_kind.setdefault(a.task_kind, {}).setdefault(a.outcome, 0)
        by_kind[a.task_kind][a.outcome] += 1
    return {
        "total_failures": total,
        "overall": {k: overall.get(k, 0) for k in FAILURE_ORDER if overall.get(k)},
        "overall_share": {
            k: round(overall[k] / total, 4) for k in FAILURE_ORDER if overall.get(k)
        }
        if total
        else {},
        "by_strategy": by_strategy,
        "by_task_kind": by_kind,
    }


def disagreements(attempts: Sequence[Attempt]) -> dict[str, Any]:
    """Where recall@k and task success point different directions."""
    rows = list(attempts)
    hit_fail = [a for a in rows if a.recall_hit and not a.success]
    miss_pass = [a for a in rows if not a.recall_hit and a.success]
    by_outcome: dict[str, int] = {}
    for a in hit_fail:
        by_outcome[a.outcome] = by_outcome.get(a.outcome, 0) + 1
    by_kind: dict[str, int] = {}
    for a in miss_pass:
        by_kind[a.task_kind] = by_kind.get(a.task_kind, 0) + 1
    return {
        "n": len(rows),
        "recall_hit_task_fail": len(hit_fail),
        "recall_hit_task_fail_share": round(len(hit_fail) / len(rows), 4) if rows else 0.0,
        "recall_hit_task_fail_by_outcome": by_outcome,
        "recall_miss_task_pass": len(miss_pass),
        "recall_miss_task_pass_share": round(len(miss_pass) / len(rows), 4) if rows else 0.0,
        "recall_miss_task_pass_by_kind": by_kind,
    }


# --------------------------------------------------------------------------


def full_report(
    *,
    sizes: Iterable[int] = DEFAULT_SIZES,
    k: int = DEFAULT_K,
    out: Path | None = None,
    progress: Callable[[str], None] | None = None,
    trajectory_out: Path | None = None,
) -> dict[str, Any]:
    """Run everything the offline measurement claims and return it as one dict."""
    catalog = build_catalog()
    tasks = build_tasks(catalog)
    indexes = Indexes.build(catalog)
    trajectory = Trajectory(run_id="sweep-offline", clock=fixed_clock()) if trajectory_out else None
    if trajectory is not None:
        trajectory.emit(
            "run_start",
            kind="offline_sweep",
            catalog_size=len(catalog),
            catalog_fingerprint=catalog.fingerprint(),
            n_tasks=len(tasks),
            k=k,
        )

    sweep = run_sweep(
        catalog=catalog,
        tasks=tasks,
        sizes=sizes,
        k=k,
        indexes=indexes,
        progress=progress,
        trajectory=trajectory,
    )
    compression = run_compression(
        catalog=catalog, tasks=tasks, k=k, indexes=indexes, progress=progress
    )
    full = max(sweep.sizes)
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sweep": sweep.to_dict(),
        "compression": [asdict(r) for r in compression],
        "taxonomy_full_catalog": taxonomy(sweep.attempts, sizes=[full]),
        "taxonomy_all_sizes": taxonomy(sweep.attempts),
        "disagreement_full_catalog": disagreements(
            [a for a in sweep.attempts if a.catalog_size == full]
        ),
        "disagreement_all_sizes": disagreements(sweep.attempts),
        "per_kind_full_catalog": _per_kind(sweep.attempts, full),
    }
    if trajectory is not None and trajectory_out is not None:
        trajectory.emit("run_end", status="ok", n_attempts=len(sweep.attempts))
        trajectory.write(trajectory_out)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    return payload


def _per_kind(attempts: Sequence[Attempt], size: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for a in attempts:
        if a.catalog_size != size:
            continue
        bucket = out.setdefault(a.strategy, {})
        kind = bucket.setdefault(a.task_kind, {"n": 0, "success": 0, "recall": 0})
        kind["n"] += 1
        kind["success"] += int(a.success)
        kind["recall"] += int(a.recall_hit)
    for strategy in out:
        for kind, row in out[strategy].items():
            row["task_success"] = round(row["success"] / row["n"], 4)
            row["recall_at_k"] = round(row["recall"] / row["n"], 4)
    return out
