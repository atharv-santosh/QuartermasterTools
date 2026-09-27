from __future__ import annotations

import pytest

from atg.agent import run_attempt
from atg.bench import CatalogWindows, disagreements, run_compression, run_sweep, summarize, taxonomy
from atg.catalog import build_catalog
from atg.events import Trajectory, fixed_clock
from atg.llm_sim import decide, extract_mentions, select_shortlist
from atg.render import FORMATS, view
from atg.retrieval.strategies import Indexes, build_strategies
from atg.tasks import Clarify, ToolCall, build_tasks


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


@pytest.fixture(scope="module")
def tasks(catalog):
    return build_tasks(catalog, n=90)


@pytest.fixture(scope="module")
def indexes(catalog):
    return Indexes.build(catalog)


# -- the simulated model ---------------------------------------------------


def test_the_model_only_sees_the_rendering(catalog) -> None:
    """A view carries no reference back to the Tool, so nothing the rendering
    dropped can leak into the decision."""
    v = view(catalog.get("refund_charge_partial"), "oneline")
    assert not hasattr(v, "tool")
    assert v.rendered.count("\n") == 0
    assert not v.description


def test_decide_is_deterministic(catalog, tasks) -> None:
    registry_views = [view(t, "json_schema") for t in catalog.tools[:8]]
    query = tasks[0].query
    first, second = decide(registry_views, query), decide(registry_views, query)
    assert type(first) is type(second)
    if isinstance(first, ToolCall) and isinstance(second, ToolCall):
        assert (first.tool, first.args) == (second.tool, second.args)


def test_decide_returns_nothing_with_no_candidates() -> None:
    assert decide([], "anything") is None


def test_the_model_copies_values_out_of_the_query(catalog, tasks) -> None:
    """On an easy task with the gold tool alone in the shortlist, arguments must
    come from the query rather than be invented."""
    easy = [t for t in tasks if t.kind == "plain"][:20]
    copied = 0
    for task in easy:
        action = decide([view(catalog.get(task.gold_tool), "json_schema")], task.query)
        if isinstance(action, ToolCall):
            for name, value in action.args.items():
                if task.slot_map.get(name) == value:
                    copied += 1
    assert copied > 0


def test_mentions_do_not_include_fragments_of_a_url() -> None:
    mentions = extract_mentions("Push 'a note' through the webhook at https://cb.northwind.example/ingest.")
    values = {m.value for m in mentions}
    assert "https://cb.northwind.example/ingest" in values
    assert not any(v != "https://cb.northwind.example/ingest" and v in "cb.northwind.example/ingest" for v in values)


def test_mentions_keep_the_words_that_introduce_them() -> None:
    mentions = extract_mentions("Drop a note in #ops-alerts saying 'deploy is frozen'.")
    channel = next(m for m in mentions if m.value == "#ops-alerts")
    assert "in" in channel.context


def test_mentions_do_not_split_a_quoted_enum_out_of_its_sentence() -> None:
    values = {m.value for m in extract_mentions("Change the account setting 'priority=urgent'.")}
    assert "priority=urgent" in values
    assert "urgent" not in values


def test_select_shortlist_respects_its_budget(catalog) -> None:
    lines = [(t.name, t.summary) for t in catalog.tools[:200]]
    assert len(select_shortlist("refund a charge", lines, 6)) == 6


# -- one attempt -----------------------------------------------------------


def test_an_attempt_records_what_it_needs_to_be_graded(catalog, tasks, indexes) -> None:
    strategy = build_strategies(indexes, select_shortlist)["rerank"]
    attempt = run_attempt(tasks[0], strategy, catalog, decide, k=8)
    assert attempt.task_id == tasks[0].task_id
    assert attempt.k == 8
    assert attempt.prompt_tokens > 0
    assert attempt.recall_hit == (attempt.gold_rank is not None and attempt.gold_rank <= 8)
    assert attempt.success == (attempt.outcome == "success")


def test_an_attempt_emits_a_well_formed_trajectory(catalog, tasks, indexes) -> None:
    strategy = build_strategies(indexes, select_shortlist)["rerank"]
    trajectory = Trajectory(run_id="t", clock=fixed_clock())
    run_attempt(tasks[0], strategy, catalog, decide, k=8, trajectory=trajectory)
    types = [e.type for e in trajectory]
    assert types[0] == "decision"
    assert "llm_call" in types and "llm_result" in types
    assert types[-1] in ("tool_result", "human_gate", "error")
    assert [e.seq for e in trajectory] == list(range(len(trajectory.events)))


def test_a_clarification_is_journalled_as_a_human_gate(catalog, tasks, indexes) -> None:
    strategy = build_strategies(indexes, select_shortlist)["rerank"]
    for task in tasks:
        trajectory = Trajectory(run_id="t", clock=fixed_clock())
        attempt = run_attempt(task, strategy, catalog, decide, k=8, trajectory=trajectory)
        if attempt.action_tool and any(e.type == "human_gate" for e in trajectory):
            gate = next(e for e in trajectory if e.type == "human_gate")
            assert gate.payload["missing"]
            return
    pytest.skip("no clarification arose in this sample")


# -- the sweep -------------------------------------------------------------


def test_catalog_windows_are_nested_and_always_contain_what_the_task_needs(catalog, tasks) -> None:
    windows = CatalogWindows(catalog, tasks)
    for task in tasks[:30]:
        small = windows.allowed(task, 25)
        large = windows.allowed(task, 200)
        assert small <= large
        for name in task.required_tools():
            assert name in small


def test_a_window_is_the_size_asked_for_unless_the_task_needs_more(catalog, tasks) -> None:
    windows = CatalogWindows(catalog, tasks)
    for task in tasks[:20]:
        assert len(windows.allowed(task, 50)) == max(50, len(task.required_tools()))


def test_retrieval_beats_putting_everything_in_context_at_scale(catalog, tasks, indexes) -> None:
    sweep = run_sweep(
        catalog=catalog,
        tasks=tasks,
        sizes=(1236,),
        indexes=indexes,
        strategies=("all_tools", "rerank"),
        keep_attempts=False,
    )
    by = {s.strategy: s for s in sweep.summaries}
    assert by["rerank"].task_success > by["all_tools"].task_success
    assert by["all_tools"].mean_prompt_tokens > 50 * by["rerank"].mean_prompt_tokens


def test_strategies_agree_when_the_catalog_fits_inside_k(catalog, tasks, indexes) -> None:
    """A sanity check on the harness.

    With 8 tools and k=8 there is nothing to retrieve, so any strategy that ranks
    the whole catalog shows the model the same set and must score the same. BM25
    is the exception, and legitimately so: it only ranks documents matching at
    least one query term, so it can hand back fewer than k candidates even when
    the catalog is smaller than k. That is a property of the scorer, not a bug in
    the harness, and it is why it scores differently here.
    """
    sweep = run_sweep(
        catalog=catalog, tasks=tasks, sizes=(8,), k=8, indexes=indexes, keep_attempts=False
    )
    index = {s.strategy: s for s in sweep.summaries}
    assert len(index) == 5
    dense = {round(index[s].task_success, 9) for s in ("all_tools", "embedding", "rerank", "progressive")}
    assert len(dense) == 1, {s: index[s].task_success for s in index}
    assert index["bm25"].recall_at_k < 1.0
    assert index["bm25"].task_success <= index["all_tools"].task_success


def test_accuracy_falls_and_recall_falls_as_the_catalog_grows(catalog, tasks, indexes) -> None:
    sweep = run_sweep(
        catalog=catalog,
        tasks=tasks,
        sizes=(25, 1236),
        indexes=indexes,
        strategies=("rerank",),
        keep_attempts=False,
    )
    small, large = sorted(sweep.summaries, key=lambda s: s.catalog_size)
    assert small.task_success > large.task_success
    assert small.recall_at_k > large.recall_at_k


def test_retrieval_holds_the_prompt_flat_while_the_baseline_does_not(catalog, tasks, indexes) -> None:
    sweep = run_sweep(
        catalog=catalog,
        tasks=tasks,
        sizes=(25, 1236),
        indexes=indexes,
        strategies=("all_tools", "bm25"),
        keep_attempts=False,
    )
    index = {(s.strategy, s.catalog_size): s for s in sweep.summaries}
    baseline_growth = index[("all_tools", 1236)].mean_prompt_tokens / index[("all_tools", 25)].mean_prompt_tokens
    retrieval_growth = index[("bm25", 1236)].mean_prompt_tokens / index[("bm25", 25)].mean_prompt_tokens
    assert baseline_growth > 20
    assert retrieval_growth < 1.5


def test_progressive_disclosure_pays_two_calls(catalog, tasks, indexes) -> None:
    sweep = run_sweep(
        catalog=catalog, tasks=tasks, sizes=(400,), indexes=indexes, strategies=("progressive",), keep_attempts=False
    )
    assert sweep.summaries[0].stages == 2


# -- compression -----------------------------------------------------------


def test_compression_saves_a_large_fraction_of_the_tokens(catalog, tasks, indexes) -> None:
    rows = {r.fmt: r for r in run_compression(catalog=catalog, tasks=tasks, size=25, indexes=indexes)}
    assert set(rows) == set(FORMATS)
    assert rows["json_schema"].savings_vs_json == 0.0
    assert rows["compact_json"].savings_vs_json > 0.3
    assert rows["signature"].savings_vs_json > 0.7
    assert rows["oneline"].savings_vs_json > rows["signature"].savings_vs_json


def test_minifying_json_is_free_because_it_drops_no_information(catalog, tasks, indexes) -> None:
    """Whitespace is not information. The parsed view is identical either way, so
    the accuracy delta has to be exactly zero, and a non-zero one would mean the
    harness is leaking something other than the rendering into the decision."""
    rows = {r.fmt: r for r in run_compression(catalog=catalog, tasks=tasks, size=25, indexes=indexes)}
    assert rows["compact_json"].success_delta_vs_json == 0.0
    assert rows["compact_json"].task_success == rows["json_schema"].task_success


def test_the_simulator_cannot_resolve_the_accuracy_cost_of_dropping_descriptions(
    catalog, indexes
) -> None:
    """A documented limitation, asserted so it stays documented.

    Dropping per-parameter descriptions should cost accuracy, because placing an
    argument on the right parameter is what those descriptions are for. In this
    simulator it does not measurably cost anything: the delta straddles zero,
    changing sign between one catalog size and another, so no direction can be
    read off it.

    The reason is structural rather than a tuning accident. The simulator places
    arguments by overlapping a parameter's words against the words introducing a
    literal in the query. The catalog's generated descriptions are largely
    restatements of the parameter name plus the resource noun, so they add about
    as many spurious matches as real ones, and a deterministic overlap score has
    no way to weigh a description as evidence the way a language model does. A
    simulator can show that compression removes information; it cannot price it.
    MEASUREMENTS.md answers this from the real-model run instead.
    """
    full = build_tasks(catalog)
    small = {r.fmt: r for r in run_compression(catalog=catalog, tasks=full, size=25, indexes=indexes)}
    large = {r.fmt: r for r in run_compression(catalog=catalog, tasks=full, size=1236, indexes=indexes)}

    # Small in magnitude everywhere: nothing like the cost the mechanism predicts.
    for fmt in ("signature", "oneline"):
        assert abs(small[fmt].success_delta_vs_json) < 0.05, (fmt, small[fmt].success_delta_vs_json)
        assert abs(large[fmt].success_delta_vs_json) < 0.05, (fmt, large[fmt].success_delta_vs_json)

    # And sign-unstable, which is the part that makes it unresolvable rather than
    # merely small: the same rendering helps at one catalog size and hurts at another.
    flips = [
        fmt
        for fmt in ("signature", "oneline")
        if small[fmt].success_delta_vs_json > 0 > large[fmt].success_delta_vs_json
    ]
    assert flips, {
        fmt: (small[fmt].success_delta_vs_json, large[fmt].success_delta_vs_json)
        for fmt in ("signature", "oneline")
    }


# -- taxonomy and disagreement --------------------------------------------


def test_taxonomy_accounts_for_every_failure(catalog, tasks, indexes) -> None:
    sweep = run_sweep(catalog=catalog, tasks=tasks, sizes=(200,), indexes=indexes)
    report = taxonomy(sweep.attempts)
    assert report["total_failures"] == sum(1 for a in sweep.attempts if not a.success)
    assert sum(report["overall"].values()) == report["total_failures"]
    assert sum(report["overall_share"].values()) == pytest.approx(1.0, abs=1e-6)


def test_the_taxonomy_contains_the_four_modes_the_project_set_out_to_measure(catalog, tasks, indexes) -> None:
    sweep = run_sweep(catalog=catalog, tasks=tasks, sizes=(200, 1236), indexes=indexes)
    modes = set(taxonomy(sweep.attempts)["overall"])
    assert {"wrong_tool_selected", "below_cutoff", "near_duplicate_confusion", "hallucinated_argument"} <= modes


def test_recall_and_task_success_disagree_in_one_direction_only(catalog, indexes) -> None:
    """The headline asymmetry, pinned down.

    Recall@k counting a hit while the task fails is the common case. The reverse,
    recall counting a miss while the task still succeeded, never happens here, so
    on this catalog and task set recall@k is an optimistic upper bound on task
    success rather than a proxy for it: every task the agent completed had the
    gold tool in the shortlist, and roughly half of the ones that did still failed.
    """
    full = build_tasks(catalog)
    sweep = run_sweep(catalog=catalog, tasks=full, sizes=(1236,), indexes=indexes)
    report = disagreements(sweep.attempts)

    assert report["recall_hit_task_fail"] > 0.3 * report["n"]
    assert report["recall_miss_task_pass"] == 0, report["recall_miss_task_pass_by_kind"]

    # Recall@k is therefore an upper bound on task success, strategy by strategy.
    for summary in sweep.summaries:
        assert summary.recall_at_k >= summary.task_success, summary.strategy


def test_some_tasks_are_completed_by_a_tool_recall_was_not_scored_against(catalog, indexes) -> None:
    """The weak half of the disagreement, measured rather than assumed.

    A second server exposing the same operation under another name does let the
    agent finish the job with a tool that is not the gold one. It happens, but
    rarely, and never rescues a retrieval miss: a substitute that does the same
    job reads almost the same to a retriever, so it arrives alongside the tool it
    substitutes for rather than instead of it.
    """
    full = build_tasks(catalog)
    sweep = run_sweep(catalog=catalog, tasks=full, sizes=(1236,), indexes=indexes)
    report = disagreements(sweep.attempts)

    assert report["success_via_non_gold_tool"] > 0
    assert set(report["success_via_non_gold_tool_by_kind"]) == {"aliased"}

    off_gold = [a for a in sweep.attempts if a.success and a.action_tool not in (None, a.gold_tool)]
    assert off_gold
    assert all(a.recall_hit for a in off_gold)
    for attempt in off_gold:
        assert catalog.get(attempt.action_tool or "").alias_of == attempt.gold_tool


def test_summarize_refuses_an_empty_run() -> None:
    with pytest.raises(AssertionError):
        summarize([])
