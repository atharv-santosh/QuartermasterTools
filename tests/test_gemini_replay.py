"""Real-model tests. Every one of these runs from a recorded fixture with no
credentials; the live ones skip unless ADC is configured and ATG_LIVE is set."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from atg.agent import run_attempt
from atg.bench import CatalogWindows, summarize
from atg.catalog import build_catalog
from atg.gemini_adapter import (
    PRICE_PER_MTOK,
    Usage,
    cache_key,
    credentials_available,
    load_cache,
    make_replay_decider,
)
from atg.llm_sim import select_shortlist
from atg.retrieval.strategies import Indexes, build_strategies
from atg.tasks import build_tasks

FIXTURES = Path(__file__).parent / "fixtures"
HEADLINE = FIXTURES / "gemini_headline.json"
HEADLINE_SUMMARY = FIXTURES / "gemini_headline_summary.json"


def _needs(path: Path):
    return pytest.mark.skipif(not path.exists(), reason=f"no recorded fixture at {path}")


live_only = pytest.mark.skipif(
    not (os.environ.get("ATG_LIVE") and credentials_available()),
    reason="live model test; set ATG_LIVE=1 with ADC configured",
)


# -- accounting, no fixture needed ----------------------------------------


def test_cost_is_computed_from_measured_tokens_at_list_price() -> None:
    usage = Usage()
    usage.add(1_000_000, 1_000_000, 2_000_000)
    inp, out = PRICE_PER_MTOK["gemini-2.5-flash-lite"]
    assert usage.cost_usd("gemini-2.5-flash-lite") == pytest.approx(inp + out)
    assert usage.cost_usd("a-model-with-no-published-price") == 0.0


def test_usage_accumulates_and_reports_retries() -> None:
    usage = Usage()
    usage.add(10, 2, 12)
    usage.add(20, 3, 23)
    usage.retries += 2
    assert usage.calls == 2 and usage.total_tokens == 35
    assert usage.to_dict("gemini-2.5-flash-lite")["retries"] == 2


def test_a_cache_key_is_the_request_plus_exactly_what_was_shown() -> None:
    from atg.render import view

    catalog = build_catalog()
    a = view(catalog.get("send_email"), "json_schema")
    b = view(catalog.get("send_bulk_email"), "json_schema")
    compressed = view(catalog.get("send_email"), "signature")
    assert cache_key("q", [a], "native") == cache_key("q", [a], "native")
    assert cache_key("q", [a], "native") != cache_key("q", [a, b], "native")
    assert cache_key("q", [a], "native") != cache_key("q", [a], "proxy")
    assert cache_key("q", [a], "native") != cache_key("q", [compressed], "native")
    assert cache_key("q1", [a], "native") != cache_key("q2", [a], "native")


# -- replay ----------------------------------------------------------------


@_needs(HEADLINE)
def test_the_fixture_keeps_calls_that_produced_no_function_call() -> None:
    """Dropping them would turn a recorded prose answer into a replay miss, which
    is a different result and would flatter the model."""
    cache = load_cache(HEADLINE)
    payload = json.loads(HEADLINE.read_text())
    assert len(cache) == len({r["key"] for r in payload["records"]})
    assert any(v["name"] is None for v in cache.values())


@_needs(HEADLINE)
def test_replay_refuses_a_prompt_it_never_recorded() -> None:
    from atg.render import view

    catalog = build_catalog()
    decider = make_replay_decider(HEADLINE, convention="native")
    with pytest.raises(KeyError):
        decider([view(catalog.get("send_email"), "json_schema")], "a request never recorded")


@_needs(HEADLINE)
def test_a_lenient_replay_returns_nothing_instead_of_raising() -> None:
    from atg.render import view

    catalog = build_catalog()
    decider = make_replay_decider(HEADLINE, convention="native", strict=False)
    assert decider([view(catalog.get("send_email"), "json_schema")], "never recorded") is None


@_needs(HEADLINE_SUMMARY)
def test_the_recorded_run_replays_to_the_numbers_it_recorded() -> None:
    """The measurement in MEASUREMENTS.md, reproduced offline from the fixture."""
    meta = json.loads(HEADLINE_SUMMARY.read_text())
    catalog = build_catalog()
    tasks = list(build_tasks(catalog))
    random.Random(meta["sample_seed"]).shuffle(tasks)
    sample = tasks[: meta["n_sampled"]]
    assert [t.task_id for t in sample] == meta["task_ids"], "task suite moved since recording"

    indexes = Indexes.build(catalog)
    windows = CatalogWindows(catalog, sample)
    strategy = build_strategies(indexes, select_shortlist, fmt=meta["fmt"])[meta["strategy"]]
    replay = make_replay_decider(HEADLINE, convention=meta["convention"])

    for row in meta["rows"]:
        summary = summarize(
            [
                run_attempt(
                    task,
                    strategy,
                    catalog,
                    replay,
                    k=meta["k"],
                    allowed=windows.allowed(task, row["size"]),
                    catalog_size=row["size"],
                )
                for task in sample
            ]
        )
        assert round(summary.task_success, 4) == row["task_success"], row["size"]
        assert round(summary.recall_at_k, 4) == row["recall_at_k"], row["size"]


@_needs(HEADLINE_SUMMARY)
def test_the_real_model_degrades_far_less_than_the_simulator() -> None:
    """The finding that matters most about the simulator: it is pessimistic.

    Recall falls with catalog size for both, because retrieval does not depend on
    the model. Task success is where they part company, and the gap widens as the
    catalog grows, so the simulator's curve should be read as a lower bound rather
    than as a prediction.
    """
    from atg.llm_sim import decide as sim_decide

    meta = json.loads(HEADLINE_SUMMARY.read_text())
    catalog = build_catalog()
    tasks = list(build_tasks(catalog))
    random.Random(meta["sample_seed"]).shuffle(tasks)
    sample = tasks[: meta["n_sampled"]]
    indexes = Indexes.build(catalog)
    windows = CatalogWindows(catalog, sample)
    strategy = build_strategies(indexes, select_shortlist, fmt=meta["fmt"])[meta["strategy"]]
    replay = make_replay_decider(HEADLINE, convention=meta["convention"])

    gaps = []
    for row in meta["rows"]:
        allowed = {t.task_id: windows.allowed(t, row["size"]) for t in sample}
        both = []
        for decider in (sim_decide, replay):
            both.append(
                summarize(
                    [
                        run_attempt(
                            t,
                            strategy,
                            catalog,
                            decider,
                            k=meta["k"],
                            allowed=allowed[t.task_id],
                            catalog_size=row["size"],
                        )
                        for t in sample
                    ]
                )
            )
        sim, real = both
        assert sim.recall_at_k == real.recall_at_k, "retrieval must not depend on the model"
        gaps.append((row["size"], real.task_success - sim.task_success))

    assert all(gap > 0 for _, gap in gaps), gaps
    assert gaps[-1][1] > gaps[0][1], gaps


# -- live ------------------------------------------------------------------


@live_only
def test_one_live_call_still_works() -> None:
    from atg.gemini_adapter import DEFAULT_MODEL, build_client, make_gemini_decider
    from atg.render import view

    catalog = build_catalog()
    decider = make_gemini_decider(client=build_client(), model=DEFAULT_MODEL, convention="native")
    views = [view(catalog.get(n), "json_schema") for n in catalog.families["email_send"]]
    action = decider(views, "Email finance@northwind.example right now with subject line 'Q3' and body 'done'")
    assert action is not None
