from __future__ import annotations

import pytest

from atg.catalog import build_catalog
from atg.llm_sim import extract_mentions
from atg.tasks import Clarify, ToolCall, build_tasks, grade


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


@pytest.fixture(scope="module")
def tasks(catalog):
    return build_tasks(catalog)


def test_task_suite_is_reproducible(catalog) -> None:
    a = build_tasks(catalog)
    b = build_tasks(catalog)
    assert [(t.task_id, t.query, t.gold_tool) for t in a] == [
        (t.task_id, t.query, t.gold_tool) for t in b
    ]


def test_every_kind_is_represented(tasks) -> None:
    kinds = {t.kind for t in tasks}
    assert kinds == {"plain", "family", "conditional", "aliased", "underspecified"}


def test_the_gold_tool_exists_and_the_gold_args_are_its_required_params(catalog, tasks) -> None:
    for task in tasks:
        tool = catalog.get(task.gold_tool)
        required = {p.name for p in tool.required_params}
        supplied = set(task.slot_map)
        assert supplied <= required, task.task_id
        if task.kind == "underspecified":
            assert task.missing_param in required
            assert supplied == required - {task.missing_param}
        else:
            assert supplied == required, task.task_id


def test_every_gold_argument_value_appears_verbatim_in_the_query(tasks) -> None:
    """The grader's hallucination check depends on this being true."""
    for task in tasks:
        for name, value in task.slots:
            assert value in task.query, f"{task.task_id}: {name}={value!r} not in {task.query!r}"


def test_the_omitted_value_is_genuinely_absent_from_the_query(catalog, tasks) -> None:
    for task in tasks:
        if task.kind != "underspecified":
            continue
        values = {m.value for m in extract_mentions(task.query)}
        assert task.missing_param not in task.slot_map
        tool = catalog.get(task.gold_tool)
        param = next(p for p in tool.required_params if p.name == task.missing_param)
        assert param.name not in task.slot_map
        # No literal in the query can be the missing value, since it was never
        # generated into the text at all.
        assert all(task.slot_map.get(task.missing_param) != v for v in values)


def test_queries_do_not_leak_the_tool_name(tasks) -> None:
    leaked = [t for t in tasks if t.gold_tool.replace("_", " ").replace(".", " ") in t.query.lower()]
    assert not leaked, [t.task_id for t in leaked[:5]]


def test_family_tasks_carry_their_near_duplicates(tasks) -> None:
    for task in tasks:
        if task.kind in ("family", "conditional"):
            assert task.confusable_with, task.task_id
            assert task.gold_tool not in task.confusable_with


def test_aliased_tasks_accept_either_name(catalog, tasks) -> None:
    aliased = [t for t in tasks if t.kind == "aliased"]
    assert aliased
    for task in aliased:
        assert len(task.acceptable) == 2
        other = next(n for n in task.acceptable if n != task.gold_tool)
        assert catalog.get(other).alias_of == task.gold_tool


# -- grading ---------------------------------------------------------------


def _task(tasks, kind):
    return next(t for t in tasks if t.kind == kind)


def test_a_correct_call_succeeds(catalog, tasks) -> None:
    task = _task(tasks, "plain")
    result = grade(task, ToolCall(task.gold_tool, task.slot_map), catalog, gold_rank=1, k=8)
    assert result.success and result.outcome == "success"


def test_an_acceptable_alias_also_succeeds(catalog, tasks) -> None:
    task = _task(tasks, "aliased")
    other = next(n for n in task.acceptable if n != task.gold_tool)
    result = grade(task, ToolCall(other, task.slot_map), catalog, gold_rank=1, k=8)
    assert result.success


def test_a_near_duplicate_is_classified_as_such(catalog, tasks) -> None:
    task = _task(tasks, "family")
    sibling = task.confusable_with[0]
    result = grade(task, ToolCall(sibling, {}), catalog, gold_rank=1, k=8)
    assert result.outcome == "near_duplicate_confusion"


def test_an_unrelated_tool_is_a_plain_wrong_tool(catalog, tasks) -> None:
    task = _task(tasks, "plain")
    other = next(t.name for t in catalog.tools if t.name not in task.acceptable and t.family != catalog.get(task.gold_tool).family)
    result = grade(task, ToolCall(other, {}), catalog, gold_rank=1, k=8)
    assert result.outcome == "wrong_tool_selected"


def test_a_miss_just_outside_k_is_distinguished_from_a_deep_miss(catalog, tasks) -> None:
    task = _task(tasks, "plain")
    other = next(t.name for t in catalog.tools if t.name not in task.acceptable)
    near = grade(task, ToolCall(other, {}), catalog, gold_rank=11, k=8)
    deep = grade(task, ToolCall(other, {}), catalog, gold_rank=900, k=8)
    unranked = grade(task, ToolCall(other, {}), catalog, gold_rank=None, k=8)
    assert near.outcome == "below_cutoff"
    assert deep.outcome == "deep_retrieval_miss"
    assert unranked.outcome == "deep_retrieval_miss"


def test_an_invented_required_value_is_caught(catalog, tasks) -> None:
    task = _task(tasks, "plain")
    args = dict(task.slot_map)
    name = next(iter(args))
    args[name] = "a value that was never in the request"
    result = grade(task, ToolCall(task.gold_tool, args), catalog, gold_rank=1, k=8)
    assert result.outcome == "wrong_argument_value"


def test_filling_a_value_that_was_never_given_is_a_hallucination(catalog, tasks) -> None:
    task = _task(tasks, "underspecified")
    args = dict(task.slot_map)
    args[task.missing_param or ""] = "invented"
    result = grade(task, ToolCall(task.gold_tool, args), catalog, gold_rank=1, k=8)
    assert result.outcome == "hallucinated_argument"
    assert not result.success


def test_asking_for_the_missing_value_is_the_correct_answer(catalog, tasks) -> None:
    task = _task(tasks, "underspecified")
    result = grade(task, Clarify(task.gold_tool, task.missing_param or ""), catalog, gold_rank=1, k=8)
    assert result.success


def test_asking_about_the_wrong_thing_is_not(catalog, tasks) -> None:
    task = _task(tasks, "underspecified")
    other = next(n for n in task.slot_map)
    result = grade(task, Clarify(task.gold_tool, other), catalog, gold_rank=1, k=8)
    assert result.outcome == "missing_argument"


def test_clarifying_about_the_wrong_tool_is_reported_as_a_wrong_tool(catalog, tasks) -> None:
    task = _task(tasks, "underspecified")
    other = next(t.name for t in catalog.tools if t.name not in task.acceptable)
    result = grade(task, Clarify(other, task.missing_param or ""), catalog, gold_rank=1, k=8)
    assert result.outcome in ("wrong_tool_selected", "near_duplicate_confusion")


def test_clarifying_on_a_fully_specified_task_fails(catalog, tasks) -> None:
    task = _task(tasks, "plain")
    result = grade(task, Clarify(task.gold_tool, next(iter(task.slot_map))), catalog, gold_rank=1, k=8)
    assert result.outcome == "missing_argument"


def test_omitting_a_required_argument_fails(catalog, tasks) -> None:
    task = _task(tasks, "plain")
    result = grade(task, ToolCall(task.gold_tool, {}), catalog, gold_rank=1, k=8)
    assert result.outcome == "missing_argument"


def test_no_action_is_classified_by_where_the_gold_was_ranked(catalog, tasks) -> None:
    task = _task(tasks, "plain")
    assert grade(task, None, catalog, gold_rank=None, k=8).outcome == "deep_retrieval_miss"
    assert grade(task, None, catalog, gold_rank=10, k=8).outcome == "below_cutoff"
    assert grade(task, None, catalog, gold_rank=2, k=8).outcome == "no_action"
