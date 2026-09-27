from __future__ import annotations

import json
from pathlib import Path

import pytest

from atg.calibrate import CALIBRATION_FIXTURE, report_calibration
from atg.catalog import build_catalog
from atg.render import FORMATS, block_tokens, render, view
from atg.tokens import count_tokens


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_rendering_round_trips_into_a_view(catalog, fmt) -> None:
    for tool in catalog.tools[:120]:
        v = view(tool, fmt)
        assert v.name == tool.name
        assert {p.name for p in v.params} == {p.name for p in tool.params}
        assert {p.name for p in v.required} == {p.name for p in tool.required_params}


def test_compression_is_monotone_in_token_cost(catalog) -> None:
    tools = list(catalog.tools[:60])
    costs = [block_tokens(tools, fmt) for fmt in FORMATS]
    assert costs == sorted(costs, reverse=True), dict(zip(FORMATS, costs))


def test_compression_drops_parameter_descriptions_and_json_keeps_them(catalog) -> None:
    """The mechanism the accuracy delta comes from, asserted directly."""
    tool = next(t for t in catalog.tools if any(p.description for p in t.params))
    assert any(p.description for p in view(tool, "json_schema").params)
    assert not any(p.description for p in view(tool, "signature").params)


def test_oneline_drops_the_tool_description_and_signature_keeps_it(catalog) -> None:
    tool = catalog.get("refund_charge_partial")
    assert view(tool, "signature").description
    assert not view(tool, "oneline").description


def test_signature_marks_required_parameters_and_lists_enums(catalog) -> None:
    tool = next(t for t in catalog.tools if any(p.enum for p in t.params) and t.required_params)
    rendered = render(tool, "signature")
    assert "!" in rendered
    enum_param = next(p for p in tool.params if p.enum)
    assert "|".join(enum_param.enum) in rendered


def test_precondition_survives_into_json_and_signature(catalog) -> None:
    tool = catalog.get("refund_charge_partial")
    assert "Only valid when" in render(tool, "json_schema")
    assert "Only valid when" in render(tool, "signature")


def test_a_block_costs_more_than_its_parts_because_of_the_wrapper(catalog) -> None:
    tools = list(catalog.tools[:10])
    assert block_tokens(tools, "signature") > sum(count_tokens(render(t, "signature")) for t in tools) * 0.9


def test_token_counter_is_stable_and_monotone() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello") == 1
    assert count_tokens("hello world") == 2
    long = "a" * 400
    assert count_tokens(long) > count_tokens("a" * 40)
    assert count_tokens("x" * 100) == count_tokens("x" * 100)


def test_token_calibration_fixture_still_holds() -> None:
    """The offline estimator against recorded Vertex counts. No network."""
    if not CALIBRATION_FIXTURE.exists():
        pytest.skip("calibration fixture not present")
    result = report_calibration(CALIBRATION_FIXTURE)
    assert "error" not in result, result
    assert result["overall"]["n"] >= 100
    assert result["overall"]["mean_abs_pct_err"] < 8.0, result["overall"]
    assert result["by_split"]["holdout"]["mean_abs_pct_err"] < 8.0, result["by_split"]
    assert 0.9 < result["overall"]["mean_ratio"] < 1.1


def test_calibration_fixture_stores_recipes_not_saved_text() -> None:
    if not CALIBRATION_FIXTURE.exists():
        pytest.skip("calibration fixture not present")
    payload = json.loads(Path(CALIBRATION_FIXTURE).read_text())
    for sample in payload["samples"][:20]:
        assert set(sample) <= {"fmt", "tools", "real_tokens", "split"}
        assert "text" not in sample
