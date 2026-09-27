from __future__ import annotations

import json
import re

import pytest

from atg.bench import full_report
from atg.report import SERIES_DARK, SERIES_LIGHT, write_report

VIEWBOX = re.compile(r'<svg viewBox="0 0 (\d+) (\d+)"')
END_LABEL = re.compile(r'<text class="end-label" x="([\d.]+)" y="([\d.]+)"')
POINT = re.compile(r"[ML]([\d.]+),([\d.]+)")


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench") / "results.json"
    return full_report(sizes=(10, 100, 1236), k=8, out=out)





@pytest.fixture(scope="module")
def html(payload, tmp_path_factory):
    path = tmp_path_factory.mktemp("report") / "report.html"
    write_report(payload, path)
    return path.read_text()


def test_the_report_is_one_self_contained_file(html) -> None:
    assert html.startswith("<!doctype html>")
    assert "<style>" in html and "<script>" in html
    for forbidden in ("http://", "https://", "src=", "@import"):
        assert forbidden not in html, forbidden


def test_no_nan_or_inf_leaked_into_any_coordinate(html) -> None:
    for bad in ("NaN", "nan,", "Infinity", "None"):
        assert bad not in html, bad


def test_every_plotted_point_is_inside_its_viewbox(html) -> None:
    for chunk in html.split("<svg ")[1:]:
        match = VIEWBOX.search("<svg " + chunk)
        assert match
        width, height = int(match.group(1)), int(match.group(2))
        for d in re.findall(r'<path class="series" d="([^"]+)"', chunk):
            for x, y in POINT.findall(d):
                assert 0 <= float(x) <= width, x
                assert 0 <= float(y) <= height, y


def test_end_labels_do_not_collide(html) -> None:
    """Converging series are where a value label is most wanted and most likely to
    overlap, so the de-collision pass is worth asserting."""
    for chart in ("success", "recall", "tokens"):
        segment = html.split(f'data-chart="{chart}"')[1].split("</svg>")[0]
        ys = sorted(float(y) for _, y in END_LABEL.findall(segment))
        assert len(ys) >= 4, chart
        gaps = [b - a for a, b in zip(ys, ys[1:])]
        assert min(gaps) >= 11.0, (chart, gaps)


def test_identity_is_never_carried_by_colour_alone(html) -> None:
    """A legend for every multi-series chart, and a table of the same numbers."""
    assert 'class="legend"' in html
    assert html.count("<table>") >= 3
    assert html.count('role="img"') == html.count("<svg ")
    assert html.count("aria-label") >= html.count("<svg ")


def test_the_palette_is_defined_for_light_and_both_dark_scopes(html) -> None:
    assert "prefers-color-scheme: dark" in html
    assert '[data-theme="dark"]' in html
    for light, dark in zip(SERIES_LIGHT, SERIES_DARK):
        assert light in html
        assert dark in html
    assert "background: var(--surface)" in html


def test_hover_tooltips_are_wired_to_every_point(html) -> None:
    dots = html.count('class="dot"')
    assert dots > 0
    assert html.count("data-series=") == dots
    # Every point carries an SVG <title> for the no-JavaScript and screen-reader
    # case. The document's own <title> is not one of them.
    assert len(re.findall(r"<circle[^>]*><title>", html)) == dots
    assert "mouseenter" in html and "mouseleave" in html


def test_the_report_states_its_provenance(html, payload) -> None:
    assert payload["sweep"]["catalog_fingerprint"] in html
    assert str(payload["sweep"]["n_tasks"]) in html
    assert payload["sweep"]["decider"] in html
    assert payload["generated_utc"] in html


def test_both_compression_tables_are_present(html, payload) -> None:
    assert payload["compression_small_catalog"]
    small = payload["compression_small_catalog"][0]["catalog_size"]
    large = payload["compression"][0]["catalog_size"]
    assert small != large
    assert f"{small} tools" in html
    assert f"{large:,} tools" in html or f"{large} tools" in html


def test_the_disagreement_is_stated_in_both_directions(html, payload) -> None:
    dis = payload["disagreement_full_catalog"]
    assert str(dis["recall_hit_task_fail"]) in html
    assert str(dis["recall_miss_task_pass"]) in html
    assert str(dis["success_via_non_gold_tool"]) in html


def test_the_results_file_is_json_and_carries_every_section(payload, tmp_path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload))
    back = json.loads(path.read_text())
    assert set(back) >= {
        "generated_utc",
        "sweep",
        "compression",
        "compression_small_catalog",
        "taxonomy_full_catalog",
        "taxonomy_all_sizes",
        "disagreement_full_catalog",
        "disagreement_all_sizes",
        "per_kind_full_catalog",
    }


def test_full_report_can_write_a_trajectory(tmp_path) -> None:
    from atg.events import EVENT_TYPES, read_trajectory

    out = tmp_path / "results.json"
    traj = tmp_path / "sweep.jsonl"
    full_report(sizes=(10, 50), k=8, out=out, trajectory_out=traj)
    events = read_trajectory(traj)
    assert events[0].type == "run_start"
    assert events[-1].type == "run_end"
    assert [e.seq for e in events] == list(range(len(events)))
    assert all(e.type in EVENT_TYPES for e in events)
