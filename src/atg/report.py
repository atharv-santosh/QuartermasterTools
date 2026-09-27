"""Self-contained HTML report: no network, no build step, inline SVG."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Sequence

from atg.retrieval.strategies import STRATEGY_ORDER

# Categorical slots 1-5 in fixed order, light and dark steps. Assigned to a
# strategy once and never re-assigned, so a strategy keeps its colour across
# every chart on the page.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181")

LABELS: dict[str, str] = {
    "all_tools": "all tools in context",
    "bm25": "BM25",
    "embedding": "embedding",
    "rerank": "retrieve then rerank",
    "progressive": "progressive disclosure",
}

FMT_LABELS: dict[str, str] = {
    "json_schema": "full JSON Schema",
    "compact_json": "minified JSON Schema",
    "signature": "signature plus summary",
    "oneline": "signature only",
}

FAILURE_LABELS: dict[str, str] = {
    "wrong_tool_selected": "wrong tool selected",
    "near_duplicate_confusion": "near duplicate confusion",
    "below_cutoff": "right tool just below the cutoff",
    "deep_retrieval_miss": "right tool nowhere near the shortlist",
    "hallucinated_argument": "hallucinated argument",
    "missing_argument": "missing argument",
    "wrong_argument_value": "wrong argument value",
    "no_action": "no action produced",
}

W, H = 560, 330
PAD_L, PAD_R, PAD_T, PAD_B = 52, 96, 18, 42


def _esc(text: Any) -> str:
    return html.escape(str(text))


def _log_x(value: float, lo: float, hi: float) -> float:
    span = math.log10(hi) - math.log10(lo)
    frac = (math.log10(value) - math.log10(lo)) / span if span else 0.0
    return PAD_L + frac * (W - PAD_L - PAD_R)


def _lin_y(value: float, lo: float, hi: float) -> float:
    frac = (value - lo) / (hi - lo) if hi > lo else 0.0
    return H - PAD_B - frac * (H - PAD_T - PAD_B)


def _log_y(value: float, lo: float, hi: float) -> float:
    value = max(value, lo)
    span = math.log10(hi) - math.log10(lo)
    frac = (math.log10(value) - math.log10(lo)) / span if span else 0.0
    return H - PAD_B - frac * (H - PAD_T - PAD_B)


def _line_chart(
    *,
    chart_id: str,
    title: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    sizes: Sequence[int],
    y_label: str,
    y_log: bool = False,
    y_max: float | None = None,
    y_fmt: str = "pct",
) -> str:
    lo_x, hi_x = min(sizes), max(sizes)
    values = [y for _, points in series for _, y in points]
    if y_log:
        lo_y = max(1.0, min(values) / 1.6)
        hi_y = max(values) * 1.3
        ypos = lambda v: _log_y(v, lo_y, hi_y)
        ticks = [t for t in (100, 1_000, 10_000, 100_000, 1_000_000) if lo_y <= t <= hi_y]
    else:
        lo_y, hi_y = 0.0, y_max or 1.0
        ypos = lambda v: _lin_y(v, lo_y, hi_y)
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0] if hi_y >= 1.0 else [0.0, hi_y / 2, hi_y]

    def fmt(v: float) -> str:
        if y_fmt == "pct":
            return f"{v:.0%}"
        if v >= 1000:
            return f"{v / 1000:.0f}k"
        return f"{v:,.0f}"

    parts: list[str] = [
        f'<figure class="chart"><figcaption>{_esc(title)}</figcaption>',
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(title)}" '
        f'class="plot" data-chart="{chart_id}">',
    ]
    for tick in ticks:
        y = ypos(tick)
        parts.append(
            f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<text class="tick" x="{PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end">{fmt(tick)}</text>'
        )
    for size in sizes:
        x = _log_x(size, lo_x, hi_x)
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{H - PAD_B + 18}" text-anchor="middle">{size}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 6}" '
        f'text-anchor="middle">tools in the catalog</text>'
    )
    parts.append(
        f'<text class="axis-label" x="10" y="{PAD_T + 4}" text-anchor="start">{_esc(y_label)}</text>'
    )

    ends: list[tuple[float, int, str]] = []
    for slot, (name, points) in enumerate(series):
        color = f"var(--series-{slot + 1})"
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{_log_x(x, lo_x, hi_x):.1f},{ypos(y):.1f}"
            for i, (x, y) in enumerate(points)
        )
        parts.append(f'<path class="series" d="{path}" stroke="{color}"/>')
        for x, y in points:
            parts.append(
                f'<circle class="dot" cx="{_log_x(x, lo_x, hi_x):.1f}" cy="{ypos(y):.1f}" r="4" '
                f'fill="{color}" data-series="{_esc(LABELS.get(name, name))}" '
                f'data-x="{int(x)}" data-y="{_esc(fmt(y))}"><title>'
                f'{_esc(LABELS.get(name, name))}: {_esc(fmt(y))} at {int(x)} tools</title></circle>'
            )
        ends.append((ypos(points[-1][1]), slot, fmt(points[-1][1])))

    last_x = _log_x(series[0][1][-1][0], lo_x, hi_x) + 9
    for y, slot, text in _spread(ends):
        parts.append(
            f'<text class="end-label" x="{last_x:.1f}" y="{y + 3.5:.1f}" '
            f'fill="var(--series-{slot + 1})">{_esc(text)}</text>'
        )
    parts.append("</svg></figure>")
    return "".join(parts)


_LABEL_GAP = 12.0


def _spread(ends: list[tuple[float, int, str]]) -> list[tuple[float, int, str]]:
    """Nudge end labels apart so converging series do not stack their values.

    Converging lines are exactly where a value label is most wanted and most
    likely to collide, so this keeps them and separates them rather than dropping
    them: one pass down enforcing a minimum gap, one pass up to pull the group
    back inside the plot, which is enough for the five series here.
    """
    ordered = sorted(ends, key=lambda e: e[0])
    out: list[tuple[float, int, str]] = []
    for y, slot, text in ordered:
        if out and y - out[-1][0] < _LABEL_GAP:
            y = out[-1][0] + _LABEL_GAP
        out.append((y, slot, text))
    overflow = out[-1][0] - (H - PAD_B)
    if overflow > 0:
        out = [(y - overflow, slot, text) for y, slot, text in out]
    return out


def _bar_chart(title: str, rows: list[tuple[str, int, float]]) -> str:
    if not rows:
        return ""
    bar_h, gap = 22, 10
    height = PAD_T + len(rows) * (bar_h + gap) + 24
    width = 620
    label_w = 250
    max_v = max(v for _, v, _ in rows)
    parts = [
        f'<figure class="chart wide"><figcaption>{_esc(title)}</figcaption>',
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}" class="plot">',
    ]
    for i, (label, count, share) in enumerate(rows):
        y = PAD_T + i * (bar_h + gap)
        w = (width - label_w - 70) * (count / max_v) if max_v else 0
        parts.append(
            f'<text class="bar-label" x="{label_w - 10}" y="{y + bar_h - 6}" text-anchor="end">'
            f"{_esc(label)}</text>"
            f'<rect class="bar" x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4"/>'
            f'<text class="bar-value" x="{label_w + w + 8:.1f}" y="{y + bar_h - 6}">'
            f"{count} ({share:.0%})</text>"
        )
    parts.append("</svg></figure>")
    return "".join(parts)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, caption: str = "") -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return f"<div class='table-wrap'><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _legend(names: Sequence[str]) -> str:
    items = "".join(
        f'<li><span class="swatch" style="background:var(--series-{i + 1})"></span>'
        f"{_esc(LABELS.get(n, n))}</li>"
        for i, n in enumerate(names)
    )
    return f'<ul class="legend">{items}</ul>'


# --------------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light;
  --surface: #fcfcfb;
  --surface-2: #f4f4f1;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --ink-3: #78766f;
  --rule: #dededa;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --series-4: #eda100; --series-5: #e87ba4;
  --bar: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --surface-2: #242423;
    --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #96958c; --rule: #3a3a38;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --series-4: #c98500; --series-5: #d55181;
    --bar: #3987e5;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface: #1a1a19; --surface-2: #242423;
  --ink: #ffffff; --ink-2: #c3c2b7; --ink-3: #96958c; --rule: #3a3a38;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --series-4: #c98500; --series-5: #d55181;
  --bar: #3987e5;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 20px 72px; }
h1 { font-size: 26px; font-weight: 620; letter-spacing: -0.01em; margin: 0 0 6px; }
.sub { color: var(--ink-2); margin: 0 0 32px; font-size: 14px; }
h2 { font-size: 17px; font-weight: 600; margin: 40px 0 6px; }
p { color: var(--ink-2); max-width: 66ch; margin: 0 0 16px; }
.tiles { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 8px; }
.tile {
  flex: 1 1 190px; background: var(--surface-2); border: 1px solid var(--rule);
  border-radius: 10px; padding: 14px 16px;
}
.tile .n { font-size: 25px; font-weight: 620; letter-spacing: -0.02em; display: block; }
.tile .l { font-size: 12.5px; color: var(--ink-2); }
.charts { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.chart { margin: 0; flex: 1 1 440px; min-width: 0; }
.chart.wide { flex-basis: 100%; }
figcaption { font-size: 13px; font-weight: 600; color: var(--ink); margin-bottom: 6px; }
.plot { width: 100%; height: auto; overflow: visible; }
.grid { stroke: var(--rule); stroke-width: 1; }
.tick { fill: var(--ink-3); font-size: 10.5px; }
.axis-label { fill: var(--ink-3); font-size: 10.5px; }
.series { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.dot { stroke: var(--surface); stroke-width: 2; }
.dot:hover { r: 6; }
.end-label { font-size: 10.5px; font-weight: 600; }
.bar { fill: var(--bar); }
.bar-label { fill: var(--ink-2); font-size: 12px; }
.bar-value { fill: var(--ink-3); font-size: 11px; }
.legend { list-style: none; display: flex; flex-wrap: wrap; gap: 6px 18px; padding: 0; margin: 4px 0 20px; font-size: 12.5px; color: var(--ink-2); }
.legend li { display: flex; align-items: center; gap: 7px; }
.swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
.table-wrap { overflow-x: auto; margin: 8px 0 20px; }
table { border-collapse: collapse; font-size: 13px; min-width: 100%; font-variant-numeric: tabular-nums; }
caption { text-align: left; font-size: 12.5px; color: var(--ink-3); padding-bottom: 6px; }
th, td { text-align: right; padding: 6px 12px; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-3); font-weight: 600; font-size: 12px; }
.note { font-size: 13px; color: var(--ink-3); border-left: 2px solid var(--rule); padding-left: 14px; margin: 18px 0; max-width: 66ch; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
"""

_JS = """
(function () {
  var tip = document.createElement('div');
  tip.style.cssText = 'position:fixed;pointer-events:none;opacity:0;transition:opacity .08s;' +
    'background:var(--ink);color:var(--surface);font:12px system-ui,sans-serif;' +
    'padding:5px 9px;border-radius:6px;z-index:9;white-space:nowrap';
  document.body.appendChild(tip);
  document.querySelectorAll('circle.dot').forEach(function (dot) {
    dot.addEventListener('mouseenter', function (e) {
      tip.textContent = dot.dataset.series + ': ' + dot.dataset.y + ' at ' + dot.dataset.x + ' tools';
      tip.style.opacity = '1';
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY - 30) + 'px';
    });
    dot.addEventListener('mousemove', function (e) {
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY - 30) + 'px';
    });
    dot.addEventListener('mouseleave', function () { tip.style.opacity = '0'; });
  });
})();
"""


def write_report(payload: dict[str, Any], out: Path) -> Path:
    """Render a results payload from `atg bench` as one HTML file."""
    sweep = payload["sweep"]
    sizes = sweep["sizes"]
    k = sweep["k"]
    index = {(s["strategy"], s["catalog_size"]): s for s in sweep["summaries"]}
    present = [s for s in STRATEGY_ORDER if any(st == s for st, _ in index)]

    def series_for(key: str) -> list[tuple[str, list[tuple[float, float]]]]:
        out_series = []
        for name in present:
            points = [
                (float(size), float(index[(name, size)][key]))
                for size in sizes
                if (name, size) in index
            ]
            if points:
                out_series.append((name, points))
        return out_series

    full = max(sizes)
    dis = payload["disagreement_full_catalog"]
    best = max(present, key=lambda s: index[(s, full)]["task_success"])
    baseline = index[("all_tools", full)] if ("all_tools", full) in index else None
    compression = payload["compression"]
    base_row = next(r for r in compression if r["fmt"] == "json_schema")
    best_saving = max(compression, key=lambda r: r["savings_vs_json"])

    tiles = [
        (f"{index[(best, full)]['task_success']:.0%}", f"best task success at {full} tools ({LABELS[best]})"),
        (f"{index[(best, full)]['recall_at_k']:.0%}", f"its recall@{k}, which is the number that flatters"),
        (f"{dis['recall_hit_task_fail_share']:.0%}", "attempts where the right tool was retrieved and the task still failed"),
        (f"{best_saving['savings_vs_json']:.0%}", f"tokens saved by {FMT_LABELS[best_saving['fmt']]}"),
    ]
    if baseline:
        tiles.insert(
            2,
            (
                f"{baseline['mean_prompt_tokens']:,.0f}",
                f"tokens per query to put all {full} schemas in context",
            ),
        )

    tile_html = "".join(
        f'<div class="tile"><span class="n">{_esc(n)}</span><span class="l">{_esc(l)}</span></div>'
        for n, l in tiles
    )

    charts = _line_chart(
        chart_id="success",
        title="End-to-end task success against catalog size",
        series=series_for("task_success"),
        sizes=sizes,
        y_label="task success",
    ) + _line_chart(
        chart_id="recall",
        title=f"Recall@{k} against catalog size, the same runs",
        series=series_for("recall_at_k"),
        sizes=sizes,
        y_label=f"recall@{k}",
    )

    token_chart = _line_chart(
        chart_id="tokens",
        title="Mean prompt tokens per query, log scale",
        series=series_for("mean_prompt_tokens"),
        sizes=sizes,
        y_label="tokens per query",
        y_log=True,
        y_fmt="num",
    )

    matrix_rows = []
    for name in present:
        for key, label in (("task_success", "task success"), ("recall_at_k", f"recall@{k}")):
            matrix_rows.append(
                [LABELS[name], label]
                + [
                    f"{index[(name, size)][key]:.3f}" if (name, size) in index else "-"
                    for size in sizes
                ]
            )
    matrix = _table(
        ["strategy", "metric"] + [str(s) for s in sizes],
        matrix_rows,
        caption="Every point in the two charts above, as numbers.",
    )

    token_rows = [
        [LABELS[name]]
        + [
            f"{index[(name, size)]['mean_prompt_tokens']:,.0f}" if (name, size) in index else "-"
            for size in sizes
        ]
        + [str(index[(name, full)]["stages"])]
        for name in present
    ]
    token_table = _table(
        ["strategy"] + [str(s) for s in sizes] + ["model calls"],
        token_rows,
        caption="Mean prompt tokens per query at each catalog size.",
    )

    def comp_table_for(rows: list[dict[str, Any]], caption: str) -> str:
        return _table(
            [
                "rendering",
                "tokens/query",
                "tokens/tool",
                "whole catalog",
                "task success",
                "saving",
                "accuracy delta",
            ],
            [
                [
                    FMT_LABELS.get(r["fmt"], r["fmt"]),
                    f"{r['mean_prompt_tokens']:,.0f}",
                    f"{r['tokens_per_tool']:,.0f}",
                    f"{r['full_catalog_tokens']:,}",
                    f"{r['task_success']:.3f}",
                    f"{r['savings_vs_json']:.1%}",
                    f"{r['success_delta_vs_json']:+.3f}",
                ]
                for r in rows
            ],
            caption=caption,
        )

    comp_table = comp_table_for(
        compression,
        f"Same retriever ({base_row['strategy']}), same k, same tasks, at a catalog of "
        f"{base_row['catalog_size']} tools. Only the rendering changes.",
    )
    small_rows = payload.get("compression_small_catalog") or []
    if small_rows:
        comp_table += comp_table_for(
            small_rows,
            f"The same comparison at {small_rows[0]['catalog_size']} tools, where tool choice is "
            "mostly right and the argument errors compression causes are not hidden behind it.",
        )

    tax = payload["taxonomy_full_catalog"]
    tax_rows = [
        (FAILURE_LABELS.get(mode, mode), count, tax["overall_share"].get(mode, 0.0))
        for mode, count in tax["overall"].items()
    ]
    tax_chart = _bar_chart(
        f"Failure modes across all strategies at {full} tools ({tax['total_failures']} failures)", tax_rows
    )

    per_kind = payload["per_kind_full_catalog"].get(best, {})
    kind_table = _table(
        ["task kind", "n", "task success", f"recall@{k}"],
        [
            [kind, row["n"], f"{row['task_success']:.3f}", f"{row['recall_at_k']:.3f}"]
            for kind, row in sorted(per_kind.items())
        ],
        caption=f"{LABELS[best]} at {full} tools, split by what the task was testing.",
    )

    dis_rows = [
        [FAILURE_LABELS.get(mode, mode), count]
        for mode, count in sorted(
            dis["recall_hit_task_fail_by_outcome"].items(), key=lambda kv: -kv[1]
        )
    ]
    dis_table = _table(
        ["why the task failed anyway", "count"],
        dis_rows,
        caption=(
            f"{dis['recall_hit_task_fail']} of {dis['n']} attempts at {full} tools had the right tool "
            f"in the shortlist and still did not complete the task."
        ),
    )

    body = f"""
<div class="wrap">
<h1>Tool retrieval at scale</h1>
<p class="sub">{_esc(sweep['n_tasks'])} tasks against a {_esc(sweep['catalog_size_total'])} tool catalog
(fingerprint <code>{_esc(sweep['catalog_fingerprint'])}</code>, seed {_esc(sweep['catalog_seed'])}),
k={_esc(k)}, {_esc(sweep['decider'])} model, {_esc(sweep['embedder'])} embeddings.
Generated {_esc(payload['generated_utc'])}.</p>

<div class="tiles">{tile_html}</div>

<h2>The two numbers disagree</h2>
<p>Recall@{_esc(k)} asks whether the right tool reached the model. Task success asks whether
the work got done. They are not the same question, and the gap between the two charts
below is the interesting part: at {_esc(full)} tools, {_esc(dis['recall_hit_task_fail'])}
attempts put the correct tool in front of the model and still failed
({_esc(f"{dis['recall_hit_task_fail_share']:.1%}")} of all attempts), while
{_esc(dis['success_via_non_gold_tool'])} completed the task by calling something other than
the tool recall was scored against, because a second server exposes the same operation under
another name. The reverse case, recall counting a miss while the task still succeeded, happened
{_esc(dis['recall_miss_task_pass'])} times: a substitute that does the same job reads almost
the same to a retriever, so it is nearly always retrieved alongside the tool it substitutes
for rather than instead of it.</p>
{_legend(present)}
<div class="charts">{charts}</div>
{matrix}

<h2>What it costs</h2>
<p>Putting every schema in context is the baseline that stops working, and it stops working
on price before it stops working on accuracy. Retrieval holds the prompt flat as the catalog
grows. Progressive disclosure does not: a line per tool is small, but there are still as many
lines as tools.</p>
<div class="charts">{token_chart}</div>
{token_table}

<h2>Schema compression</h2>
<p>How much of the token cost is the JSON and how much is the information. Dropping per
parameter descriptions is most of the saving and most of the accuracy cost, because argument
placement is what those descriptions are for.</p>
{comp_table}

<h2>Where it fails</h2>
{tax_chart}
{kind_table}
{dis_table}

<p class="note">Every number on this page came from running the code in this repository.
The model in the loop is the deterministic simulator unless the subtitle says otherwise;
MEASUREMENTS.md reports the real-model run separately, with model id and date.</p>
</div>
"""

    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Tool retrieval at scale</title>"
        f"<style>{_CSS}</style></head><body>{body}<script>{_JS}</script></body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build the HTML report from a results file.")
    parser.add_argument("--results", default="out/results.json")
    parser.add_argument("--out", default="out/report.html")
    args = parser.parse_args(argv)
    path = write_report(json.loads(Path(args.results).read_text()), Path(args.out))
    print(path)
    return 0
