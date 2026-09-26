"""Token estimator error against recorded Vertex `count_tokens` results.

The fixture stores recipes, not text: a format and a list of tool names. The text
is regenerated from the catalog, so the check is that the estimator still agrees
with the real tokenizer on renderings this repo produces today, not that a blob of
saved strings still hashes the same.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atg.catalog import build_catalog
from atg.render import Format, render, render_many
from atg.tokens import count_tokens

CALIBRATION_FIXTURE = Path("tests/fixtures/token_calibration.json")


@dataclass(frozen=True, slots=True)
class Sample:
    fmt: Format
    tools: tuple[str, ...]
    real_tokens: int
    split: str

    def text(self, catalog: Any) -> str:
        if len(self.tools) == 1:
            return render(catalog.get(self.tools[0]), self.fmt)
        return render_many([catalog.get(n) for n in self.tools], self.fmt)


def load_samples(path: Path) -> tuple[dict[str, Any], list[Sample]]:
    payload = json.loads(path.read_text())
    samples = [
        Sample(s["fmt"], tuple(s["tools"]), int(s["real_tokens"]), s.get("split", "fit"))
        for s in payload["samples"]
    ]
    return payload, samples


def report_calibration(path: Path = CALIBRATION_FIXTURE) -> dict[str, Any]:
    """Mean and tail error of the offline estimator, by split and by format."""
    meta, samples = load_samples(path)
    catalog = build_catalog(seed=meta.get("catalog_seed", 20260925))
    if catalog.fingerprint() != meta.get("catalog_fingerprint"):
        return {
            "error": "catalog fingerprint moved since the fixture was recorded",
            "fixture_fingerprint": meta.get("catalog_fingerprint"),
            "current_fingerprint": catalog.fingerprint(),
        }

    rows: list[dict[str, Any]] = []
    for sample in samples:
        estimate = count_tokens(sample.text(catalog))
        rows.append(
            {
                "fmt": sample.fmt,
                "split": sample.split,
                "real": sample.real_tokens,
                "estimate": estimate,
                "abs_pct_err": abs(estimate - sample.real_tokens) / sample.real_tokens,
                "ratio": estimate / sample.real_tokens,
            }
        )

    def stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {}
        errs = sorted(r["abs_pct_err"] for r in subset)
        return {
            "n": len(subset),
            "mean_abs_pct_err": round(statistics.mean(errs) * 100, 2),
            "p95_abs_pct_err": round(errs[min(len(errs) - 1, int(0.95 * len(errs)))] * 100, 2),
            "max_abs_pct_err": round(errs[-1] * 100, 2),
            "mean_ratio": round(statistics.mean(r["ratio"] for r in subset), 4),
        }

    return {
        "model": meta.get("model"),
        "recorded_utc": meta.get("recorded_utc"),
        "overall": stats(rows),
        "by_split": {s: stats([r for r in rows if r["split"] == s]) for s in ("fit", "holdout")},
        "by_format": {f: stats([r for r in rows if r["fmt"] == f]) for f in sorted({r["fmt"] for r in rows})},
    }
