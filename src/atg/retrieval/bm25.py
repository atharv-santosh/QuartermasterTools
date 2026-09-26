"""Okapi BM25 over the tool catalog, implemented here rather than pulled in.

Fielded: a query term matching a tool's name should count for more than the same
term buried in a rate-limit sentence, so each field is scored separately and the
scores are combined with fixed weights.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from atg.catalog import Catalog, Tool
from atg.retrieval.text import tokenize

K1 = 1.4
B = 0.72

FIELD_WEIGHTS: dict[str, float] = {
    "name": 3.0,
    "summary": 1.6,
    "description": 1.0,
    "params": 0.8,
    "tags": 0.5,
}


def _fields(tool: Tool) -> dict[str, str]:
    return {
        "name": tool.name.replace("_", " ").replace(".", " "),
        "summary": tool.summary,
        "description": tool.full_description(),
        "params": " ".join(p.name.replace("_", " ") for p in tool.params),
        "tags": " ".join((*tool.tags, tool.domain, tool.resource.replace("_", " "))),
    }


@dataclass(slots=True)
class _FieldIndex:
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    avg_len: float = 0.0

    def finish(self) -> None:
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0


class BM25Index:
    """Fielded BM25 index over a catalog. Built once, queried many times."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.names: list[str] = [t.name for t in catalog.tools]
        self._n = len(self.names)
        self._fields: dict[str, _FieldIndex] = {k: _FieldIndex() for k in FIELD_WEIGHTS}
        for doc_id, tool in enumerate(catalog.tools):
            for fname, text in _fields(tool).items():
                terms = tokenize(text)
                idx = self._fields[fname]
                idx.lengths.append(len(terms))
                for term, tf in Counter(terms).items():
                    idx.postings.setdefault(term, []).append((doc_id, tf))
        for idx in self._fields.values():
            idx.finish()

    def _idf(self, idx: _FieldIndex, term: str) -> float:
        df = len(idx.postings.get(term, ()))
        if df == 0:
            return 0.0
        # Robertson/Sparck-Jones idf with the +1 that keeps it non-negative for
        # terms appearing in more than half the collection.
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> dict[str, float]:
        """Sparse score map, only over tools that match at least one term."""
        terms = tokenize(query)
        if not terms:
            return {}
        acc: dict[int, float] = {}
        for fname, weight in FIELD_WEIGHTS.items():
            idx = self._fields[fname]
            if idx.avg_len == 0.0:
                continue
            for term in set(terms):
                postings = idx.postings.get(term)
                if not postings:
                    continue
                idf = self._idf(idx, term)
                for doc_id, tf in postings:
                    dl = idx.lengths[doc_id]
                    denom = tf + K1 * (1.0 - B + B * dl / idx.avg_len)
                    acc[doc_id] = acc.get(doc_id, 0.0) + weight * idf * (tf * (K1 + 1.0)) / denom
        return {self.names[doc_id]: s for doc_id, s in acc.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scored = self.score(query)
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
