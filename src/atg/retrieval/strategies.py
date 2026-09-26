"""The five retrieval strategies, behind one interface.

All of them are backed by indexes built once over the whole catalog and then
masked down to whatever subset a measurement point is using. That mirrors how a
gateway actually works (index the estate once, serve many clients) and it is what
makes an 8-point size sweep over 240 queries finish in seconds rather than hours.
Corpus statistics, idf included, therefore come from the full catalog at every
size, which is stated here because it would otherwise be a silent confound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from atg.catalog import Catalog, Tool
from atg.render import Format, block_tokens, render
from atg.retrieval.bm25 import BM25Index
from atg.retrieval.embed import Embedder, EmbeddingIndex, fit_hashing_embedder
from atg.retrieval.text import tokenize
from atg.tokens import count_tokens

Allowed = frozenset[str] | None


@dataclass(slots=True)
class Shortlist:
    """What a strategy hands the agent, and what it cost to get there."""

    strategy: str
    candidates: tuple[str, ...]
    ranking: tuple[str, ...]
    prompt_tokens: int
    fmt: Format
    stages: int = 1
    stage_tokens: tuple[int, ...] = ()
    notes: str = ""

    def gold_rank(self, gold: str) -> int | None:
        try:
            return self.ranking.index(gold) + 1
        except ValueError:
            return None


class Strategy(Protocol):
    name: str
    fmt: Format

    def retrieve(self, query: str, k: int, allowed: Allowed = None) -> Shortlist: ...


def _tools(catalog: Catalog, names: Sequence[str]) -> list[Tool]:
    return [catalog.get(n) for n in names]


def _mask(scores: dict[str, float], allowed: Allowed) -> dict[str, float]:
    if allowed is None:
        return scores
    return {n: s for n, s in scores.items() if n in allowed}


def _rank(scores: dict[str, float]) -> tuple[str, ...]:
    return tuple(n for n, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])))


# --------------------------------------------------------------------------


@dataclass(slots=True)
class AllTools:
    """The baseline that stops working: every schema, every call."""

    catalog: Catalog
    fmt: Format = "json_schema"
    name: str = "all_tools"

    def retrieve(self, query: str, k: int, allowed: Allowed = None) -> Shortlist:
        names = tuple(t.name for t in self.catalog.tools if allowed is None or t.name in allowed)
        tools = _tools(self.catalog, names)
        return Shortlist(
            self.name, names, names, block_tokens(tools, self.fmt), self.fmt, notes="no retrieval"
        )


@dataclass(slots=True)
class BM25Strategy:
    catalog: Catalog
    index: BM25Index
    fmt: Format = "json_schema"
    name: str = "bm25"

    def retrieve(self, query: str, k: int, allowed: Allowed = None) -> Shortlist:
        ranking = _rank(_mask(self.index.score(query), allowed))
        chosen = ranking[:k]
        return Shortlist(
            self.name, chosen, ranking, block_tokens(_tools(self.catalog, chosen), self.fmt), self.fmt
        )


@dataclass(slots=True)
class EmbeddingStrategy:
    catalog: Catalog
    index: EmbeddingIndex
    fmt: Format = "json_schema"
    name: str = "embedding"

    def retrieve(self, query: str, k: int, allowed: Allowed = None) -> Shortlist:
        ranking = _rank(_mask(self.index.score(query), allowed))
        chosen = ranking[:k]
        return Shortlist(
            self.name, chosen, ranking, block_tokens(_tools(self.catalog, chosen), self.fmt), self.fmt
        )


# --------------------------------------------------------------------------
# Stage two: a reranker that looks at things neither first-stage scorer can.


_VERB_CUES: dict[str, tuple[str, ...]] = {
    "create": ("create", "new", "add", "register", "open", "set", "stage", "queue"),
    "get": ("get", "look", "fetch", "show", "read", "pull"),
    "list": ("list", "all", "every", "enumerate", "page"),
    "update": ("update", "change", "edit", "patch", "adjust", "amend", "set"),
    "delete": ("delete", "remove", "purge", "wipe", "erase", "drop", "rid"),
    "search": ("search", "find", "hunt", "look", "match", "track"),
    "export": ("export", "dump", "download", "extract", "file", "csv"),
    "archive": ("archive", "retire", "deactivate", "keep"),
}

_TYPE_CUES: dict[str, tuple[str, ...]] = {
    "email": ("@",),
    "url": ("http",),
    "money": ("$", "."),
    "date": ("202", "T0", "T1", "T2"),
}


def rerank_features(tool: Tool, query: str) -> dict[str, float]:
    """Signals a first-stage scorer does not have: verb agreement, argument type
    compatibility, and whether the query actually asserts the tool's narrow
    precondition."""
    lower = query.lower()
    terms = set(tokenize(query))

    cues = _VERB_CUES.get(tool.verb, ())
    verb_match = 1.0 if any(c in terms or c in lower for c in cues) else 0.0

    satisfiable = 0
    for param in tool.required_params:
        cue = _TYPE_CUES.get(param.kind)
        if param.kind in ("text", "id", "path", "int", "bool", "enum"):
            satisfiable += 1
        elif cue and any(c in query for c in cue):
            satisfiable += 1
    coverage = satisfiable / max(1, len(tool.required_params))

    if tool.precondition is None:
        precondition = 0.0
    else:
        cue_terms = set(tokenize(tool.precondition.query_phrase))
        overlap = len(cue_terms & terms) / max(1, len(cue_terms))
        # Asserting the condition is evidence for; saying nothing is evidence
        # against, since a conditional tool is wrong by default.
        precondition = 2.0 * overlap - 0.9

    legacy = -0.35 if tool.alias_of is not None else 0.0
    return {
        "verb": verb_match,
        "coverage": coverage,
        "precondition": precondition,
        "legacy": legacy,
    }


RERANK_WEIGHTS: dict[str, float] = {
    "fusion": 1.0,
    "verb": 0.55,
    "coverage": 0.30,
    "precondition": 0.85,
    "legacy": 1.0,
}


@dataclass(slots=True)
class RerankStrategy:
    """Retrieve wide with both first-stage scorers, then rerank the union."""

    catalog: Catalog
    bm25: BM25Index
    embeddings: EmbeddingIndex
    candidate_pool: int = 40
    fmt: Format = "json_schema"
    name: str = "rerank"

    def retrieve(self, query: str, k: int, allowed: Allowed = None) -> Shortlist:
        lex = _mask(self.bm25.score(query), allowed)
        dense = _mask(self.embeddings.score(query), allowed)
        lex_rank = {n: i for i, n in enumerate(_rank(lex))}
        dense_rank = {n: i for i, n in enumerate(_rank(dense))}
        pool = set(list(lex_rank)[: self.candidate_pool]) | set(list(dense_rank)[: self.candidate_pool])

        scored: dict[str, float] = {}
        for name in pool:
            # Reciprocal rank fusion over the two first stages, then the features.
            fusion = 0.0
            if name in lex_rank:
                fusion += 1.0 / (10.0 + lex_rank[name])
            if name in dense_rank:
                fusion += 1.0 / (10.0 + dense_rank[name])
            feats = rerank_features(self.catalog.get(name), query)
            feats["fusion"] = fusion * 60.0
            scored[name] = sum(RERANK_WEIGHTS[f] * v for f, v in feats.items())

        ranking = _rank(scored)
        chosen = ranking[:k]
        return Shortlist(
            self.name,
            chosen,
            ranking,
            block_tokens(_tools(self.catalog, chosen), self.fmt),
            self.fmt,
            notes=f"pool={self.candidate_pool}",
        )


# --------------------------------------------------------------------------

ShortlistSelector = Callable[[str, list[tuple[str, str]], int], list[str]]
"""(query, [(tool name, one-line summary)], budget) -> chosen tool names."""


@dataclass(slots=True)
class ProgressiveStrategy:
    """Show the model one line per tool, let it ask for the schemas it wants.

    Two model round trips instead of one, and the first prompt still scales with
    the catalog. The question the measurement answers is whether a line per tool
    is cheap enough for that to be worth it.
    """

    catalog: Catalog
    selector: ShortlistSelector
    fmt: Format = "json_schema"
    name: str = "progressive"
    summary_fmt: Format = "oneline"

    def _summary_lines(self, names: Sequence[str]) -> list[tuple[str, str]]:
        return [(n, self.catalog.get(n).summary) for n in names]

    def retrieve(self, query: str, k: int, allowed: Allowed = None) -> Shortlist:
        names = tuple(t.name for t in self.catalog.tools if allowed is None or t.name in allowed)
        lines = self._summary_lines(names)
        stage1 = "Tool index, one line each:\n" + "\n".join(f"{n}: {s}" for n, s in lines)
        stage1_tokens = count_tokens(stage1)

        chosen = tuple(self.selector(query, lines, k))
        stage2_tokens = count_tokens(
            "\n".join(render(self.catalog.get(n), self.fmt) for n in chosen)
        )
        # The ranking the model produced is all the ordering information there
        # is; anything it did not ask for is unranked.
        return Shortlist(
            self.name,
            chosen,
            chosen,
            stage1_tokens + stage2_tokens,
            self.fmt,
            stages=2,
            stage_tokens=(stage1_tokens, stage2_tokens),
        )


# --------------------------------------------------------------------------


@dataclass(slots=True)
class Indexes:
    """Built once per catalog and shared by every strategy and every size."""

    catalog: Catalog
    bm25: BM25Index
    embeddings: EmbeddingIndex
    embedder_name: str = field(default="")

    @classmethod
    def build(cls, catalog: Catalog, embedder: Embedder | None = None) -> "Indexes":
        emb = embedder or fit_hashing_embedder(catalog)
        return cls(catalog, BM25Index(catalog), EmbeddingIndex(catalog, emb), emb.name)


def build_strategies(
    indexes: Indexes,
    selector: ShortlistSelector,
    *,
    fmt: Format = "json_schema",
) -> dict[str, Strategy]:
    cat = indexes.catalog
    return {
        "all_tools": AllTools(cat, fmt),
        "bm25": BM25Strategy(cat, indexes.bm25, fmt),
        "embedding": EmbeddingStrategy(cat, indexes.embeddings, fmt),
        "rerank": RerankStrategy(cat, indexes.bm25, indexes.embeddings, fmt=fmt),
        "progressive": ProgressiveStrategy(cat, selector, fmt),
    }


STRATEGY_ORDER: tuple[str, ...] = ("all_tools", "bm25", "embedding", "rerank", "progressive")
