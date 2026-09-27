from __future__ import annotations

import math

import pytest

from atg.catalog import build_catalog
from atg.llm_sim import select_shortlist
from atg.retrieval.bm25 import BM25Index
from atg.retrieval.embed import (
    cosine,
    fit_hashing_embedder,
    replay_embedder,
)
from atg.retrieval.strategies import Indexes, build_strategies, rerank_features
from atg.retrieval.text import tokenize
from atg.tasks import build_tasks


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


@pytest.fixture(scope="module")
def indexes(catalog):
    return Indexes.build(catalog)


# -- BM25 ------------------------------------------------------------------


def test_bm25_ranks_an_exact_name_match_first(catalog) -> None:
    index = BM25Index(catalog)
    top, _ = index.search("send email message immediately", 1)[0]
    assert top in catalog.families["email_send"]


def test_bm25_scores_are_non_negative(catalog) -> None:
    index = BM25Index(catalog)
    for score in index.score("refund part of a charge").values():
        assert score >= 0.0


def test_bm25_saturates_term_frequency(catalog) -> None:
    """The whole point of k1: a term repeated twenty times must not score twenty
    times as high as a term appearing once."""
    index = BM25Index(catalog)
    once = index.score("refund")
    many = index.score(" ".join(["refund"] * 20))
    name = max(once, key=lambda n: once[n])
    assert many[name] == pytest.approx(once[name], rel=1e-9)


def test_bm25_idf_prefers_the_rarer_term(catalog) -> None:
    index = BM25Index(catalog)
    rare = index._idf(index._fields["description"], tokenize("dkim")[0] if tokenize("dkim") else "dkim")
    common = index._idf(index._fields["description"], "record")
    assert rare > common


def test_bm25_returns_nothing_for_a_query_of_only_stopwords(catalog) -> None:
    assert BM25Index(catalog).search("the and of a", 5) == []


# -- embeddings ------------------------------------------------------------


def test_hashing_embedder_is_deterministic_and_normalized(catalog) -> None:
    embedder = fit_hashing_embedder(catalog)
    first = embedder.encode(["refund part of a charge"])[0]
    second = embedder.encode(["refund part of a charge"])[0]
    assert first == second
    assert math.sqrt(sum(v * v for v in first)) == pytest.approx(1.0, abs=1e-9)


def test_embedding_similarity_survives_a_word_ending(catalog) -> None:
    """Character n-grams are why this is not just exact matching."""
    embedder = fit_hashing_embedder(catalog)
    a, b, c = embedder.encode(["archive a contact", "archiving contacts", "provision a firewall rule"])
    assert cosine(a, b) > cosine(a, c)


def test_embedding_index_ranks_the_right_family(catalog, indexes) -> None:
    top = [name for name, _ in indexes.embeddings.search("refund the whole charge amount", 12)]
    assert set(top) & set(catalog.families["payment_refund"])


def test_local_embedder_cannot_bridge_a_purely_semantic_gap(catalog, indexes) -> None:
    """A documented limitation, asserted so it stays documented.

    The local embedder is a hashed tf-idf projection, not a trained model. A
    paraphrase sharing no substrings with the tool ("give the customer all their
    money back" for a refund) is beyond it, and no amount of n-gram tolerance
    fixes that. This is why MEASUREMENTS.md does not read the embedding column
    as an upper bound on what dense retrieval can do, and why VertexEmbedder
    exists as an optional swap.
    """
    top = [name for name, _ in indexes.embeddings.search("give the customer all their money back", 12)]
    assert not set(top) & set(catalog.families["payment_refund"])


def test_replay_embedder_refuses_text_it_never_recorded(catalog) -> None:
    embedder = fit_hashing_embedder(catalog)
    store = {"known": embedder.encode(["known"])[0]}
    replay = replay_embedder(store, "replay")
    assert replay.encode(["known"])[0] == store["known"]
    with pytest.raises(KeyError):
        replay.encode(["never seen"])


# -- reranking -------------------------------------------------------------


def test_rerank_penalises_a_conditional_tool_the_query_did_not_ask_for(catalog) -> None:
    partial = catalog.get("refund_charge_partial")
    asked = rerank_features(partial, "Send back 48.50 of charge c_1 and leave the rest, only part of it")
    unasked = rerank_features(partial, "Give the customer all their money back on charge c_1")
    assert asked["precondition"] > unasked["precondition"]
    assert unasked["precondition"] < 0


def test_rerank_prefers_the_canonical_tool_over_the_legacy_bridge(catalog) -> None:
    alias = next(t for t in catalog.tools if t.alias_of)
    source = catalog.get(alias.alias_of or "")
    query = f"work with the {source.resource.replace('_', ' ')}"
    assert rerank_features(alias, query)["legacy"] < rerank_features(source, query)["legacy"]


def test_rerank_recall_is_at_least_as_good_as_either_first_stage(catalog, indexes) -> None:
    tasks = build_tasks(catalog, n=120)
    registry = build_strategies(indexes, select_shortlist)
    scores = {}
    for name in ("bm25", "embedding", "rerank"):
        hits = sum(1 for t in tasks if t.gold_tool in registry[name].retrieve(t.query, 8).candidates)
        scores[name] = hits / len(tasks)
    assert scores["rerank"] >= max(scores["bm25"], scores["embedding"])


# -- strategies ------------------------------------------------------------


def test_all_tools_returns_everything_and_costs_the_most(catalog, indexes) -> None:
    registry = build_strategies(indexes, select_shortlist)
    everything = registry["all_tools"].retrieve("anything", 8)
    shortlisted = registry["rerank"].retrieve("anything", 8)
    assert len(everything.candidates) == len(catalog)
    assert everything.prompt_tokens > 20 * shortlisted.prompt_tokens


def test_strategies_respect_the_allowed_subset(catalog, indexes) -> None:
    allowed = frozenset(t.name for t in catalog.tools[:40])
    registry = build_strategies(indexes, select_shortlist)
    for name, strategy in registry.items():
        result = strategy.retrieve("archive a contact record", 8, allowed)
        assert set(result.candidates) <= allowed, name


def test_shortlist_never_exceeds_k(catalog, indexes) -> None:
    registry = build_strategies(indexes, select_shortlist)
    for name, strategy in registry.items():
        if name == "all_tools":
            continue
        assert len(strategy.retrieve("refund a charge", 5).candidates) <= 5, name


def test_progressive_disclosure_costs_two_calls_and_scales_with_the_catalog(catalog, indexes) -> None:
    registry = build_strategies(indexes, select_shortlist)
    progressive = registry["progressive"]
    small = progressive.retrieve("refund a charge", 5, frozenset(t.name for t in catalog.tools[:30]))
    large = progressive.retrieve("refund a charge", 5)
    assert small.stages == large.stages == 2
    assert large.prompt_tokens > 4 * small.prompt_tokens
    assert len(large.stage_tokens) == 2


def test_gold_rank_is_one_based_and_none_when_unranked(catalog, indexes) -> None:
    registry = build_strategies(indexes, select_shortlist)
    result = registry["bm25"].retrieve("send a single email message immediately", 8)
    assert result.gold_rank(result.ranking[0]) == 1
    assert result.gold_rank("not_a_tool_name") is None
