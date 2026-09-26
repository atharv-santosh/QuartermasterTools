from __future__ import annotations

import json

import pytest

from atg.catalog import CONFUSABLE_CLUSTERS, DOMAINS, build_catalog


def test_catalog_is_large_enough_and_reproducible() -> None:
    a = build_catalog()
    b = build_catalog()
    assert len(a) >= 1000
    assert a.fingerprint() == b.fingerprint()
    assert [t.name for t in a.tools] == [t.name for t in b.tools]


def test_a_different_seed_gives_a_different_catalog() -> None:
    assert build_catalog(seed=1).fingerprint() != build_catalog(seed=2).fingerprint()


def test_tool_names_are_unique() -> None:
    catalog = build_catalog()
    assert len(catalog.by_name) == len(catalog.tools)


def test_every_domain_and_server_is_represented() -> None:
    catalog = build_catalog()
    assert {t.domain for t in catalog.tools} >= set(DOMAINS)
    assert len(catalog.servers()) >= len(DOMAINS)


def test_confusable_families_are_present_and_populated() -> None:
    catalog = build_catalog()
    for cluster in CONFUSABLE_CLUSTERS:
        members = catalog.families[cluster["family"]]
        for member in cluster["members"]:
            assert member["name"] in members
        assert len(members) >= 4


def test_near_duplicates_really_are_near_duplicates() -> None:
    """The email family must be lexically close, or the benchmark is not testing
    disambiguation at all."""
    catalog = build_catalog()
    names = catalog.families["email_send"]
    assert "send_email" in names
    assert "send_bulk_email" in names
    assert "email_draft_create" in names
    for name in names:
        tokens = set(name.replace(".", "_").split("_"))
        assert tokens & {"email", "notify", "mail"}


def test_aliases_point_at_a_real_tool_and_share_its_arguments() -> None:
    catalog = build_catalog()
    aliases = [t for t in catalog.tools if t.alias_of]
    assert len(aliases) >= 20
    for alias in aliases:
        source = catalog.get(alias.alias_of or "")
        assert alias.name != source.name
        assert alias.server != source.server
        assert [p.name for p in alias.params] == [p.name for p in source.params]


def test_preconditions_are_domain_plausible() -> None:
    catalog = build_catalog()
    conditional = [t for t in catalog.tools if t.precondition and t.alias_of is None]
    assert len(conditional) >= 20
    for tool in conditional:
        pre = tool.precondition
        assert pre is not None
        if tool.family.count("."):
            assert pre.applies_to(tool.domain, tool.verb), f"{tool.name} / {pre.marker}"


def test_precondition_wording_differs_between_schema_and_query() -> None:
    """A query that quoted the schema would hand lexical retrieval the answer."""
    from atg.catalog import PRECONDITIONS

    for pre in PRECONDITIONS:
        assert pre.phrase != pre.query_phrase
        shared = set(pre.phrase.lower().split()) & set(pre.query_phrase.lower().split())
        assert len(shared) <= 4, pre.marker


def test_json_schema_is_valid_shaped() -> None:
    catalog = build_catalog()
    for tool in catalog.tools:
        schema = tool.json_schema()
        assert schema["type"] == "object"
        assert set(schema["required"]) <= set(schema["properties"])
        json.dumps(schema)
        for spec in schema["properties"].values():
            assert spec["type"] in ("string", "integer", "number", "boolean")
            assert spec["description"]


def test_every_tool_has_at_least_one_parameter_and_a_summary() -> None:
    for tool in build_catalog().tools:
        assert tool.summary.endswith(".")
        assert tool.params or tool.verb == "list"


@pytest.mark.parametrize("size", [1, 10, 50, 500])
def test_subset_is_nested_and_the_right_size(size: int) -> None:
    catalog = build_catalog()
    smaller = catalog.subset(size)
    assert len(smaller) == size
    assert set(t.name for t in smaller.tools) <= set(catalog.by_name)


def test_subset_keeps_what_it_is_told_to_without_growing() -> None:
    catalog = build_catalog()
    keep = ["send_email", "refund_charge_partial", catalog.tools[-1].name]
    smaller = catalog.subset(20, keep=keep)
    assert len(smaller) == 20
    for name in keep:
        assert name in smaller.by_name
