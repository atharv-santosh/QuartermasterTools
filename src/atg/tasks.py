"""The benchmark suite: queries with a gold tool, gold arguments, and a grader.

Queries are phrased with synonyms for the verb and, where one exists, for the
resource, so a query is not a lexical copy of the tool it should reach. Every
argument value the grader expects appears verbatim in the query text, which is
what makes "the agent invented this value" a decidable question rather than a
judgement call.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from atg.catalog import CLUSTER_INTENTS, Catalog, Param, Tool, slot_value

TaskKind = Literal["plain", "family", "conditional", "aliased", "underspecified"]

# Resource words a user would actually say instead of the schema's word. The gap
# is the point: lexical retrieval has to bridge it or lose.
RESOURCE_SYNONYMS: dict[str, str] = {
    "purchase_order": "PO",
    "timeoff_request": "vacation request",
    "sku": "product code",
    "stock_level": "inventory count",
    "journal_entry": "ledger line",
    "expense_report": "expense claim",
    "role_binding": "permission grant",
    "api_key": "access token",
    "firewall_rule": "network rule",
    "audit_log": "access history",
    "training_job": "model training run",
    "model_version": "trained model",
    "feature_view": "feature table",
    "device_group": "device fleet",
    "firmware_bundle": "firmware image",
    "ota_job": "over the air update",
    "satisfaction_survey": "CSAT survey",
    "sla_policy": "response time policy",
    "knowledge_article": "help centre article",
    "room_booking": "meeting room reservation",
    "recurring_series": "repeating meeting",
    "working_hours": "office hours",
    "signed_url": "temporary link",
    "lifecycle_rule": "retention rule",
    "storage_class": "storage tier",
    "event_stream": "event feed",
    "retention_curve": "retention chart",
    "attribution_model": "credit model",
    "alert_rule": "monitor",
    "config_map": "config bundle",
    "autoscaler": "scaling policy",
    "connection_pool": "db pooler",
    "query_job": "sql job",
    "audience_list": "target list",
    "utm_link": "tracking link",
    "drip_sequence": "email sequence",
    "creative_asset": "ad creative",
    "landing_page": "campaign page",
    "quiet_hours": "do not disturb window",
    "delivery_receipt": "delivery confirmation",
    "subscription_pref": "notification preference",
    "forwarding_rule": "auto forward rule",
    "vacation_responder": "out of office reply",
    "cycle_count": "stock audit",
    "carrier_rate": "shipping rate",
    "cost_center": "budget code",
    "reconciliation": "month end match",
    "job_requisition": "open role",
    "org_unit": "department",
    "escalation_path": "escalation chain",
}

_VERB_PHRASES: dict[str, list[str]] = {
    "create": ["Set up a new", "Add a", "Register a new", "Open a", "I need a new"],
    "get": ["Pull up the", "Look up the", "Fetch the", "Show me the", "Read back the"],
    "list": ["Give me all the", "Enumerate the", "Page through the", "Show every"],
    "update": ["Change the", "Amend the", "Patch the", "Adjust the"],
    "delete": ["Get rid of the", "Wipe the", "Permanently remove the", "Purge the"],
    "search": ["Hunt for", "Look for any", "Track down the", "Find the"],
    "export": ["Dump the", "Pull a file of the", "Ship the", "Extract the"],
    "archive": ["Put the", "Move the", "Retire the"],
}

_VERB_TAILS: dict[str, str] = {
    "archive": "into the archive",
}

_PARAM_PHRASES: dict[str, str] = {
    "name": "called",
    "query": "matching",
    "limit": "capped at",
    "format": "as",
    "fields": "setting",
    "amount": "for",
    "to": "to",
    "recipients": "to",
    "user": "to",
    "subject": "subject line",
    "body": "saying",
    "text": "saying",
    "channel": "in channel",
    "service": "for service",
    "image": "on image",
    "bucket": "in bucket",
    "key": "at key",
    "dest": "to",
    "template_id": "using template",
    "webhook_url": "via webhook",
    "post_at": "at",
    "effective_at": "effective",
    "owner_email": "owned by",
    "callback_url": "calling back to",
    "path": "at path",
}


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    query: str
    gold_tool: str
    acceptable: tuple[str, ...]
    slots: tuple[tuple[str, str], ...]
    kind: TaskKind
    confusable_with: tuple[str, ...] = ()
    missing_param: str | None = None

    @property
    def slot_map(self) -> dict[str, str]:
        return dict(self.slots)

    def required_tools(self) -> tuple[str, ...]:
        """Tools that must be present in a catalog for this task to be scorable."""
        return tuple({self.gold_tool, *self.acceptable, *self.confusable_with})


@dataclass(slots=True)
class ToolCall:
    tool: str
    args: dict[str, str]


@dataclass(slots=True)
class Clarify:
    """The agent declining to invent a value it was never given."""

    tool: str
    missing: str


Action = ToolCall | Clarify | None


Outcome = Literal[
    "success",
    "wrong_tool_selected",
    "near_duplicate_confusion",
    "below_cutoff",
    "deep_retrieval_miss",
    "hallucinated_argument",
    "missing_argument",
    "wrong_argument_value",
    "no_action",
]

FAILURE_ORDER: tuple[Outcome, ...] = (
    "wrong_tool_selected",
    "near_duplicate_confusion",
    "below_cutoff",
    "deep_retrieval_miss",
    "hallucinated_argument",
    "missing_argument",
    "wrong_argument_value",
    "no_action",
)


def _quote(param: Param, value: str) -> str:
    return f"'{value}'" if param.kind in ("text", "path") else value


def _phrase_param(param: Param, value: str) -> str:
    lead = _PARAM_PHRASES.get(param.name)
    if lead is None:
        lead = "id" if param.name.endswith("_id") else param.name.replace("_", " ")
    return f"{lead} {_quote(param, value)}"


def _human_resource(tool: Tool) -> str:
    return RESOURCE_SYNONYMS.get(tool.resource, tool.resource.replace("_", " "))


def _build_query(
    rng: random.Random, tool: Tool, values: dict[str, str], *, omit: str | None, condition: str | None
) -> str:
    """Phrase one request. Cluster members carry their own intent template."""
    intent = CLUSTER_INTENTS.get(tool.name)
    if intent is not None:
        by_name = {p.name: p for p in tool.params}
        filled = intent
        for name, value in values.items():
            if name == omit:
                continue
            filled = filled.replace(f"{{{name}}}", _quote(by_name[name], value))
        # Any slot left unfilled belongs to an omitted parameter; drop the phrase
        # around it rather than leaving a literal brace in the query.
        for name in values:
            filled = filled.replace(f" {{{name}}}", "").replace(f"{{{name}}}", "")
        return filled.strip().rstrip(",") + "."

    verb = tool.verb if tool.verb in _VERB_PHRASES else "create"
    lead = rng.choice(_VERB_PHRASES[verb])
    noun = _human_resource(tool)
    parts = [f"{lead} {noun}"]
    tail = _VERB_TAILS.get(verb)
    if tail:
        parts.append(tail)
    clauses = [
        _phrase_param(p, values[p.name])
        for p in tool.required_params
        if p.name in values and p.name != omit
    ]
    if clauses:
        parts.append(", ".join(clauses))
    if condition:
        parts.append(f"- {condition}")
    if rng.random() < 0.4:
        parts.append(f"in the {tool.domain} system")
    return " ".join(parts).replace("  ", " ").strip() + "."


def _values_for(rng: random.Random, tool: Tool) -> dict[str, str]:
    return {p.name: slot_value(rng, p) for p in tool.required_params}


def build_tasks(catalog: Catalog, *, n: int = 240, seed: int = 424242) -> tuple[Task, ...]:
    """Generate a reproducible task suite against a catalog."""
    rng = random.Random(seed)
    tools = list(catalog.tools)
    cluster_tools = [t for t in tools if not t.family.count(".") and t.alias_of is None]
    conditional_tools = [t for t in tools if t.precondition is not None and t.alias_of is None]
    aliased_sources = [t for t in tools if t.alias_of is not None]
    plain_pool = [t for t in tools if t.alias_of is None and t.family.count(".")]

    plan: list[TaskKind] = (
        ["family"] * int(n * 0.20)
        + ["conditional"] * int(n * 0.12)
        + ["aliased"] * int(n * 0.15)
        + ["underspecified"] * int(n * 0.12)
    )
    plan += ["plain"] * (n - len(plan))
    rng.shuffle(plan)

    tasks: list[Task] = []
    for i, kind in enumerate(plan):
        tid = f"t{i:04d}"
        if kind == "family" and cluster_tools:
            tool = rng.choice(cluster_tools)
            siblings = tuple(x for x in catalog.families[tool.family] if x != tool.name)
            values = _values_for(rng, tool)
            query = _build_query(rng, tool, values, omit=None, condition=None)
            tasks.append(
                Task(tid, query, tool.name, (tool.name,), tuple(values.items()), "family", siblings)
            )
        elif kind == "conditional" and conditional_tools:
            tool = rng.choice(conditional_tools)
            assert tool.precondition is not None
            siblings = tuple(x for x in catalog.families[tool.family] if x != tool.name)
            values = _values_for(rng, tool)
            query = _build_query(rng, tool, values, omit=None, condition=tool.precondition.query_phrase)
            tasks.append(
                Task(tid, query, tool.name, (tool.name,), tuple(values.items()), "conditional", siblings)
            )
        elif kind == "aliased" and aliased_sources:
            alias = rng.choice(aliased_sources)
            assert alias.alias_of is not None
            canonical = catalog.get(alias.alias_of)
            values = _values_for(rng, canonical)
            query = _build_query(rng, canonical, values, omit=None, condition=None)
            tasks.append(
                Task(
                    tid,
                    query,
                    canonical.name,
                    (canonical.name, alias.name),
                    tuple(values.items()),
                    "aliased",
                )
            )
        elif kind == "underspecified":
            candidates = [t for t in plain_pool if len(t.required_params) >= 2]
            tool = rng.choice(candidates)
            values = _values_for(rng, tool)
            omit = rng.choice([p.name for p in tool.required_params])
            query = _build_query(rng, tool, values, omit=omit, condition=None)
            kept = tuple((k, v) for k, v in values.items() if k != omit)
            tasks.append(
                Task(tid, query, tool.name, (tool.name,), kept, "underspecified", missing_param=omit)
            )
        else:
            tool = rng.choice(plain_pool)
            values = _values_for(rng, tool)
            condition = tool.precondition.query_phrase if tool.precondition else None
            query = _build_query(rng, tool, values, omit=None, condition=condition)
            alias = next((a.name for a in aliased_sources if a.alias_of == tool.name), None)
            acceptable = (tool.name, alias) if alias else (tool.name,)
            tasks.append(
                Task(tid, query, tool.name, acceptable, tuple(values.items()), "plain")
            )
    return tuple(tasks)


@dataclass(slots=True)
class Grade:
    outcome: Outcome
    success: bool
    detail: str = ""
    hallucinated: tuple[str, ...] = field(default_factory=tuple)


def grade(task: Task, action: Action, catalog: Catalog, *, gold_rank: int | None, k: int) -> Grade:
    """Score one agent action against a task.

    `gold_rank` is the gold tool's 1-based rank in the retriever's full ordering,
    or None if it was not ranked at all. It only classifies a failure; it never
    turns a wrong action into a right one.
    """
    if action is None:
        return Grade(_miss_kind(gold_rank, k), False, "agent produced no action")

    if isinstance(action, Clarify):
        # Tool choice is graded before the question it asked, so a clarification
        # about the wrong tool is reported as a wrong tool.
        if action.tool not in task.acceptable:
            return Grade(_wrong_tool_kind(task, action.tool, gold_rank, k), False, action.tool)
        if task.kind != "underspecified":
            return Grade("missing_argument", False, f"clarified on a fully specified task: {action.missing}")
        if action.missing != task.missing_param:
            return Grade("missing_argument", False, f"asked for {action.missing}, needed {task.missing_param}")
        return Grade("success", True, "correctly refused to invent a value")

    if action.tool not in task.acceptable:
        return Grade(_wrong_tool_kind(task, action.tool, gold_rank, k), False, action.tool)

    tool = catalog.get(action.tool)
    slots = task.slot_map
    query_lower = task.query.lower()

    for param in tool.required_params:
        if param.name not in action.args or action.args[param.name] == "":
            if task.kind == "underspecified" and param.name == task.missing_param:
                return Grade("missing_argument", False, f"omitted {param.name} instead of asking")
            return Grade("missing_argument", False, param.name)

    hallucinated: list[str] = []
    for name, value in action.args.items():
        expected = slots.get(name)
        if expected is not None:
            if str(value).strip().strip("'\"") != expected:
                return Grade("wrong_argument_value", False, f"{name}={value!r} expected {expected!r}")
            continue
        if not str(value).strip():
            continue
        if str(value).strip().strip("'\"").lower() not in query_lower:
            param = next((p for p in tool.params if p.name == name), None)
            if param is not None and param.required:
                hallucinated.append(name)

    if hallucinated:
        return Grade(
            "hallucinated_argument",
            False,
            ", ".join(f"{h}={action.args[h]!r}" for h in hallucinated),
            tuple(hallucinated),
        )

    if task.kind == "underspecified":
        return Grade("hallucinated_argument", False, f"filled {task.missing_param} unasked")

    return Grade("success", True)


def _miss_kind(gold_rank: int | None, k: int) -> Outcome:
    if gold_rank is None:
        return "deep_retrieval_miss"
    if gold_rank > k:
        return "below_cutoff" if gold_rank <= 3 * k else "deep_retrieval_miss"
    return "no_action"


def _wrong_tool_kind(task: Task, chosen: str, gold_rank: int | None, k: int) -> Outcome:
    if gold_rank is None or gold_rank > k:
        return _miss_kind(gold_rank, k)
    if chosen in task.confusable_with:
        return "near_duplicate_confusion"
    return "wrong_tool_selected"
