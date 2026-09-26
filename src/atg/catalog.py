"""Deterministic synthetic tool catalog: ~1200 tools with confusable near-duplicates."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

SlotKind = Literal["email", "id", "text", "int", "enum", "bool", "path", "date", "money", "url"]

# Naming conventions differ per server on purpose: a real gateway multiplexes
# servers written by different teams, and the inconsistency is part of the
# retrieval difficulty rather than an accident.
NamingStyle = Literal["verb_noun", "noun_verb", "dotted", "camel_suffix"]


@dataclass(frozen=True, slots=True)
class Param:
    name: str
    kind: SlotKind
    required: bool
    description: str
    enum: tuple[str, ...] = ()

    def json_schema(self) -> dict[str, Any]:
        base: dict[str, Any]
        if self.kind == "int":
            base = {"type": "integer"}
        elif self.kind == "bool":
            base = {"type": "boolean"}
        elif self.kind == "money":
            base = {"type": "number"}
        elif self.kind == "enum":
            base = {"type": "string", "enum": list(self.enum)}
        else:
            base = {"type": "string"}
            fmt = {"email": "email", "date": "date-time", "url": "uri"}.get(self.kind)
            if fmt:
                base["format"] = fmt
        base["description"] = self.description
        return base


@dataclass(frozen=True, slots=True)
class Precondition:
    """A narrow condition under which a tool is the correct choice.

    `phrase` is how the schema states it. `query_phrase` is how a person would
    say the same thing, and they are deliberately different words: a query that
    quoted the schema back would hand lexical retrieval the answer.
    """

    marker: str
    phrase: str
    query_phrase: str
    domains: frozenset[str]
    verbs: frozenset[str]

    def describe(self) -> str:
        return f"Only valid when {self.phrase}."

    def applies_to(self, domain: str, verb: str) -> bool:
        return domain in self.domains and verb in self.verbs


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    server: str
    domain: str
    resource: str
    verb: str
    summary: str
    description: str
    params: tuple[Param, ...]
    family: str
    tags: tuple[str, ...] = ()
    precondition: Precondition | None = None
    scopes: tuple[str, ...] = ()
    alias_of: str | None = None

    @property
    def required_params(self) -> tuple[Param, ...]:
        return tuple(p for p in self.params if p.required)

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {p.name: p.json_schema() for p in self.params},
            "required": [p.name for p in self.params if p.required],
            "additionalProperties": False,
        }

    def full_description(self) -> str:
        parts = [self.description]
        if self.precondition:
            parts.append(self.precondition.describe())
        return " ".join(parts)

    def as_mcp_tool(self) -> dict[str, Any]:
        """The shape an MCP `tools/list` response carries."""
        return {
            "name": self.name,
            "description": self.full_description(),
            "inputSchema": self.json_schema(),
        }

    def retrieval_text(self) -> str:
        """Everything a retriever is allowed to index."""
        words = [
            self.name.replace("_", " ").replace(".", " "),
            self.summary,
            self.full_description(),
            self.domain,
            self.resource.replace("_", " "),
            " ".join(self.tags),
            " ".join(p.name.replace("_", " ") for p in self.params),
        ]
        return " ".join(words)


@dataclass(slots=True)
class Catalog:
    tools: tuple[Tool, ...]
    seed: int
    by_name: dict[str, Tool] = field(default_factory=dict, repr=False)
    families: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.by_name = {t.name: t for t in self.tools}
        fam: dict[str, list[str]] = {}
        for t in self.tools:
            fam.setdefault(t.family, []).append(t.name)
        self.families = {k: tuple(v) for k, v in fam.items()}

    def __len__(self) -> int:
        return len(self.tools)

    def get(self, name: str) -> Tool:
        return self.by_name[name]

    def servers(self) -> tuple[str, ...]:
        return tuple(sorted({t.server for t in self.tools}))

    def fingerprint(self) -> str:
        """Stable hash over every field that affects retrieval, for reproducibility checks."""
        h = hashlib.sha256()
        for t in self.tools:
            h.update(t.name.encode())
            h.update(t.full_description().encode())
            h.update(json.dumps(t.json_schema(), sort_keys=True).encode())
        return h.hexdigest()[:16]

    def subset(self, n: int, *, keep: Iterable[str] = ()) -> "Catalog":
        """First `n` tools in catalog order, with `keep` forced in.

        Catalog order is itself shuffled by the generator, so a prefix is an
        unbiased sample across domains while staying nested: the size-50
        catalog is a subset of the size-100 one, which is what makes the
        accuracy-versus-size curves comparable point to point.
        """
        keep_set = set(keep)
        head = list(self.tools[:n])
        present = {t.name for t in head}
        missing = [self.by_name[k] for k in keep_set if k not in present and k in self.by_name]
        if missing:
            # Evict from the tail to hold the size fixed, never evicting a kept tool.
            room = len(missing)
            trimmed: list[Tool] = []
            for t in reversed(head):
                if room > 0 and t.name not in keep_set:
                    room -= 1
                    continue
                trimmed.append(t)
            head = list(reversed(trimmed)) + missing
        return Catalog(tools=tuple(head), seed=self.seed)


# --------------------------------------------------------------------------
# Generation vocabulary
# --------------------------------------------------------------------------

DOMAINS: dict[str, dict[str, Any]] = {
    "email": {
        "server": "mail-svc",
        "style": "verb_noun",
        "resources": ["message", "draft", "thread", "label", "signature", "alias", "filter", "forwarding_rule", "vacation_responder"],
        "tags": ["mail", "smtp", "inbox"],
    },
    "calendar": {
        "server": "cal-svc",
        "style": "verb_noun",
        "resources": ["event", "invite", "availability", "calendar", "reminder", "room_booking", "recurring_series", "working_hours"],
        "tags": ["schedule", "meeting", "time"],
    },
    "crm": {
        "server": "crm-core",
        "style": "noun_verb",
        "resources": ["contact", "company", "deal", "pipeline", "activity", "lead", "note", "quote", "territory"],
        "tags": ["sales", "customer", "account"],
    },
    "payments": {
        "server": "billing",
        "style": "verb_noun",
        "resources": ["charge", "refund", "subscription", "invoice", "payout", "dispute", "coupon", "tax_rate", "payment_method"],
        "tags": ["money", "billing", "transaction"],
    },
    "storage": {
        "server": "objstore",
        "style": "dotted",
        "resources": ["object", "bucket", "prefix", "lifecycle_rule", "signed_url", "replication", "cors_policy", "storage_class"],
        "tags": ["file", "blob", "s3"],
    },
    "analytics": {
        "server": "analytics",
        "style": "verb_noun",
        "resources": ["event_stream", "funnel", "cohort", "dashboard", "metric", "segment", "retention_curve", "attribution_model"],
        "tags": ["report", "telemetry", "kpi"],
    },
    "devops": {
        "server": "platform",
        "style": "dotted",
        "resources": ["deployment", "service", "pod", "alert_rule", "runbook", "incident", "rollout", "config_map", "autoscaler"],
        "tags": ["infra", "kubernetes", "oncall"],
    },
    "hr": {
        "server": "people",
        "style": "noun_verb",
        "resources": ["employee", "timeoff_request", "review_cycle", "offer", "payroll_run", "org_unit", "job_requisition"],
        "tags": ["people", "staff", "hiring"],
    },
    "support": {
        "server": "helpdesk",
        "style": "verb_noun",
        "resources": ["ticket", "macro", "sla_policy", "satisfaction_survey", "queue", "knowledge_article", "escalation_path"],
        "tags": ["helpdesk", "customer", "case"],
    },
    "messaging": {
        "server": "chatops",
        "style": "verb_noun",
        "resources": ["channel_message", "direct_message", "channel", "reaction", "webhook_post", "user_group", "pinned_item"],
        "tags": ["chat", "slack", "notify"],
    },
    "docs": {
        "server": "wiki",
        "style": "noun_verb",
        "resources": ["page", "space", "comment", "attachment", "template", "revision", "label_taxonomy", "export_job"],
        "tags": ["wiki", "document", "knowledge"],
    },
    "database": {
        "server": "dbadmin",
        "style": "dotted",
        "resources": ["table", "index", "query_job", "snapshot", "replica", "migration", "connection_pool", "grant"],
        "tags": ["sql", "schema", "postgres"],
    },
    "search": {
        "server": "searchd",
        "style": "verb_noun",
        "resources": ["index_document", "synonym_set", "query_suggestion", "ranking_profile", "stopword_list", "analyzer_chain", "reindex_job"],
        "tags": ["retrieval", "lucene", "relevance"],
    },
    "ml": {
        "server": "mlplat",
        "style": "dotted",
        "resources": ["training_job", "model_version", "endpoint", "feature_view", "evaluation", "dataset_split", "hyperparam_sweep"],
        "tags": ["model", "training", "inference"],
    },
    "iot": {
        "server": "fleet",
        "style": "noun_verb",
        "resources": ["device", "firmware_bundle", "telemetry_stream", "device_group", "ota_job", "provisioning_claim", "geofence"],
        "tags": ["device", "sensor", "edge"],
    },
    "finance": {
        "server": "ledger",
        "style": "verb_noun",
        "resources": ["journal_entry", "account", "budget", "expense_report", "tax_document", "cost_center", "reconciliation"],
        "tags": ["accounting", "ledger", "gl"],
    },
    "inventory": {
        "server": "wms",
        "style": "noun_verb",
        "resources": ["sku", "stock_level", "shipment", "purchase_order", "warehouse_bin", "cycle_count", "carrier_rate"],
        "tags": ["warehouse", "stock", "logistics"],
    },
    "marketing": {
        "server": "growth",
        "style": "verb_noun",
        "resources": ["campaign", "audience_list", "ad_group", "landing_page", "utm_link", "creative_asset", "drip_sequence"],
        "tags": ["campaign", "growth", "ads"],
    },
    "security": {
        "server": "secops",
        "style": "dotted",
        "resources": ["api_key", "role_binding", "audit_log", "secret", "firewall_rule", "finding", "mfa_enrollment", "ip_allowlist"],
        "tags": ["iam", "audit", "access"],
    },
    "notifications": {
        "server": "notify",
        "style": "verb_noun",
        "resources": ["push_notification", "sms", "digest", "subscription_pref", "template_render", "quiet_hours", "delivery_receipt"],
        "tags": ["push", "sms", "alerting"],
    },
}

VERBS: dict[str, dict[str, Any]] = {
    "create": {"phrases": ["create", "add", "register", "open", "set up"], "mutating": True},
    "get": {"phrases": ["get", "fetch", "look up", "read", "show me"], "mutating": False},
    "list": {"phrases": ["list", "enumerate", "show all", "page through"], "mutating": False},
    "update": {"phrases": ["update", "change", "edit", "modify", "patch"], "mutating": True},
    "delete": {"phrases": ["delete", "remove", "drop", "purge"], "mutating": True},
    "search": {"phrases": ["search", "find", "query", "look for"], "mutating": False},
    "export": {"phrases": ["export", "dump", "download as csv"], "mutating": False},
    "archive": {"phrases": ["archive", "retire", "deactivate"], "mutating": True},
}

# Slot value pools. Values are drawn deterministically and appear verbatim in
# generated queries, so an argument the agent produces is either copied from
# the query or invented.
_EMAILS = [
    "alice@northwind.example",
    "bob.chen@northwind.example",
    "ops-team@northwind.example",
    "priya@vendor.example",
    "finance@northwind.example",
    "d.okafor@partner.example",
]
_TEXTS = [
    "Q3 revenue recap",
    "migration cutover window",
    "vendor contract renewal",
    "incident 4471 postmortem",
    "onboarding checklist",
    "pricing experiment results",
]
_PATHS = [
    "reports/2026/q3-close.csv",
    "exports/customers-eu.parquet",
    "backups/ledger-0918.tar.gz",
    "assets/brand/logo-dark.svg",
]
_ENUMS: dict[str, tuple[str, ...]] = {
    "status": ("active", "paused", "archived"),
    "priority": ("low", "normal", "urgent"),
    "format": ("csv", "json", "parquet"),
    "visibility": ("private", "team", "public"),
    "region": ("us-east-1", "eu-west-1", "ap-south-1"),
}

# Generic extras any resource plausibly carries, plus a few that only make
# sense in some domains. A money field on an org unit reads as a generator bug
# even in a synthetic catalog.
_PARAM_LIBRARY: list[tuple[str, SlotKind]] = [
    ("name", "text"),
    ("owner_email", "email"),
    ("limit", "int"),
    ("status", "enum"),
    ("priority", "enum"),
    ("region", "enum"),
    ("notes", "text"),
    ("dry_run", "bool"),
    ("effective_at", "date"),
    ("visibility", "enum"),
]

_DOMAIN_PARAMS: dict[str, list[tuple[str, SlotKind]]] = {
    "payments": [("amount", "money"), ("currency", "text")],
    "finance": [("amount", "money"), ("cost_center", "text")],
    "inventory": [("quantity", "int")],
    "storage": [("path", "path"), ("format", "enum")],
    "analytics": [("format", "enum"), ("callback_url", "url")],
    "database": [("format", "enum")],
    "devops": [("callback_url", "url")],
    "ml": [("callback_url", "url")],
    "notifications": [("callback_url", "url")],
}

PRECONDITIONS: tuple[Precondition, ...] = (
    Precondition(
        "partial",
        "the caller is settling only part of the original amount",
        "only part of it, not the whole thing",
        frozenset({"payments", "finance", "inventory"}),
        frozenset({"create", "update"}),
    ),
    Precondition(
        "bulk",
        "sending to more than 50 recipients in one call",
        "this needs to go to a few hundred people at once",
        frozenset({"email", "messaging", "notifications", "marketing"}),
        frozenset({"create", "export"}),
    ),
    Precondition(
        "legal_hold",
        "the record is under legal hold and cannot be hard deleted",
        "legal has a hold on it so the row has to stay",
        frozenset({"crm", "hr", "support", "docs", "finance", "security"}),
        frozenset({"delete", "archive", "update"}),
    ),
    Precondition(
        "sandbox",
        "operating against the sandbox environment, not production",
        "on the test environment, do not touch prod",
        frozenset({"devops", "ml", "database", "storage", "search"}),
        frozenset({"create", "update", "delete"}),
    ),
    Precondition(
        "verified",
        "the sending domain has completed DKIM verification",
        "it has to go out from our own domain, which is already set up for it",
        frozenset({"email", "notifications", "marketing"}),
        frozenset({"create", "update"}),
    ),
)


def _style_name(style: NamingStyle, verb: str, resource: str) -> str:
    if style == "verb_noun":
        return f"{verb}_{resource}"
    if style == "noun_verb":
        return f"{resource}_{verb}"
    if style == "dotted":
        return f"{resource}.{verb}"
    return f"{resource}_{verb}V2"


def _params_for(rng: random.Random, verb: str, resource: str, domain: str) -> tuple[Param, ...]:
    params: list[Param] = []
    if verb in ("get", "update", "delete", "archive"):
        params.append(Param(f"{resource}_id", "id", True, f"Identifier of the {resource.replace('_', ' ')}."))
    if verb == "create":
        params.append(Param("name", "text", True, f"Human readable name for the new {resource.replace('_', ' ')}."))
    if verb in ("list", "search", "export"):
        params.append(Param("limit", "int", False, "Maximum number of results to return."))
    if verb == "search":
        params.append(Param("query", "text", True, "Free text search expression."))
    if verb == "export":
        params.append(Param("format", "enum", True, "Serialization format.", _ENUMS["format"]))
    if verb == "update":
        params.append(Param("fields", "text", True, "Comma separated field=value pairs to apply."))

    pool = _PARAM_LIBRARY + _DOMAIN_PARAMS.get(domain, [])
    extra = rng.sample(pool, rng.randint(1, min(4, len(pool))))
    taken = {p.name for p in params}
    for name, kind in extra:
        if name in taken:
            continue
        taken.add(name)
        enum = _ENUMS.get(name, ())
        if kind == "enum" and not enum:
            continue
        params.append(
            Param(
                name,
                kind,
                rng.random() < 0.25,
                f"{name.replace('_', ' ').capitalize()} for the {resource.replace('_', ' ')}.",
                enum,
            )
        )
    return tuple(params)


def _summary(verb: str, resource: str, domain: str) -> str:
    human = resource.replace("_", " ")
    templates = {
        "create": f"Create a new {human} in {domain}.",
        "get": f"Retrieve a single {human} by id.",
        "list": f"List {human}s with optional paging.",
        "update": f"Update fields on an existing {human}.",
        "delete": f"Permanently delete a {human}.",
        "search": f"Search {human}s by free text.",
        "export": f"Export {human}s to a file.",
        "archive": f"Archive a {human} without deleting it.",
    }
    return templates[verb]


def _description(rng: random.Random, verb: str, resource: str, domain: str, tags: list[str]) -> str:
    human = resource.replace("_", " ")
    clauses = {
        "create": f"Creates a {human} record in the {domain} service and returns its assigned id.",
        "get": f"Returns the full {human} record for the supplied id, including nested relations.",
        "list": f"Returns a page of {human} records ordered by creation time, newest first.",
        "update": f"Applies a partial update to a {human} record and returns the new state.",
        "delete": f"Hard deletes a {human} record. This cannot be undone and cascades to children.",
        "search": f"Runs a relevance ranked search over {human} records in the {domain} index.",
        "export": f"Streams matching {human} records to a file in the requested format.",
        "archive": f"Marks a {human} archived so it stops appearing in default listings.",
    }
    tail = rng.choice(
        [
            f"Related {tags[0]} records are not modified.",
            "Callers need the corresponding write scope.",
            "Rate limited to 100 calls per minute per token.",
            "Results are eventually consistent for up to five seconds.",
        ]
    )
    return f"{clauses[verb]} {tail}"


# Hand written confusable clusters. These are the cases the thesis is about:
# names and descriptions so close that a retriever ranks them adjacently and a
# model has to disambiguate on a detail buried in the description.
CONFUSABLE_CLUSTERS: tuple[dict[str, Any], ...] = (
    {
        "family": "email_send",
        "domain": "email",
        "members": [
            {
                "name": "send_email",
                "summary": "Send a single email message immediately.",
                "params": ["to", "subject", "body"],
                "intent": "Email {to} right now with subject line {subject} and body {body}",
            },
            {
                "name": "send_bulk_email",
                "summary": "Send one email to many recipients at once.",
                "pre": "bulk",
                "params": ["recipients", "subject", "body"],
                "intent": "Blast the whole announce list, {recipients}, with subject line {subject} and body {body}",
            },
            {
                "name": "email_draft_create",
                "summary": "Create an unsent draft email.",
                "params": ["to", "subject", "body"],
                "intent": "Stage an email to {to} for me to review before it goes out, subject line {subject}, body {body}",
            },
            {
                "name": "send_email_template",
                "summary": "Send an email rendered from a saved template.",
                "params": ["to", "template_id"],
                "intent": "Mail {to} using the saved template {template_id}",
            },
            {
                "name": "notify_via_email",
                "summary": "Send a system notification email from the platform address.",
                "params": ["to", "event"],
                "intent": "Have the platform email {to} an automated notice about {event}",
            },
            {
                "name": "send_email_verified_domain",
                "summary": "Send email from a DKIM verified custom domain.",
                "pre": "verified",
                "params": ["to", "subject", "body"],
                "intent": "Email {to} from our own verified sending domain, subject line {subject}, body {body}",
            },
        ],
    },
    {
        "family": "payment_refund",
        "domain": "payments",
        "members": [
            {
                "name": "refund_charge_full",
                "summary": "Refund the entire original amount of a charge.",
                "params": ["charge_id"],
                "intent": "Give the customer all their money back on charge {charge_id}",
            },
            {
                "name": "refund_charge_partial",
                "summary": "Refund part of a charge's amount.",
                "pre": "partial",
                "params": ["charge_id", "amount"],
                "intent": "Send back {amount} of charge {charge_id} and leave the rest",
            },
            {
                "name": "void_charge",
                "summary": "Void an authorization that has not captured yet.",
                "params": ["charge_id"],
                "intent": "Cancel the uncaptured authorization on charge {charge_id} before it settles",
            },
            {
                "name": "create_credit_note",
                "summary": "Issue a credit note against an invoice.",
                "params": ["invoice_id", "amount"],
                "intent": "Put {amount} of credit against invoice {invoice_id} for them to use later",
            },
            {
                "name": "reverse_payout",
                "summary": "Reverse a payout that already left the platform balance.",
                "params": ["payout_id"],
                "intent": "Claw back payout {payout_id}, it already left our balance",
            },
        ],
    },
    {
        "family": "record_delete",
        "domain": "crm",
        "members": [
            {
                "name": "contact_delete",
                "summary": "Hard delete a contact and all of its activity.",
                "params": ["contact_id"],
                "intent": "Erase contact {contact_id} and everything attached to it, permanently",
            },
            {
                "name": "contact_archive",
                "summary": "Archive a contact, keeping history intact.",
                "params": ["contact_id"],
                "intent": "Take contact {contact_id} out of the active list but keep the history",
            },
            {
                "name": "contact_anonymize",
                "summary": "Strip personal fields from a contact under legal hold.",
                "pre": "legal_hold",
                "params": ["contact_id"],
                "intent": "Strip the personal details off contact {contact_id}, legal says we cannot actually delete the row",
            },
            {
                "name": "contact_merge",
                "summary": "Merge a duplicate contact into a surviving record.",
                "params": ["contact_id", "into_id"],
                "intent": "Fold duplicate contact {contact_id} into {into_id}",
            },
        ],
    },
    {
        "family": "deploy",
        "domain": "devops",
        "members": [
            {
                "name": "deployment.create",
                "summary": "Roll out a new deployment revision to production.",
                "params": ["service", "image"],
                "intent": "Ship image {image} of {service} to production",
            },
            {
                "name": "deployment.create_sandbox",
                "summary": "Roll out a revision to the sandbox cluster only.",
                "pre": "sandbox",
                "params": ["service", "image"],
                "intent": "Put image {image} of {service} on the sandbox cluster, not production",
            },
            {
                "name": "deployment.rollback",
                "summary": "Roll a service back to its previous revision.",
                "params": ["service"],
                "intent": "Take {service} back to the revision it was on before",
            },
            {
                "name": "deployment.restart",
                "summary": "Restart all pods of a service without changing the image.",
                "params": ["service"],
                "intent": "Bounce every pod of {service} without changing what is deployed",
            },
            {
                "name": "rollout.pause",
                "summary": "Pause an in-progress progressive rollout.",
                "params": ["service"],
                "intent": "Freeze the rollout of {service} where it is, it looks wrong",
            },
        ],
    },
    {
        "family": "chat_post",
        "domain": "messaging",
        "members": [
            {
                "name": "post_channel_message",
                "summary": "Post a message to a channel.",
                "params": ["channel", "text"],
                "intent": "Drop a note in {channel} saying {text}",
            },
            {
                "name": "post_direct_message",
                "summary": "Send a direct message to one user.",
                "params": ["user", "text"],
                "intent": "DM {user} privately saying {text}",
            },
            {
                "name": "post_thread_reply",
                "summary": "Reply inside an existing message thread.",
                "params": ["thread_ts", "text"],
                "intent": "Reply under the existing thread {thread_ts} saying {text}",
            },
            {
                "name": "post_webhook_message",
                "summary": "Post via an incoming webhook URL.",
                "params": ["webhook_url", "text"],
                "intent": "Push {text} through the incoming webhook at {webhook_url}",
            },
            {
                "name": "schedule_channel_message",
                "summary": "Schedule a channel message for later delivery.",
                "params": ["channel", "text", "post_at"],
                "intent": "Queue up a message in {channel} saying {text} to go out at {post_at}",
            },
        ],
    },
    {
        "family": "file_share",
        "domain": "storage",
        "members": [
            {
                "name": "object.signed_url_read",
                "summary": "Mint a time limited read URL for an object.",
                "params": ["bucket", "key"],
                "intent": "Give me an expiring link so someone can download {key} from {bucket}",
            },
            {
                "name": "object.signed_url_write",
                "summary": "Mint a time limited upload URL for an object.",
                "params": ["bucket", "key"],
                "intent": "Give me an expiring link so a vendor can upload to {key} in {bucket}",
            },
            {
                "name": "object.make_public",
                "summary": "Make an object world readable.",
                "params": ["bucket", "key"],
                "intent": "Open up {key} in {bucket} so anyone on the internet can read it",
            },
            {
                "name": "object.copy",
                "summary": "Copy an object to another bucket or prefix.",
                "params": ["bucket", "key", "dest"],
                "intent": "Duplicate {key} from {bucket} over to {dest}",
            },
        ],
    },
)


_MEMBER_PARAM_KINDS: dict[str, SlotKind] = {
    "to": "email",
    "recipients": "email",
    "user": "email",
    "subject": "text",
    "body": "text",
    "text": "text",
    "event": "text",
    "template_id": "id",
    "charge_id": "id",
    "invoice_id": "id",
    "payout_id": "id",
    "contact_id": "id",
    "into_id": "id",
    "amount": "money",
    "service": "text",
    "image": "text",
    "channel": "text",
    "thread_ts": "id",
    "webhook_url": "url",
    "post_at": "date",
    "bucket": "text",
    "key": "path",
    "dest": "path",
}


CLUSTER_INTENTS: dict[str, str] = {}
for _cluster in CONFUSABLE_CLUSTERS:
    for _member in _cluster["members"]:
        CLUSTER_INTENTS[_member["name"]] = _member["intent"]


def _cluster_tools() -> list[Tool]:
    tools: list[Tool] = []
    for cluster in CONFUSABLE_CLUSTERS:
        domain = cluster["domain"]
        meta = DOMAINS[domain]
        for member in cluster["members"]:
            pre = next((p for p in PRECONDITIONS if p.marker == member.get("pre")), None)
            params = tuple(
                Param(
                    pn,
                    _MEMBER_PARAM_KINDS.get(pn, "text"),
                    True,
                    f"{pn.replace('_', ' ').capitalize()} argument.",
                )
                for pn in member["params"]
            )
            tools.append(
                Tool(
                    name=member["name"],
                    server=meta["server"],
                    domain=domain,
                    resource=cluster["family"],
                    verb="create",
                    summary=member["summary"],
                    description=(
                        member["summary"]
                        + " One of several closely related operations in this service; the"
                        " distinguishing constraint is stated here and nowhere else."
                    ),
                    params=params,
                    family=cluster["family"],
                    tags=tuple(meta["tags"]),
                    precondition=pre,
                    scopes=(f"{domain}:write",),
                )
            )
    return tools


_ALIAS_PREFIXES = ("legacy_", "erp_")
_ALIAS_SERVER = "legacy-erp"
N_ALIASES = 36


def _alias_tools(rng: random.Random, source: list[Tool]) -> list[Tool]:
    """Re-expose some operations under a second, differently worded name.

    A gateway in front of a real estate of servers nearly always has two ways to
    do the same thing, because one server is the migration target and the other
    has not been decommissioned. This is where recall@k and task success come
    apart: recall is scored against the canonical name, while the task is
    genuinely completed by either.
    """
    aliases: list[Tool] = []
    pool = [t for t in source if t.verb in ("create", "update", "get", "list", "delete")]
    for tool in rng.sample(pool, min(N_ALIASES, len(pool))):
        prefix = rng.choice(_ALIAS_PREFIXES)
        name = f"{prefix}{tool.resource}_{tool.verb}"
        human = tool.resource.replace("_", " ")
        aliases.append(
            Tool(
                name=name,
                server=_ALIAS_SERVER,
                domain=tool.domain,
                resource=tool.resource,
                verb=tool.verb,
                summary=f"{tool.verb.capitalize()} a {human} record via the legacy ERP bridge.",
                description=(
                    f"Legacy bridge for the same {human} operation as the {tool.domain} service exposes. "
                    "Kept for clients that have not migrated; semantics and arguments are identical."
                ),
                params=tool.params,
                family=tool.family,
                tags=(*tool.tags, "legacy"),
                precondition=tool.precondition,
                scopes=tool.scopes,
                alias_of=tool.name,
            )
        )
    return aliases


def build_catalog(seed: int = 20260925, target: int = 1200) -> Catalog:
    """Generate a reproducible catalog of `target` tools."""
    rng = random.Random(seed)
    tools: list[Tool] = list(_cluster_tools())
    taken = {t.name for t in tools}

    combos: list[tuple[str, str, str]] = []
    for domain, meta in DOMAINS.items():
        for resource in meta["resources"]:
            for verb in VERBS:
                combos.append((domain, resource, verb))
    rng.shuffle(combos)

    for domain, resource, verb in combos:
        if len(tools) >= target:
            break
        meta = DOMAINS[domain]
        name = _style_name(meta["style"], verb, resource)
        if name in taken:
            continue
        taken.add(name)
        eligible = [p for p in PRECONDITIONS if p.applies_to(domain, verb)]
        pre = rng.choice(eligible) if eligible and rng.random() < 0.14 else None
        tools.append(
            Tool(
                name=name,
                server=meta["server"],
                domain=domain,
                resource=resource,
                verb=verb,
                summary=_summary(verb, resource, domain),
                description=_description(rng, verb, resource, domain, list(meta["tags"])),
                params=_params_for(rng, verb, resource, domain),
                family=f"{domain}.{resource}",
                tags=tuple(meta["tags"]),
                precondition=pre,
                scopes=(f"{domain}:{'write' if VERBS[verb]['mutating'] else 'read'}",),
            )
        )

    # Shuffle so that a size-N prefix is a cross-domain sample, then force the
    # hand written clusters and their aliases to the front: every catalog size
    # must contain the confusable families or the small-catalog measurements
    # would be easy for a reason that has nothing to do with size.
    cluster_names = {t.name for t in _cluster_tools()}
    clustered = [t for t in tools if t.name in cluster_names]
    rest = [t for t in tools if t.name not in cluster_names]
    rng.shuffle(rest)
    ordered = clustered + rest
    aliases = _alias_tools(rng, [t for t in rest[:300]])
    by_source: dict[str, list[Tool]] = {}
    for alias in aliases:
        by_source.setdefault(alias.alias_of or "", []).append(alias)
    woven: list[Tool] = []
    for tool in ordered:
        woven.append(tool)
        woven.extend(by_source.get(tool.name, ()))
    return Catalog(tools=tuple(woven), seed=seed)


# Value pools keyed by parameter name, so a "service" argument reads like a
# service and a "subject" reads like a subject line. A benchmark whose gold
# arguments are nonsense cannot tell a correct extraction from a lucky one.
_NAMED_POOLS: dict[str, tuple[str, ...]] = {
    "service": ("checkout-api", "ledger-worker", "notify-relay", "search-indexer"),
    "image": ("ghcr.io/northwind/checkout:1.42.0", "ghcr.io/northwind/ledger:0.9.3"),
    "channel": ("#ops-alerts", "#release-train", "#finance-close"),
    "bucket": ("northwind-reports", "northwind-archive", "northwind-exports"),
    "subject": ("Q3 revenue recap", "Cutover window confirmed", "Renewal terms attached"),
    "body": (
        "numbers are final, see the attached sheet",
        "cutover starts at 02:00 UTC Saturday",
        "please countersign by Friday",
    ),
    "text": (
        "deploy is frozen until the incident clears",
        "close is done, books are locked",
        "on call handover in ten minutes",
    ),
    "event": ("invoice.payment_failed", "deployment.rolled_back", "quota.exceeded"),
    "name": ("Q3 pallet restock", "EU customer export", "Renewal outreach wave 2"),
    "fields": ("status=paused", "priority=urgent", "owner_email=ops-team@northwind.example"),
    "query": ("overdue and unassigned", "region:eu and status:active", "created after 2026-09-01"),
    "notes": ("raised by the vendor on the Thursday call", "follow up after month end"),
    "dest": ("archive/2026/q3/", "cold/ledger/", "handoff/vendor/"),
}


def slot_value(rng: random.Random, param: Param) -> str:
    """A concrete, quotable value for a parameter."""
    pool = _NAMED_POOLS.get(param.name)
    if pool:
        return rng.choice(pool)
    if param.kind == "email":
        return rng.choice(_EMAILS)
    if param.kind == "id":
        stem = param.name[:-3] if param.name.endswith("_id") else param.name
        token = "".join(w[0] for w in stem.split("_") if w)[:4] or "rec"
        return f"{token}_{rng.randrange(10**6, 10**7)}"
    if param.kind == "int":
        return str(rng.choice([10, 25, 50, 100, 250]))
    if param.kind == "bool":
        return rng.choice(["true", "false"])
    if param.kind == "enum":
        return rng.choice(param.enum or _ENUMS["status"])
    if param.kind == "path":
        return rng.choice(_PATHS)
    if param.kind == "date":
        return rng.choice(["2026-10-01T09:00:00Z", "2026-11-15T17:30:00Z", "2026-09-30T23:59:00Z"])
    if param.kind == "money":
        return rng.choice(["48.50", "1200.00", "317.25"])
    if param.kind == "url":
        return rng.choice(["https://hooks.northwind.example/T01/B02", "https://cb.northwind.example/ingest"])
    return rng.choice(_TEXTS)
