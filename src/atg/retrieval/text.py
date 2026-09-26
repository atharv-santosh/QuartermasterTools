"""Tokenization and vocabulary normalization shared by every lexical retriever."""

from __future__ import annotations

import re
from functools import lru_cache

_SPLIT = re.compile(r"[^a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or that the their to
    with without will would can could should please i me my we our you your this these those
    all any some more most not no do does did done than then there here when what which who
    how why if so such also only just very much many new old get set""".split()
)

# Deliberately small and hand written. A generated catalog whose queries reuse
# the catalog's own wording would make lexical retrieval look better than it is,
# so queries are written with synonyms and the retriever has to bridge them.
_SYNONYMS: dict[str, str] = {
    "mail": "email",
    "emails": "email",
    "messages": "message",
    "msg": "message",
    "send": "send",
    "sending": "send",
    "sent": "send",
    "delete": "delete",
    "deleting": "delete",
    "deleted": "delete",
    "remove": "delete",
    "removing": "delete",
    "drop": "delete",
    "purge": "delete",
    "erase": "delete",
    "create": "create",
    "creating": "create",
    "created": "create",
    "add": "create",
    "adding": "create",
    "register": "create",
    "open": "create",
    "make": "create",
    "update": "update",
    "updating": "update",
    "change": "update",
    "changing": "update",
    "edit": "update",
    "editing": "update",
    "modify": "update",
    "patch": "update",
    "fetch": "get",
    "fetching": "get",
    "retrieve": "get",
    "lookup": "get",
    "read": "get",
    "reading": "get",
    "show": "list",
    "listing": "list",
    "enumerate": "list",
    "find": "search",
    "finding": "search",
    "searching": "search",
    "query": "search",
    "querying": "search",
    "export": "export",
    "exporting": "export",
    "dump": "export",
    "download": "export",
    "archive": "archive",
    "archiving": "archive",
    "retire": "archive",
    "deactivate": "archive",
    "refund": "refund",
    "refunding": "refund",
    "reimburse": "refund",
    "post": "post",
    "posting": "post",
    "notify": "notify",
    "notifying": "notify",
    "notification": "notify",
    "alert": "notify",
    "deploy": "deployment",
    "deploying": "deployment",
    "rollout": "deployment",
    "ship": "deployment",
    "customers": "customer",
    "contacts": "contact",
    "users": "user",
    "tickets": "ticket",
    "events": "event",
    "files": "file",
    "objects": "object",
    "charges": "charge",
    "invoices": "invoice",
    "devices": "device",
    "channels": "channel",
    "pages": "page",
    "tables": "table",
    "records": "record",
}


def _stem(token: str) -> str:
    if token in _SYNONYMS:
        return _SYNONYMS[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and not token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    return token


@lru_cache(maxsize=32768)
def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase, split on non-alphanumerics, drop stopwords, light stemming."""
    out: list[str] = []
    for raw in _SPLIT.split(text.lower()):
        if not raw or raw in STOPWORDS:
            continue
        stemmed = _stem(raw)
        if stemmed and stemmed not in STOPWORDS:
            out.append(stemmed)
    return tuple(out)


def char_ngrams(text: str, n: int = 4) -> list[str]:
    """Character n-grams over a space-padded normalized string."""
    flat = " " + " ".join(tokenize(text)) + " "
    if len(flat) <= n:
        return [flat]
    return [flat[i : i + n] for i in range(len(flat) - n + 1)]
