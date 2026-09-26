"""A deterministic stand-in for the model in the agent loop.

This is not a language model and does not pretend to be one. It is a scoring
function with two properties that matter for the measurement:

  * It reads only the rendered candidate text, so information a compressed
    rendering dropped is information it does not have.
  * Its choice between candidates is noisy, with the noise fixed per
    (query, tool) pair. Accuracy therefore falls as candidates are added
    without any size-dependent term anywhere in the code: more distractors is
    simply more chances that one of them out-draws the right answer.

The one deliberate borrowing from real model behaviour is `MIDDLE_PENALTY`,
which discounts candidates buried in the centre of a long list. That is a
modelling assumption, not a measurement, and it is the reason the real-Gemini
run in MEASUREMENTS.md exists: to say how far off this simulator is.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from typing import Sequence

from atg.render import ToolView, ViewParam
from atg.retrieval.text import tokenize
from atg.tasks import Action, Clarify, ToolCall

NOISE = 0.45
MIDDLE_PENALTY = 0.55
INVENT_RATE = 0.62  # how often it fills a value it was never given
_LONG_LIST = 24  # candidate count past which the middle starts getting lost


def _unit_hash(*parts: str) -> float:
    """Deterministic value in [0, 1) from the parts, stable across runs."""
    digest = hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest()
    return (struct.unpack("<Q", digest)[0] >> 11) / float(1 << 53)


_QUOTED = re.compile(r"'([^']+)'")
_EMAIL = re.compile(r"\b[\w.\-+]+@[\w.\-]+\b")
_URL = re.compile(r"https?://[^\s'\"]+")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\b")
_ID = re.compile(r"\b[a-z]{1,5}_\d{5,9}\b")
_MONEY = re.compile(r"\b\d+\.\d{2}\b")
_INT = re.compile(r"\b\d{1,4}\b")
_PATHY = re.compile(r"\b[\w\-/]+/[\w\-./]+\b")
_CHANNEL = re.compile(r"#[\w-]+")
_IMAGE = re.compile(r"\b[\w.\-]+/[\w.\-/]+:[\w.\-]+\b")

_KIND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", _EMAIL),
    ("url", _URL),
    ("date", _DATE),
    ("id", _ID),
    ("money", _MONEY),
    ("path", _PATHY),
    ("int", _INT),
)


_ENUM_WORD = re.compile(
    r"\b(?:true|false|csv|json|parquet|private|team|public|active|paused|archived"
    r"|low|normal|urgent|us-east-1|eu-west-1|ap-south-1)\b"
)
_WORDS_BEFORE = 5
_CONTEXT_WINDOW = 48


@dataclass(frozen=True, slots=True)
class Mention:
    """A literal in the query the model could copy into an argument."""

    value: str
    kinds: frozenset[str]
    context: frozenset[str]
    start: int
    end: int


def _context_words(query: str, start: int) -> frozenset[str]:
    """The words just before a literal, stopwords kept.

    Keeping them matters: 'in', 'to', 'at' and 'as' are exactly the words that
    say which parameter a value belongs to, and a stopword list throws them away.
    """
    window = query[max(0, start - _CONTEXT_WINDOW) : start].lower()
    raw = re.findall(r"[a-z0-9_]+", window)[-_WORDS_BEFORE:]
    return frozenset(raw) | frozenset(tokenize(" ".join(raw)))


def extract_mentions(query: str) -> tuple[Mention, ...]:
    """Every literal in the query, with the words that introduced it.

    Matches nested inside a longer match are dropped, so the host part of a URL
    does not become a candidate path and an enum word inside a quoted sentence
    does not become a candidate enum.
    """
    raw: list[tuple[int, int, str, str]] = []

    def add(start: int, end: int, value: str, kind: str) -> None:
        stripped = value.rstrip(".,;")
        if stripped:
            raw.append((start, start + len(stripped), stripped, kind))

    for match in _QUOTED.finditer(query):
        inner = match.group(1)
        at = match.start(1)
        add(at, match.end(1), inner, "text")
        if _IMAGE.fullmatch(inner) or _PATHY.fullmatch(inner):
            add(at, match.end(1), inner, "path")
    for pattern, kind in ((_URL, "url"), (_EMAIL, "email"), (_IMAGE, "text"), (_CHANNEL, "text")):
        for match in pattern.finditer(query):
            add(match.start(), match.end(), match.group(0), kind)
    for kind, pattern in _KIND_PATTERNS:
        for match in pattern.finditer(query):
            add(match.start(), match.end(), match.group(0), kind)
    for match in _ENUM_WORD.finditer(query):
        word = match.group(0)
        add(match.start(), match.end(), word, "bool" if word in ("true", "false") else "enum")

    spans = {(s, e) for s, e, _, _ in raw}
    kept: dict[tuple[int, int, str], set[str]] = {}
    for start, end, value, kind in raw:
        contained = any(
            (os_ <= start and end <= oe) and (os_, oe) != (start, end) for os_, oe in spans
        )
        if contained:
            continue
        kept.setdefault((start, end, value), set()).add(kind)

    mentions = [
        Mention(value, frozenset(kinds), _context_words(query, start), start, end)
        for (start, end, value), kinds in kept.items()
    ]
    return tuple(sorted(mentions, key=lambda m: m.start))


# English words that point at a parameter, which a real model knows and a
# string-matcher does not. Kept small and generic on purpose: the point is that
# the simulator is not deciding argument placement on a coin flip, not that it
# has been shown how the benchmark phrases things.
_PARAM_CUES: dict[str, tuple[str, ...]] = {
    "name": ("called", "named", "titled"),
    "notes": ("note", "comment", "remark"),
    "subject": ("subject", "line", "re"),
    "body": ("saying", "say", "content", "text", "message"),
    "text": ("saying", "say", "message", "note"),
    "query": ("matching", "match", "search"),
    "fields": ("setting", "set"),
    "format": ("as", "format"),
    "limit": ("cap", "capped", "most", "max"),
    "amount": ("amount", "back", "credit"),
    "recipients": ("to", "list"),
    "to": ("to",),
    "user": ("to", "dm", "privately"),
    "channel": ("channel", "in"),
    "service": ("service", "of"),
    "image": ("image", "build"),
    "bucket": ("bucket", "from", "in"),
    "key": ("key", "download", "upload"),
    "dest": ("to", "over", "into"),
    "owner_email": ("owned", "owner"),
    "effective_at": ("effective", "at", "from"),
    "post_at": ("at", "out"),
    "webhook_url": ("webhook", "through"),
    "callback_url": ("callback", "calling"),
    "template_id": ("template", "using"),
    "path": ("path", "at"),
    "visibility": ("visibility", "visible"),
    "status": ("status",),
    "priority": ("priority",),
    "region": ("region",),
    "dry_run": ("dry", "run"),
}

# Every JSON Schema string parameter is just `"type": "string"`, so an id, a
# path and a sentence are indistinguishable by type alone. The model therefore
# has to place them by name and description, which is exactly the evidence
# compression removes.
_STRINGY = frozenset({"text", "id", "path", "email", "url", "date", "enum"})


def _param_words(param: ViewParam) -> frozenset[str]:
    cues = _PARAM_CUES.get(param.name, ())
    if param.name.endswith("_id"):
        cues = (*cues, "id", *tokenize(param.name[:-3].replace("_", " ")))
    return frozenset(tokenize(param.name.replace("_", " ") + " " + param.description)) | frozenset(cues)


def _kind_of_hint(hint: str) -> str:
    return {
        "string": "text",
        "integer": "int",
        "number": "money",
        "boolean": "bool",
        "date-time": "date",
        "uri": "url",
    }.get(hint, hint)


def _compatible(kind: str, param: ViewParam, mention: Mention) -> bool:
    if param.enum:
        return mention.value in param.enum
    if kind == "text":
        return bool(mention.kinds & _STRINGY)
    return kind in mention.kinds


def _select_score(view: ToolView, query: str, position: int, total: int) -> float:
    """How strongly one rendered candidate matches the query, plus fixed noise."""
    q_terms = tokenize(query)
    q_set = set(q_terms)
    text = view.name.replace("_", " ").replace(".", " ") + " " + view.description
    t_set = set(tokenize(text))
    param_set: set[str] = set()
    for param in view.params:
        param_set |= _param_words(param)

    if not q_set:
        overlap = 0.0
    else:
        overlap = len(q_set & t_set) / math.sqrt(len(q_set) * max(1, len(t_set)))
    param_overlap = len(q_set & param_set) / max(1, len(q_set))

    # The model can only tell that a required argument is satisfiable if it can
    # see the parameter list, which every rendering keeps, and can only tell
    # what a parameter means if a description survived, which compression drops.
    described = sum(1 for p in view.params if p.description) / max(1, len(view.params))

    score = 2.4 * overlap + 0.8 * param_overlap + 0.25 * described
    score += NOISE * (_unit_hash(query, view.name) - 0.5)

    if total > _LONG_LIST:
        # Lost in the middle: ends of a long list get attended to, the centre
        # does not. Peaks at 1.0 in the middle, 0 at either end.
        rel = position / max(1, total - 1)
        score -= MIDDLE_PENALTY * math.sin(math.pi * rel) ** 2
    return score


def choose_tool(views: Sequence[ToolView], query: str) -> ToolView | None:
    if not views:
        return None
    total = len(views)
    best = max(
        enumerate(views),
        key=lambda iv: (_select_score(iv[1], query, iv[0], total), iv[1].name),
    )
    return best[1]


def _invent(param: ViewParam, query: str) -> str:
    kind = _kind_of_hint(param.type_hint)
    seed = _unit_hash(query, param.name, "invent")
    if kind == "email":
        return "team@northwind.example"
    if kind == "id":
        return f"{param.name[:3]}_{int(seed * 9_000_000) + 1_000_000}"
    if kind == "int":
        return str(int(seed * 100) + 1)
    if kind == "bool":
        return "true"
    if kind == "enum" and param.enum:
        return param.enum[int(seed * len(param.enum))]
    if kind == "money":
        return f"{seed * 500:.2f}"
    if kind == "date":
        return "2026-10-15T12:00:00Z"
    if kind == "url":
        return "https://northwind.example/hook"
    if kind == "path":
        return "exports/generated.csv"
    return "as discussed"


def fill_arguments(view: ToolView, query: str) -> Action:
    """Assign query literals to the chosen tool's parameters.

    Assignment is greedy over (type compatibility, context-word overlap with the
    parameter's name and description). Where a rendering dropped the parameter
    descriptions there is less to match on, so values land on the wrong
    parameter more often. That is the accuracy cost of compression, and it is
    produced here rather than assumed anywhere.
    """
    mentions = list(extract_mentions(query))
    args: dict[str, str] = {}
    used: set[str] = set()

    ordered = sorted(view.params, key=lambda p: (not p.required, p.name))
    for param in ordered:
        kind = _kind_of_hint(param.type_hint)
        words = _param_words(param)
        best: tuple[float, str] | None = None
        for mention in mentions:
            if mention.value in used:
                continue
            if not _compatible(kind, param, mention):
                continue
            hits = words & mention.context
            context_score = len(hits) / max(1, len(mention.context))
            exact = 1.0 if param.name in mention.context else 0.0
            # A typed literal landing on a same-typed parameter is weak evidence
            # on its own, but it separates an id from a sentence.
            typed = 0.6 if (kind != "text" and kind in mention.kinds) else 0.0
            score = (
                2.0 * exact
                + 1.8 * context_score
                + typed
                + 0.15 * _unit_hash(query, param.name, mention.value)
            )
            if best is None or score > best[0]:
                best = (score, mention.value)
        if best is not None:
            args[param.name] = best[1]
            used.add(best[1])
        elif param.required:
            if _unit_hash(query, view.name, param.name, "clarify") < INVENT_RATE:
                args[param.name] = _invent(param, query)
            else:
                return Clarify(view.name, param.name)
    return ToolCall(view.name, args)


def decide(views: Sequence[ToolView], query: str) -> Action:
    """One simulated model turn: pick a tool from the candidates, fill its arguments."""
    chosen = choose_tool(views, query)
    if chosen is None:
        return None
    return fill_arguments(chosen, query)


def select_shortlist(query: str, lines: Sequence[tuple[str, str]], budget: int) -> list[str]:
    """Stage one of progressive disclosure, on one-line summaries only.

    Same scorer as `decide`, over views that carry a summary and no parameters,
    because a one-line index is all the model has been shown at this point.
    """
    views = [ToolView(name, summary, (), f"{name}: {summary}", "oneline") for name, summary in lines]
    total = len(views)
    scored = sorted(
        ((_select_score(v, query, i, total), v.name) for i, v in enumerate(views)),
        key=lambda sn: (-sn[0], sn[1]),
    )
    return [name for _, name in scored[:budget]]
