"""Real Gemini backend for the agent loop, plus recording so runs replay offline.

Two calling conventions, because the measurement needs both:

`native` builds one `FunctionDeclaration` per candidate from its JSON Schema.
That is how an agent normally sees tools, and it is what the headline
accuracy-versus-catalog-size numbers use.

`proxy` declares a single `call_tool(name, arguments)` function and puts the
rendered candidate block in the prompt as text. Every compression format can be
expressed that way, so the compression comparison holds the protocol fixed and
varies only the rendering. A native declaration cannot represent a one-line
signature, so comparing formats under `native` would not be comparing formats.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from atg.render import SIGNATURE_LEGEND, ToolView
from atg.tasks import Action, Clarify, ToolCall

DEFAULT_PROJECT = "research-501307"
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-2.5-flash-lite"

Convention = Literal["native", "proxy"]

# List prices used for the dollar figures in MEASUREMENTS.md, in dollars per
# million tokens. These are a calculation from measured token counts, not a bill.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}

SYSTEM_PROMPT = (
    "You are the tool-calling layer of an agent. You are given a user request and a "
    "set of candidate tools. Choose exactly one tool and call it with arguments taken "
    "from the request. Copy argument values verbatim from the request; never invent a "
    "value the request does not contain. If a required argument is genuinely absent "
    "from the request, call ask_user instead, naming the tool you would use and the "
    "single missing argument. Some candidates are near duplicates of each other and "
    "differ only in a constraint stated in their description; read those before "
    "choosing."
)


class GeminiUnavailable(RuntimeError):
    """Raised when the SDK or ADC credentials are not usable."""


def resolve_project() -> str:
    return os.environ.get("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT)


def resolve_location() -> str:
    return os.environ.get("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)


def credentials_available() -> bool:
    """Side-effect-free check for whether a live call is even worth trying."""
    try:
        import google.auth

        google.auth.default()
        return True
    except Exception:
        return False


def build_client() -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise GeminiUnavailable("google-genai not installed; pip install -e '.[gemini]'") from exc
    try:
        return genai.Client(vertexai=True, project=resolve_project(), location=resolve_location())
    except Exception as exc:
        raise GeminiUnavailable(f"could not construct a Vertex AI client: {exc}") from exc


# --------------------------------------------------------------------------


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    retries: int = 0

    def add(self, prompt: int, output: int, total: int) -> None:
        self.prompt_tokens += prompt
        self.output_tokens += output
        self.total_tokens += total
        self.calls += 1

    def cost_usd(self, model: str) -> float:
        rate = PRICE_PER_MTOK.get(model)
        if rate is None:
            return 0.0
        return self.prompt_tokens / 1e6 * rate[0] + self.output_tokens / 1e6 * rate[1]

    def to_dict(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "calls": self.calls,
            "retries": self.retries,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd_list_price": round(self.cost_usd(model), 6),
        }


@dataclass(slots=True)
class CallRecorder:
    """Every live call, as a replayable fixture keyed by prompt identity."""

    model: str
    convention: Convention
    records: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "convention": self.convention,
                    "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "usage": self.usage.to_dict(self.model),
                    "records": self.records,
                },
                indent=2,
            )
        )


def cache_key(query: str, views: Sequence[ToolView], convention: Convention) -> str:
    """Identity of a model call: the request plus exactly what was shown."""
    return json.dumps(
        {
            "q": query,
            "c": convention,
            "v": [v.rendered for v in views],
        },
        sort_keys=True,
    )


# --------------------------------------------------------------------------

_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE")


def generate_with_retry(
    client: Any,
    *,
    max_attempts: int = 7,
    base_delay: float = 8.0,
    usage: Usage | None = None,
    **kwargs: Any,
) -> Any:
    """`generate_content` with exponential backoff and jitter on quota errors."""
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            message = str(exc)
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if str(status) not in ("429", "503", "500") and not any(t in message for t in _RETRYABLE):
                raise
            last = exc
            if usage is not None:
                usage.retries += 1
            if attempt == max_attempts - 1:
                break
            delay = base_delay * (2**attempt) * (0.7 + 0.6 * random.random())
            time.sleep(min(delay, 180.0))
    assert last is not None
    raise last


def _schema_for(view: ToolView) -> Any:
    from google.genai import types

    def one(param: Any) -> Any:
        hint = param.type_hint
        if param.enum:
            return types.Schema(type="STRING", enum=list(param.enum), description=param.description or None)
        if hint == "integer":
            return types.Schema(type="INTEGER", description=param.description or None)
        if hint == "number":
            return types.Schema(type="NUMBER", description=param.description or None)
        if hint == "boolean":
            return types.Schema(type="BOOLEAN", description=param.description or None)
        return types.Schema(type="STRING", description=param.description or None)

    return types.Schema(
        type="OBJECT",
        properties={p.name: one(p) for p in view.params},
        required=[p.name for p in view.params if p.required],
    )


_ASK_USER = "ask_user"


def _declarations(views: Sequence[ToolView], convention: Convention) -> list[Any]:
    from google.genai import types

    ask = types.FunctionDeclaration(
        name=_ASK_USER,
        description="Ask the user for one required argument that the request does not contain.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "tool": types.Schema(type="STRING", description="The tool you would call."),
                "missing": types.Schema(type="STRING", description="The one missing argument name."),
            },
            required=["tool", "missing"],
        ),
    )
    if convention == "proxy":
        call = types.FunctionDeclaration(
            name="call_tool",
            description="Call one of the listed tools by name with its arguments.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "name": types.Schema(type="STRING", description="Exact tool name from the list."),
                    "arguments": types.Schema(
                        type="STRING", description="JSON object of argument name to value."
                    ),
                },
                required=["name", "arguments"],
            ),
        )
        return [types.Tool(function_declarations=[call, ask])]

    decls = [
        types.FunctionDeclaration(
            name=_sanitize(v.name),
            description=v.description[:1000] or v.name,
            parameters=_schema_for(v),
        )
        for v in views
    ]
    return [types.Tool(function_declarations=[*decls, ask])]


def _sanitize(name: str) -> str:
    """Vertex function names allow no dots."""
    return name.replace(".", "__")


def _unsanitize(name: str, views: Sequence[ToolView]) -> str:
    for v in views:
        if _sanitize(v.name) == name:
            return v.name
    return name.replace("__", ".")


def _prompt(query: str, views: Sequence[ToolView], convention: Convention) -> str:
    if convention == "native":
        return f"Request: {query}"
    legend = SIGNATURE_LEGEND + "\n" if views and views[0].fmt in ("signature", "oneline") else ""
    block = "\n".join(v.rendered for v in views)
    return f"{legend}Candidate tools:\n{block}\n\nRequest: {query}"


def _first_call(response: Any) -> tuple[str, dict[str, Any]] | None:
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            fc = getattr(part, "function_call", None)
            if fc is not None:
                return fc.name, dict(fc.args or {})
    return None


def _usage_of(response: Any) -> tuple[int, int, int]:
    u = getattr(response, "usage_metadata", None)
    if u is None:
        return (0, 0, 0)
    return (
        int(u.prompt_token_count or 0),
        int((u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)),
        int(u.total_token_count or 0),
    )


def _to_action(name: str, args: dict[str, Any], views: Sequence[ToolView], convention: Convention) -> Action:
    if name == _ASK_USER:
        return Clarify(str(args.get("tool", "")), str(args.get("missing", "")))
    if convention == "proxy":
        if name != "call_tool":
            return None
        raw = args.get("arguments", "{}")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            parsed = {}
        return ToolCall(str(args.get("name", "")), {k: str(v) for k, v in parsed.items()})
    return ToolCall(_unsanitize(name, views), {k: str(v) for k, v in args.items()})


def make_gemini_decider(
    *,
    client: Any,
    model: str = DEFAULT_MODEL,
    convention: Convention = "native",
    recorder: CallRecorder | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    max_candidates: int | None = 64,
):
    """A `Decider` backed by real Gemini function calling."""
    from google.genai import types

    def decide(views: Sequence[ToolView], query: str) -> Action:
        shown = list(views[:max_candidates]) if max_candidates else list(views)
        key = cache_key(query, shown, convention)
        if cache is not None and key in cache:
            hit = cache[key]
            return _to_action(hit["name"], hit["args"], shown, convention)

        response = generate_with_retry(
            client,
            model=model,
            contents=_prompt(query, shown, convention),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=_declarations(shown, convention),
                temperature=0.0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
            usage=recorder.usage if recorder else None,
        )
        prompt_tokens, output_tokens, total = _usage_of(response)
        call = _first_call(response)
        if recorder is not None:
            recorder.usage.add(prompt_tokens, output_tokens, total)
            recorder.add(
                {
                    "key": key,
                    "query": query,
                    "n_candidates": len(shown),
                    "fmt": shown[0].fmt if shown else None,
                    "name": call[0] if call else None,
                    "args": {k: str(v) for k, v in call[1].items()} if call else {},
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                }
            )
        if call is None:
            return None
        if cache is not None:
            cache[key] = {"name": call[0], "args": {k: str(v) for k, v in call[1].items()}}
        return _to_action(call[0], call[1], shown, convention)

    return decide


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Replay cache from a recorded fixture."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        r["key"]: {"name": r["name"], "args": r["args"]}
        for r in payload.get("records", [])
        if r.get("name")
    }


def make_replay_decider(path: Path, *, convention: Convention = "native", strict: bool = True):
    """A `Decider` that answers only from a recorded fixture. No network."""
    cache = load_cache(path)

    def decide(views: Sequence[ToolView], query: str) -> Action:
        key = cache_key(query, list(views), convention)
        hit = cache.get(key)
        if hit is None:
            if strict:
                raise KeyError(f"no recorded Gemini call for {query[:60]!r} with {len(views)} candidates")
            return None
        return _to_action(hit["name"], hit["args"], list(views), convention)

    return decide
