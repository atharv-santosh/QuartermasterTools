"""Tool renderings at four compression levels, and the parser the agent sees them through.

The agent never touches a `Tool`. It only ever sees a `ToolView` parsed back out
of a rendered string, so a rendering that drops information really does deprive
the model of it. Compression cost is measured, not asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from atg.catalog import Param, Tool
from atg.tokens import count_tokens

Format = Literal["json_schema", "compact_json", "signature", "oneline"]

FORMATS: tuple[Format, ...] = ("json_schema", "compact_json", "signature", "oneline")

_TYPE_ABBREV: dict[str, str] = {
    "email": "email",
    "id": "id",
    "text": "str",
    "int": "int",
    "enum": "enum",
    "bool": "bool",
    "path": "path",
    "date": "date",
    "money": "num",
    "url": "url",
}
_ABBREV_TYPE = {v: k for k, v in _TYPE_ABBREV.items()}

SIGNATURE_LEGEND = (
    "Tool signatures: name(param:type, ...) - description. "
    "A trailing ! marks a required parameter, a|b|c lists allowed values."
)


@dataclass(frozen=True, slots=True)
class ViewParam:
    name: str
    type_hint: str
    required: bool
    description: str
    enum: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolView:
    """What the agent knows about one candidate tool, and nothing more."""

    name: str
    description: str
    params: tuple[ViewParam, ...]
    rendered: str
    fmt: Format

    @property
    def required(self) -> tuple[ViewParam, ...]:
        return tuple(p for p in self.params if p.required)

    def tokens(self) -> int:
        return count_tokens(self.rendered)


def render(tool: Tool, fmt: Format) -> str:
    if fmt == "json_schema":
        return json.dumps(tool.as_mcp_tool(), indent=2)
    if fmt == "compact_json":
        return json.dumps(tool.as_mcp_tool(), separators=(",", ":"))
    if fmt == "signature":
        return _signature(tool, with_description=True)
    return _signature(tool, with_description=False)


def _param_sig(p: Param) -> str:
    hint = _TYPE_ABBREV[p.kind]
    if p.kind == "enum" and p.enum:
        hint = "|".join(p.enum)
    return f"{p.name}:{hint}{'!' if p.required else ''}"


def _signature(tool: Tool, *, with_description: bool) -> str:
    sig = f"{tool.name}({', '.join(_param_sig(p) for p in tool.params)})"
    if not with_description:
        return sig
    desc = tool.summary
    if tool.precondition:
        desc += " " + tool.precondition.describe()
    return f"{sig} - {desc}"


def view(tool: Tool, fmt: Format) -> ToolView:
    """Render a tool, then parse the rendering back into what the agent can know."""
    rendered = render(tool, fmt)
    return parse_view(rendered, fmt)


def parse_view(rendered: str, fmt: Format) -> ToolView:
    if fmt in ("json_schema", "compact_json"):
        return _parse_json(rendered, fmt)
    return _parse_signature(rendered, fmt)


def _parse_json(rendered: str, fmt: Format) -> ToolView:
    obj = json.loads(rendered)
    schema = obj.get("inputSchema", {})
    props: dict[str, dict] = schema.get("properties", {})
    required = set(schema.get("required", []))
    params = tuple(
        ViewParam(
            name=name,
            type_hint=spec.get("format") or spec.get("type", "string"),
            required=name in required,
            description=spec.get("description", ""),
            enum=tuple(spec.get("enum", ())),
        )
        for name, spec in props.items()
    )
    return ToolView(obj["name"], obj.get("description", ""), params, rendered, fmt)


def _parse_signature(rendered: str, fmt: Format) -> ToolView:
    head, _, desc = rendered.partition(" - ")
    name, _, rest = head.partition("(")
    body = rest.rstrip(")")
    params: list[ViewParam] = []
    if body.strip():
        for chunk in body.split(", "):
            pname, _, hint = chunk.partition(":")
            req = hint.endswith("!")
            hint = hint.rstrip("!")
            enum: tuple[str, ...] = ()
            if "|" in hint:
                enum = tuple(hint.split("|"))
                hint = "enum"
            # A signature carries no per-parameter prose. The agent gets the
            # name and the type and has to work the rest out.
            params.append(ViewParam(pname, _ABBREV_TYPE.get(hint, hint), req, "", enum))
    return ToolView(name, desc.strip(), tuple(params), rendered, fmt)


def render_many(tools: list[Tool], fmt: Format) -> str:
    """The block of text a prompt would carry for a candidate set."""
    if fmt in ("json_schema", "compact_json"):
        body = "\n".join(render(t, fmt) for t in tools)
        return f"Available tools:\n{body}"
    body = "\n".join(render(t, fmt) for t in tools)
    return f"{SIGNATURE_LEGEND}\nAvailable tools:\n{body}"


def block_tokens(tools: list[Tool], fmt: Format) -> int:
    return count_tokens(render_many(tools, fmt))
