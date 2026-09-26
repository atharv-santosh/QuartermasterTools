"""Demo downstream MCP server: mail. Requires an auth token, held by the gateway."""

from __future__ import annotations

import sys
from typing import Any

from atg.gateway.demo_servers._base import dumps, serve, string_schema

NAME = "mailroom"

_MAILBOXES = {
    "ops-team@northwind.example": ["m_1001", "m_1002"],
    "finance@northwind.example": ["m_2001"],
}
_MESSAGES = {
    "m_1001": {"from": "priya@vendor.example", "subject": "Renewal terms attached", "unread": True},
    "m_1002": {"from": "alice@northwind.example", "subject": "Cutover window confirmed", "unread": False},
    "m_2001": {"from": "d.okafor@partner.example", "subject": "Q3 revenue recap", "unread": True},
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_mailboxes",
        "description": "List the mailboxes this token can read. Read only and safe to cache.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_message",
        "description": "Fetch one message's headers by id. Read only and safe to cache.",
        "inputSchema": string_schema(message_id="Message identifier, for example m_1001."),
    },
    {
        "name": "send_email",
        "description": "Send a single email message immediately. Not idempotent, never cached.",
        "inputSchema": string_schema(
            to="Recipient address.", subject="Subject line.", body="Message body."
        ),
    },
    {
        "name": "email_draft_create",
        "description": "Create an unsent draft. Use when the message should be reviewed first.",
        "inputSchema": string_schema(
            to="Recipient address.", subject="Subject line.", body="Message body."
        ),
    },
]


def _list_mailboxes(_: dict[str, Any]) -> str:
    return dumps({"mailboxes": sorted(_MAILBOXES)})


def _get_message(args: dict[str, Any]) -> str:
    mid = str(args.get("message_id", ""))
    message = _MESSAGES.get(mid)
    if message is None:
        return dumps({"error": f"no message {mid}"})
    return dumps({"message_id": mid, **message})


def _send_email(args: dict[str, Any]) -> str:
    return dumps({"sent": True, "to": args.get("to"), "subject": args.get("subject")})


def _draft(args: dict[str, Any]) -> str:
    return dumps({"draft_id": "d_5001", "to": args.get("to"), "subject": args.get("subject")})


HANDLERS = {
    "list_mailboxes": _list_mailboxes,
    "get_message": _get_message,
    "send_email": _send_email,
    "email_draft_create": _draft,
}


def main() -> int:
    return serve(NAME, TOOLS, HANDLERS, require_auth=True)


if __name__ == "__main__":
    sys.exit(main())
