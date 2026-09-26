"""Demo downstream MCP server: warehouse. No auth, to exercise the mixed case."""

from __future__ import annotations

import sys
from typing import Any

from atg.gateway.demo_servers._base import dumps, serve, string_schema

NAME = "warehouse"

_STOCK = {
    "SKU-1187": {"warehouse": "eu-west", "on_hand": 412, "reserved": 60},
    "SKU-2290": {"warehouse": "us-east", "on_hand": 0, "reserved": 0},
    "SKU-3341": {"warehouse": "eu-west", "on_hand": 78, "reserved": 12},
}
_RATES = {"eu-west": {"standard": 6.40, "express": 18.90}, "us-east": {"standard": 5.10, "express": 15.75}}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_warehouses",
        "description": "List warehouse codes. Read only and safe to cache.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_stock_level",
        "description": "On-hand and reserved quantity for one SKU. Read only and safe to cache.",
        "inputSchema": string_schema(sku="Stock keeping unit, for example SKU-1187."),
    },
    {
        "name": "get_carrier_rate",
        "description": "Shipping rate for a warehouse and service level. Read only and safe to cache.",
        "inputSchema": string_schema(
            warehouse="Warehouse code.", service="Service level, standard or express."
        ),
    },
    {
        "name": "create_purchase_order",
        "description": "Raise a purchase order for a SKU. Mutating, never cached.",
        "inputSchema": string_schema(sku="Stock keeping unit.", quantity="Units to order."),
    },
    {
        "name": "adjust_stock_level",
        "description": "Apply a signed correction to on-hand stock after a count. Mutating.",
        "inputSchema": string_schema(sku="Stock keeping unit.", delta="Signed adjustment."),
    },
]


def _list_warehouses(_: dict[str, Any]) -> str:
    return dumps({"warehouses": sorted(_RATES)})


def _get_stock(args: dict[str, Any]) -> str:
    sku = str(args.get("sku", ""))
    row = _STOCK.get(sku)
    if row is None:
        return dumps({"error": f"no sku {sku}"})
    return dumps({"sku": sku, **row, "available": row["on_hand"] - row["reserved"]})


def _get_rate(args: dict[str, Any]) -> str:
    rates = _RATES.get(str(args.get("warehouse", "")))
    if rates is None:
        return dumps({"error": "unknown warehouse"})
    service = str(args.get("service", "standard"))
    return dumps({"warehouse": args.get("warehouse"), "service": service, "rate": rates.get(service)})


def _create_po(args: dict[str, Any]) -> str:
    return dumps({"purchase_order_id": "po_8801", "sku": args.get("sku"), "quantity": args.get("quantity")})


def _adjust(args: dict[str, Any]) -> str:
    return dumps({"sku": args.get("sku"), "delta": args.get("delta"), "applied": True})


HANDLERS = {
    "list_warehouses": _list_warehouses,
    "get_stock_level": _get_stock,
    "get_carrier_rate": _get_rate,
    "create_purchase_order": _create_po,
    "adjust_stock_level": _adjust,
}


def main() -> int:
    return serve(NAME, TOOLS, HANDLERS)


if __name__ == "__main__":
    sys.exit(main())
