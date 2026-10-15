"""What the model needs to know about a JSON answer that the server's constraint does not tell it.

vLLM's JSON-schema constraint only limits which tokens may come next: the model
never reads the schema, so field descriptions ("short names", "copied verbatim")
are invisible to it, and any field with a default may simply be skipped. On the
first real run (2026-10-12) that left 88 of 98 paper extractions without a
single method. Two remedies, both generic:

* ``schema_guide`` writes the fields out in plain words for the prompt;
* ``require_all`` marks every field required in the schema sent to the server,
  so each one is written out (an empty list is still allowed).
"""

from __future__ import annotations

import copy
from typing import Any

MAX_DEPTH = 4


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in schema:
        target: Any = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            target = target[part]
        schema = target
    return schema


def _options(schema: dict[str, Any], root: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("anyOf", "oneOf"):
        if key in schema:
            return [_resolve(s, root) for s in schema[key] if s.get("type") != "null"]
    return [schema]


def _bounds(schema: dict[str, Any]) -> str:
    low, high = schema.get("minimum"), schema.get("maximum")
    if low is not None and high is not None:
        return f" {low:g}-{high:g}"
    if low is not None:
        return f" ≥ {low:g}"
    return f" ≤ {high:g}" if high is not None else ""


def _kind(schema: dict[str, Any], root: dict[str, Any]) -> str:
    """A short plain-language type: text, whole number 0-10, "yes" | "no", list of up to 5 texts..."""
    schema = _resolve(schema, root)
    options = _options(schema, root)
    nullable = len(options) < len(schema.get("anyOf", schema.get("oneOf", options)))
    if len(options) > 1:
        return " or ".join(_kind(o, root) for o in options)
    schema = options[0]
    if "enum" in schema or "const" in schema:
        values = schema.get("enum", [schema.get("const")])
        kind = " | ".join(f'"{v}"' if isinstance(v, str) else str(v) for v in values)
    elif schema.get("type") == "array":
        items = _resolve(schema.get("items", {}), root)
        size = f"up to {schema['maxItems']} " if "maxItems" in schema else ""
        inner = "objects" if "properties" in items else _kind(items, root) + "s"
        kind = f"list of {size}{inner}"
    elif schema.get("type") == "integer":
        kind = "whole number" + _bounds(schema)
    elif schema.get("type") == "number":
        kind = "number" + _bounds(schema)
    elif schema.get("type") == "boolean":
        kind = "true or false"
    elif "properties" in schema or schema.get("type") == "object":
        kind = "object"
    else:
        kind = "text"
    return kind + (", or null" if nullable else "")


def _lines(schema: dict[str, Any], root: dict[str, Any], depth: int) -> list[str]:
    lines = []
    indent = "  " * depth
    for name, prop in schema.get("properties", {}).items():
        resolved = _resolve(prop, root)
        description = prop.get("description") or resolved.get("description") or ""
        nested = _options(resolved, root)[0]
        if nested.get("type") == "array":
            nested = _resolve(nested.get("items", {}), root)
        expand = "properties" in nested and depth < MAX_DEPTH
        line = f'{indent}- "{name}" ({_kind(prop, root)}{", each with:" if expand else ""})'
        lines.append(line + (f": {description}" if description else ""))
        if expand:
            lines += _lines(nested, root, depth + 1)
    return lines


def schema_guide(schema: dict[str, Any]) -> str:
    """The fields of a JSON answer, one line each, for the end of a prompt."""
    root = schema
    body = _resolve(schema, root)
    if not body.get("properties"):
        return ""
    return "\n".join(
        [
            "Answer with one JSON object with exactly these fields, in this order. Fill in every field the text "
            "supports; use an empty list or empty text only when it gives nothing for that field.",
            *_lines(body, root, 0),
        ]
    )


def require_all(schema: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``schema`` where every object lists all of its properties as required."""
    out = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("properties"), dict) and node["properties"]:
                node["required"] = list(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(out)
    return out
