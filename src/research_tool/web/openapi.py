"""Write the API's OpenAPI schema (the frontend generates its TypeScript types from it).

    python -m research_tool.web.openapi frontend/src/api/openapi.json

Responses always contain every field (defaults included), but Pydantic lists
fields with defaults as optional. Models that only appear in responses are
therefore marked as having all fields required, so the generated types say so;
request models keep their real optional fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .app import create_app
from .settings import UISettings

REF = "#/components/schemas/"


def _refs(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(REF):
            found.add(ref[len(REF) :])
        for value in node.values():
            found |= _refs(value)
    elif isinstance(node, list):
        for value in node:
            found |= _refs(value)
    return found


def _closure(names: set[str], schemas: dict[str, Any]) -> set[str]:
    todo, seen = list(names), set()
    while todo:
        name = todo.pop()
        if name in seen or name not in schemas:
            continue
        seen.add(name)
        todo.extend(_refs(schemas[name]))
    return seen


def schema() -> dict[str, Any]:
    app = create_app(UISettings(runs_root=Path("/nonexistent"), token="schema-only-not-a-real-token"))
    spec = app.openapi()
    schemas: dict[str, Any] = spec.get("components", {}).get("schemas", {})
    inputs = _closure(
        {name for path in spec["paths"].values() for op in path.values() for name in _refs(op.get("requestBody", {}))},
        schemas,
    )
    for name, model in schemas.items():
        if name not in inputs and isinstance(model.get("properties"), dict):
            model["required"] = sorted(model["properties"])
    return spec


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frontend/src/api/openapi.json")
    target.write_text(json.dumps(schema(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {target}")
