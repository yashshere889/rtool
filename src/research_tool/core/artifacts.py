"""Typed, versioned artifact store: the only channel between agents.

Layout::

    artifacts/<name>/v0001.json      # immutable envelope + payload
    artifacts/<name>/v0002.json
    artifacts/.staging/<stage>/<name>.json   # outputs of a running stage, not yet committed

Every artifact name maps to a Pydantic model registered with
``@artifact_type("name")``; payloads are validated on write and on read.
A stage's outputs are *staged* while it runs and *committed* (given new
versions) only when it succeeds, so a failed or killed stage never leaves a
half-finished artifact visible to other stages.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from .errors import ArtifactError, RegistryError
from .fs import atomic_write_json, canonical_json, now_iso, read_json, sha256_text

M = TypeVar("M", bound=BaseModel)

ARTIFACT_TYPES: dict[str, type[BaseModel]] = {}

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _qualname(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def artifact_type(name: str):
    """Class decorator registering a Pydantic model as the schema of artifact ``name``."""
    if not _NAME.match(name):
        raise RegistryError(f"invalid artifact name {name!r} (use lowercase letters, digits, underscores)")

    def decorate(cls: type[M]) -> type[M]:
        register_artifact_type(name, cls)
        return cls

    return decorate


def register_artifact_type(name: str, cls: type[BaseModel]) -> None:
    existing = ARTIFACT_TYPES.get(name)
    if existing is not None and _qualname(existing) != _qualname(cls):
        raise RegistryError(f"artifact {name!r} is already registered to {_qualname(existing)}")
    ARTIFACT_TYPES[name] = cls


def _load_builtin_schemas() -> None:
    """Import every module in research_tool.schemas, so built-in artifact types never depend on import order."""
    import importlib
    import pkgutil

    from .. import schemas

    for module in pkgutil.iter_modules(schemas.__path__):
        importlib.import_module(f"{schemas.__name__}.{module.name}")


def artifact_model(name: str) -> type[BaseModel]:
    if name not in ARTIFACT_TYPES:
        _load_builtin_schemas()
    try:
        return ARTIFACT_TYPES[name]
    except KeyError:
        raise ArtifactError(
            f"no schema registered for artifact {name!r}; decorate its model with @artifact_type({name!r})"
        ) from None


class Producer(BaseModel):
    stage: str
    agent: str
    agent_version: str = ""
    attempt: int = 0
    token: str = ""


class ArtifactEnvelope(BaseModel):
    name: str
    version: int
    schema_ref: str
    producer: Producer
    inputs: dict[str, int] = Field(default_factory=dict)
    created_at: str
    content_hash: str
    meta: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any]


def payload_hash(payload: dict[str, Any]) -> str:
    return sha256_text(canonical_json(payload))


def _check_name(name: str) -> None:
    if not _NAME.match(name):
        raise ArtifactError(f"invalid artifact name {name!r}")


def to_payload(name: str, value: BaseModel | dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate ``value`` against the schema registered for ``name``; return (payload, schema_ref)."""
    cls = artifact_model(name)
    if isinstance(value, BaseModel):
        if not isinstance(value, cls):
            raise ArtifactError(f"artifact {name!r} expects {cls.__name__}, got {type(value).__name__}")
        model = value
    else:
        try:
            model = cls.model_validate(value)
        except ValidationError as exc:
            raise ArtifactError(f"payload for {name!r} does not match {cls.__name__}: {exc}") from None
    return model.model_dump(mode="json"), _qualname(cls)


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        _check_name(name)
        return self.root / name

    def names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and _NAME.match(p.name))

    def versions(self, name: str) -> list[int]:
        directory = self._dir(name)
        if not directory.is_dir():
            return []
        found = []
        for path in directory.glob("v*.json"):
            digits = path.stem[1:]
            if digits.isdigit():
                found.append(int(digits))
        return sorted(found)

    def head(self, name: str) -> int | None:
        versions = self.versions(name)
        return versions[-1] if versions else None

    def heads(self) -> dict[str, int]:
        return {name: v for name in self.names() if (v := self.head(name)) is not None}

    def path(self, name: str, version: int) -> Path:
        return self._dir(name) / f"v{version:04d}.json"

    def envelope(self, name: str, version: int | None = None) -> ArtifactEnvelope:
        resolved = self.head(name) if version is None else version
        if resolved is None:
            raise ArtifactError(f"artifact {name!r} does not exist yet")
        path = self.path(name, resolved)
        if not path.exists():
            raise ArtifactError(f"artifact {name!r} has no version {resolved}")
        return ArtifactEnvelope.model_validate(read_json(path))

    def get(self, name: str, version: int | None = None, model: type[M] | None = None) -> M:
        env = self.envelope(name, version)
        cls = model or artifact_model(name)
        try:
            return cls.model_validate(env.payload)  # type: ignore[return-value]
        except ValidationError as exc:
            raise ArtifactError(f"{name} v{env.version} does not match {cls.__name__}: {exc}") from None

    def version_with_token(self, name: str, token: str) -> int | None:
        """Head version if it was written by the commit identified by ``token`` (crash recovery)."""
        head = self.head(name)
        if head is None or not token:
            return None
        return head if self.envelope(name, head).producer.token == token else None

    def put(
        self,
        name: str,
        value: BaseModel | dict[str, Any],
        *,
        producer: Producer,
        inputs: dict[str, int] | None = None,
        meta: dict[str, Any] | None = None,
        dedupe: bool = True,
    ) -> int:
        """Write a new version and return its number.

        Idempotent per ``producer.token`` (re-committing after a crash returns the
        existing version) and, with ``dedupe``, a payload identical to the head
        version is not written again — so unchanged outputs don't make
        downstream stages stale.
        """
        payload, schema_ref = to_payload(name, value)
        digest = payload_hash(payload)
        head = self.head(name)
        if head is not None:
            previous = self.envelope(name, head)
            if producer.token and previous.producer.token == producer.token:
                return head
            if dedupe and previous.content_hash == digest:
                return head
        version = (head or 0) + 1
        envelope = ArtifactEnvelope(
            name=name,
            version=version,
            schema_ref=schema_ref,
            producer=producer,
            inputs=dict(inputs or {}),
            created_at=now_iso(),
            content_hash=digest,
            meta=dict(meta or {}),
            payload=payload,
        )
        path = self.path(name, version)
        if path.exists():
            raise ArtifactError(f"refusing to overwrite {path}; is another episode writing this run?")
        atomic_write_json(path, envelope.model_dump(mode="json"))
        return version


class StagedOutputs:
    """Outputs written by a running stage; committed only when the stage succeeds."""

    def __init__(self, store: ArtifactStore, stage_id: str, allowed: set[str] | None = None) -> None:
        self.store = store
        self.stage_id = stage_id
        self.allowed = allowed
        self.dir = store.root / ".staging" / stage_id

    def put(self, name: str, value: BaseModel | dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        if self.allowed is not None and name not in self.allowed:
            raise ArtifactError(
                f"stage {self.stage_id!r} may not produce {name!r}; declared outputs: {sorted(self.allowed)}"
            )
        payload, schema_ref = to_payload(name, value)
        atomic_write_json(
            self.dir / f"{name}.json",
            {"name": name, "schema_ref": schema_ref, "meta": meta or {}, "payload": payload},
        )

    def names(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json") if _NAME.match(p.stem))

    def has(self, name: str) -> bool:
        return (self.dir / f"{name}.json").exists()

    def get(self, name: str, model: type[M] | None = None) -> M:
        path = self.dir / f"{name}.json"
        if not path.exists():
            raise ArtifactError(f"stage {self.stage_id!r} has not staged {name!r}")
        cls = model or artifact_model(name)
        return cls.model_validate(read_json(path)["payload"])  # type: ignore[return-value]

    def commit(self, producer: Producer, inputs: dict[str, int]) -> dict[str, int]:
        """Publish every staged output; safe to call again after a crash mid-commit."""
        versions: dict[str, int] = {}
        for name in self.names():
            path = self.dir / f"{name}.json"
            record = read_json(path)
            versions[name] = self.store.put(
                name, record["payload"], producer=producer, inputs=inputs, meta=record.get("meta")
            )
            path.unlink()
        self.discard()
        return versions

    def discard(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
