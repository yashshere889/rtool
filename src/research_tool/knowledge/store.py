"""The knowledge base: one SQLite file per run.

Tables: papers (records as JSON), chunks (FTS5 full-text index of paper text),
extractions (per-paper structured facts), vectors (embeddings), citations
(the citation graph among corpus papers), meta.

Network filesystems (Lustre, NFS) and SQLite locking don't mix well, so a
long-running writer can work on a node-local copy (``working_copy``) and sync
it back atomically; downstream agents read an immutable snapshot.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import sqlite3
from array import array
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..core.fs import replace_file, sha256_file
from ..schemas.literature import PaperRecord
from ..tools.documents.sections import Chunk, Section

SCHEMA = """
create table if not exists papers (id text primary key, status text, record text not null);
create virtual table if not exists chunks using fts5(
    paper_id unindexed, seq unindexed, section, text, tokenize = 'porter unicode61'
);
create table if not exists texts (paper_id text primary key, sections text not null);
create table if not exists extractions (paper_id text primary key, data text not null, meta text not null);
create table if not exists vectors (kind text, owner text, model text, vector blob, primary key (kind, owner, model));
create table if not exists citations (src text, dst text, primary key (src, dst));
create table if not exists meta (key text primary key, value text);
"""
_TOKEN = re.compile(r"[A-Za-z0-9]{2,}")


def fts_query(text: str, max_terms: int = 24) -> str:
    """A safe FTS5 query: any of the words (quoted, so punctuation can't break the syntax)."""
    terms = list(dict.fromkeys(t.lower() for t in _TOKEN.findall(text)))[:max_terms]
    return " OR ".join(f'"{t}"' for t in terms)


class KnowledgeStore:
    def __init__(self, path: Path, *, readonly: bool = False) -> None:
        self.path = Path(path)
        if readonly:
            self.db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.execute("pragma journal_mode=delete")
            self.db.executescript(SCHEMA)
        self.db.row_factory = sqlite3.Row

    # ----------------------------------------------------------- lifecycle
    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> KnowledgeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.db.commit()
        self.close()

    def copy_to(self, dest: Path) -> str:
        """Consistent copy of the database to ``dest`` (atomic replace); returns its sha256."""
        self.db.commit()
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.tmp")
        with contextlib.closing(sqlite3.connect(tmp)) as target:
            self.db.backup(target)
        replace_file(tmp, dest)
        return sha256_file(dest)

    @classmethod
    @contextlib.contextmanager
    def working_copy(cls, path: Path, local_dir: Path | None) -> Iterator[KnowledgeStore]:
        """Work on a node-local copy (fast, safe locking) and write it back atomically at the end."""
        if local_dir is None:
            store = cls(path)
            try:
                yield store
            finally:
                store.db.commit()
                store.close()
            return
        local_dir.mkdir(parents=True, exist_ok=True)
        local = local_dir / f"kb-{os.getpid()}-{Path(path).stem}.sqlite"
        if Path(path).exists():
            shutil.copy2(path, local)
        store = cls(local)
        store.home = Path(path)  # type: ignore[attr-defined]
        try:
            yield store
        finally:
            store.copy_to(Path(path))
            store.close()
            local.unlink(missing_ok=True)

    def sync(self) -> None:
        """Write a working copy back to its home now (call at checkpoints)."""
        home = getattr(self, "home", None)
        if home is not None:
            self.copy_to(home)
        else:
            self.db.commit()

    # --------------------------------------------------------------- papers
    def upsert_papers(self, records: Iterable[PaperRecord]) -> None:
        self.db.executemany(
            "insert or replace into papers (id, status, record) values (?, ?, ?)",
            [(r.id, r.status, r.model_dump_json()) for r in records],
        )
        self.db.commit()

    def paper(self, paper_id: str) -> PaperRecord | None:
        row = self.db.execute("select record from papers where id = ?", (paper_id,)).fetchone()
        return PaperRecord.model_validate_json(row["record"]) if row else None

    def papers(self, statuses: Iterable[str] | None = None) -> list[PaperRecord]:
        rows = self.db.execute("select record from papers order by rowid").fetchall()
        records = [PaperRecord.model_validate_json(r["record"]) for r in rows]
        wanted = set(statuses) if statuses is not None else None
        return [r for r in records if wanted is None or r.status in wanted]

    # --------------------------------------------------------------- text
    def set_sections(self, paper_id: str, sections: Iterable[Section]) -> None:
        """Full (reference-free) text of a paper, by section: used for extraction and quote checks."""
        payload = json.dumps([{"title": s.title, "text": s.text} for s in sections])
        self.db.execute("insert or replace into texts (paper_id, sections) values (?, ?)", (paper_id, payload))
        self.db.commit()

    def sections(self, paper_id: str) -> list[Section]:
        row = self.db.execute("select sections from texts where paper_id = ?", (paper_id,)).fetchone()
        return [Section(s["title"], s["text"]) for s in json.loads(row["sections"])] if row else []

    def full_text(self, paper_id: str) -> str:
        return "\n\n".join(s.text for s in self.sections(paper_id))

    def set_chunks(self, paper_id: str, chunks: Iterable[Chunk]) -> int:
        self.db.execute("delete from chunks where paper_id = ?", (paper_id,))
        rows = [(paper_id, c.seq, c.section, c.text) for c in chunks]
        self.db.executemany("insert into chunks (paper_id, seq, section, text) values (?, ?, ?, ?)", rows)
        self.db.commit()
        return len(rows)

    def paper_text(self, paper_id: str) -> str:
        rows = self.db.execute(
            "select section, text from chunks where paper_id = ? order by cast(seq as integer)", (paper_id,)
        ).fetchall()
        return "\n\n".join(r["text"] for r in rows)

    def search(self, query: str, k: int = 10, paper_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Full-text search over paper chunks (BM25 ranking)."""
        match = fts_query(query)
        if not match:
            return []
        sql = "select paper_id, section, text, bm25(chunks) as score from chunks where chunks match ?"
        params: list[Any] = [match]
        if paper_ids is not None:
            ids = list(paper_ids)
            if not ids:
                return []
            sql += f" and paper_id in ({','.join('?' * len(ids))})"
            params += ids
        sql += " order by score limit ?"
        params.append(k)
        return [dict(row) for row in self.db.execute(sql, params).fetchall()]

    # ---------------------------------------------------------- extractions
    def put_extraction(self, paper_id: str, data: dict[str, Any], meta: dict[str, Any]) -> None:
        self.db.execute(
            "insert or replace into extractions (paper_id, data, meta) values (?, ?, ?)",
            (paper_id, json.dumps(data), json.dumps(meta)),
        )
        self.db.commit()

    def extraction(self, paper_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        row = self.db.execute("select data, meta from extractions where paper_id = ?", (paper_id,)).fetchone()
        return (json.loads(row["data"]), json.loads(row["meta"])) if row else None

    def extractions(self) -> dict[str, dict[str, Any]]:
        return {r["paper_id"]: json.loads(r["data"]) for r in self.db.execute("select paper_id, data from extractions")}

    # -------------------------------------------------------------- vectors
    def put_vectors(self, kind: str, model: str, vectors: dict[str, list[float]]) -> None:
        rows = [(kind, owner, model, array("f", vector).tobytes()) for owner, vector in vectors.items()]
        self.db.executemany("insert or replace into vectors (kind, owner, model, vector) values (?, ?, ?, ?)", rows)
        self.db.commit()

    def vectors(self, kind: str, model: str) -> dict[str, list[float]]:
        out = {}
        for row in self.db.execute("select owner, vector from vectors where kind = ? and model = ?", (kind, model)):
            values = array("f")
            values.frombytes(row["vector"])
            out[row["owner"]] = list(values)
        return out

    # ------------------------------------------------------------ citations
    def add_citations(self, pairs: Iterable[tuple[str, str]]) -> None:
        self.db.executemany("insert or ignore into citations (src, dst) values (?, ?)", list(pairs))
        self.db.commit()

    def citations(self) -> list[tuple[str, str]]:
        return [(r["src"], r["dst"]) for r in self.db.execute("select src, dst from citations")]

    # ----------------------------------------------------------------- meta
    def set_meta(self, key: str, value: Any) -> None:
        self.db.execute("insert or replace into meta (key, value) values (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("select value from meta where key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default
