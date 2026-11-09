"""PostgreSQL + pgvector store.

* HNSW index (m=16, ef_construction=64 build; ef_search=100 query) is the 2026 consensus
  starting point: tune ef_search first (no rebuild), raise m only if that is not enough.
* pgvector >= 0.8 iterative index scans are enabled automatically, so tenant filters do not
  starve the result set.
* Keyword side uses Postgres full-text search with OR semantics (see `_or_tsquery`). Built-in
  ts_rank is not true BM25; RRF only uses ranks, so it works, but for large corpora install
  pg_textsearch or ParadeDB pg_search and swap `keyword_search`.
* Tenant isolation is a WHERE pre-filter on every query. sql/rls.sql adds database-enforced
  row-level security on top.
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg

from ..text import tokens
from ..types import Chunk
from .base import DocState, VectorStore

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS gg_meta (key text PRIMARY KEY, value text NOT NULL);
CREATE TABLE IF NOT EXISTS gg_index_versions (tenant_id text PRIMARY KEY, version bigint NOT NULL);
CREATE TABLE IF NOT EXISTS gg_documents (
    tenant_id    text NOT NULL,
    doc_id       text NOT NULL,
    content_hash text NOT NULL,
    embed_model  text NOT NULL,
    version      int  NOT NULL DEFAULT 1,
    title        text NOT NULL DEFAULT '',
    source       text NOT NULL DEFAULT '',
    updated_at   timestamptz NOT NULL DEFAULT now(),
    chunk_count  int  NOT NULL DEFAULT 0,
    flagged      boolean NOT NULL DEFAULT false,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, doc_id)
);
"""

CHUNKS = """
CREATE TABLE IF NOT EXISTS gg_chunks (
    tenant_id    text NOT NULL,
    chunk_id     text NOT NULL,
    doc_id       text NOT NULL,
    ordinal      int  NOT NULL,
    text         text NOT NULL,
    index_text   text NOT NULL,
    content_hash text NOT NULL,
    title        text NOT NULL DEFAULT '',
    source       text NOT NULL DEFAULT '',
    heading_path text[] NOT NULL DEFAULT '{{}}',
    metadata     jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    quarantined  boolean NOT NULL DEFAULT false,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', index_text)) STORED,
    PRIMARY KEY (tenant_id, chunk_id),
    FOREIGN KEY (tenant_id, doc_id) REFERENCES gg_documents (tenant_id, doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS gg_chunks_doc ON gg_chunks (tenant_id, doc_id);
CREATE INDEX IF NOT EXISTS gg_chunks_tsv ON gg_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS gg_chunks_hnsw ON gg_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = {m}, ef_construction = {efc});
"""

_COLS = ("chunk_id, doc_id, ordinal, text, index_text, content_hash, tenant_id, title, source, "
         "heading_path, metadata")


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def _or_tsquery(query: str) -> tuple[str, list[str]]:
    """websearch_to_tsquery() ANDs every word, so a natural-language question rarely matches
    anything. Build an OR of per-term queries instead and let ranking sort it out."""
    raw = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]*", query)
    terms = [t for t in dict.fromkeys(raw) if tokens(t)][:32]
    if not terms:
        return "", []
    sql = " || ".join(["websearch_to_tsquery('english', %s)"] * len(terms))
    return sql, terms


class PostgresStore(VectorStore):
    def __init__(self, dsn: str, *, m: int = 16, ef_construction: int = 64, ef_search: int = 100) -> None:
        self.dsn = dsn
        self.m, self.efc, self.ef_search = m, ef_construction, ef_search
        self._lock = threading.RLock()
        self._conn = psycopg.connect(dsn, autocommit=False)
        self._dim: int | None = None
        with self._tx() as cur:
            cur.execute(SCHEMA)
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            ver = cur.fetchone()[0]
            self.iterative_scans = tuple(int(x) for x in ver.split(".")[:2]) >= (0, 8)
            cur.execute("SELECT value FROM gg_meta WHERE key='embed_dim'")
            row = cur.fetchone()
            if row:
                self._dim = int(row[0])

    @contextmanager
    def _tx(self):
        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _ensure_chunks_table(self, dim: int) -> None:
        if self._dim == dim:
            return
        if self._dim is not None and self._dim != dim:
            raise RuntimeError(
                f"Index was built with {self._dim}-dim embeddings but the active model returns "
                f"{dim}. Re-index into a fresh database (see the GroundGate guide, model migration).")
        with self._tx() as cur:
            cur.execute(CHUNKS.format(dim=dim, m=self.m, efc=self.efc))
            cur.execute("INSERT INTO gg_meta VALUES ('embed_dim', %s) ON CONFLICT (key) DO UPDATE "
                        "SET value = EXCLUDED.value", (str(dim),))
        self._dim = dim

    # --- writes ---------------------------------------------------------------
    def get_doc_state(self, tenant_id: str, doc_id: str) -> DocState | None:
        with self._tx() as cur:
            cur.execute("SELECT doc_id, tenant_id, content_hash, embed_model, version, title, source,"
                        " updated_at, chunk_count, flagged, metadata FROM gg_documents"
                        " WHERE tenant_id=%s AND doc_id=%s", (tenant_id, doc_id))
            r = cur.fetchone()
        return DocState(*r) if r else None

    def replace_document(self, state: DocState, chunks: list[Chunk]) -> None:
        if chunks:
            self._ensure_chunks_table(len(chunks[0].embedding or []))
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO gg_documents (tenant_id, doc_id, content_hash, embed_model, version,"
                " title, source, updated_at, chunk_count, flagged, metadata)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,now(),%s,%s,%s)"
                " ON CONFLICT (tenant_id, doc_id) DO UPDATE SET content_hash=EXCLUDED.content_hash,"
                " embed_model=EXCLUDED.embed_model, version=EXCLUDED.version, title=EXCLUDED.title,"
                " source=EXCLUDED.source, updated_at=now(), chunk_count=EXCLUDED.chunk_count,"
                " flagged=EXCLUDED.flagged, metadata=EXCLUDED.metadata",
                (state.tenant_id, state.doc_id, state.content_hash, state.embed_model, state.version,
                 state.title, state.source, len(chunks), state.flagged, json.dumps(state.metadata)))
            if self._dim is not None:
                cur.execute("DELETE FROM gg_chunks WHERE tenant_id=%s AND doc_id=%s",
                            (state.tenant_id, state.doc_id))
            if chunks:
                cur.executemany(
                    "INSERT INTO gg_chunks (tenant_id, chunk_id, doc_id, ordinal, text, index_text,"
                    " content_hash, title, source, heading_path, metadata, quarantined, embedding)"
                    " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)",
                    [(c.tenant_id, c.chunk_id, c.doc_id, c.ordinal, c.text, c.index_text, c.content_hash,
                      c.title, c.source, c.heading_path, json.dumps(c.metadata),
                      bool(c.metadata.get("injection_risk")), _vec(c.embedding or [])) for c in chunks])
            self._bump(cur, state.tenant_id)
        state.chunk_count = len(chunks)
        state.updated_at = datetime.now(timezone.utc)

    def delete_document(self, tenant_id: str, doc_id: str) -> bool:
        with self._tx() as cur:
            cur.execute("DELETE FROM gg_documents WHERE tenant_id=%s AND doc_id=%s", (tenant_id, doc_id))
            n = cur.rowcount
            self._bump(cur, tenant_id)
        return n > 0

    def list_documents(self, tenant_id: str) -> list[DocState]:
        with self._tx() as cur:
            cur.execute("SELECT doc_id, tenant_id, content_hash, embed_model, version, title, source,"
                        " updated_at, chunk_count, flagged, metadata FROM gg_documents"
                        " WHERE tenant_id=%s ORDER BY doc_id", (tenant_id,))
            return [DocState(*r) for r in cur.fetchall()]

    def index_version(self, tenant_id: str) -> int:
        with self._tx() as cur:
            cur.execute("SELECT version FROM gg_index_versions WHERE tenant_id=%s", (tenant_id,))
            r = cur.fetchone()
        return int(r[0]) if r else 0

    def all_chunks(self, tenant_id: str) -> list[Chunk]:
        if self._dim is None:
            return []
        with self._tx() as cur:
            cur.execute(f"SELECT {_COLS} FROM gg_chunks WHERE tenant_id=%s ORDER BY doc_id, ordinal",
                        (tenant_id,))
            return [self._row_to_chunk(r) for r in cur.fetchall()]

    @staticmethod
    def _bump(cur, tenant_id: str) -> None:
        cur.execute("INSERT INTO gg_index_versions VALUES (%s, 1) ON CONFLICT (tenant_id) DO UPDATE "
                    "SET version = gg_index_versions.version + 1", (tenant_id,))

    # --- reads ----------------------------------------------------------------
    def _row_to_chunk(self, r) -> Chunk:
        (cid, doc_id, ordinal, text, index_text, h, tenant, title, source, hp, meta) = r
        return Chunk(chunk_id=cid, doc_id=doc_id, ordinal=ordinal, text=text, index_text=index_text,
                     content_hash=h, tenant_id=tenant, title=title, source=source,
                     heading_path=list(hp or []), metadata=meta or {})

    def vector_search(self, tenant_id: str, qvec: list[float], k: int,
                      include_flagged: bool = False) -> list[tuple[Chunk, float]]:
        if self._dim is None:
            return []
        q = _vec(qvec)
        with self._tx() as cur:
            cur.execute(f"SET LOCAL hnsw.ef_search = {max(int(self.ef_search), k)}")
            if self.iterative_scans:
                cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            cur.execute(
                f"SELECT {_COLS}, 1 - (embedding <=> %s::vector) AS sim FROM gg_chunks"
                " WHERE tenant_id = %s AND (%s OR NOT quarantined)"
                " ORDER BY embedding <=> %s::vector LIMIT %s",
                (q, tenant_id, include_flagged, q, k))
            rows = cur.fetchall()
        out = [(self._row_to_chunk(r[:-1]), float(r[-1])) for r in rows]
        out.sort(key=lambda x: -x[1])  # relaxed_order may return slightly unordered rows
        return out

    def keyword_search(self, tenant_id: str, query: str, k: int,
                       include_flagged: bool = False) -> list[tuple[Chunk, float]]:
        if self._dim is None:
            return []
        tsq, terms = _or_tsquery(query)
        if not tsq:
            return []
        with self._tx() as cur:
            cur.execute(
                f"WITH q AS (SELECT ({tsq}) AS query) "
                f"SELECT {_COLS}, ts_rank_cd(tsv, q.query, 1) AS rank FROM gg_chunks, q"
                " WHERE tenant_id = %s AND (%s OR NOT quarantined) AND tsv @@ q.query"
                " ORDER BY rank DESC LIMIT %s",
                (*terms, tenant_id, include_flagged, k))
            rows = cur.fetchall()
        return [(self._row_to_chunk(r[:-1]), float(r[-1])) for r in rows]

    def drift_report(self, tenant_id: str, active_embed_model: str) -> dict:
        with self._tx() as cur:
            cur.execute("SELECT count(*) FROM gg_documents WHERE tenant_id=%s", (tenant_id,))
            docs = cur.fetchone()[0]
            cur.execute("SELECT doc_id FROM gg_documents WHERE tenant_id=%s AND embed_model <> %s",
                        (tenant_id, active_embed_model))
            stale = [r[0] for r in cur.fetchall()]
            if self._dim is None:
                return {"documents": docs, "chunks": 0, "duplicate_chunks": 0, "orphan_chunks": 0,
                        "docs_with_stale_embedding_model": stale, "quarantined_chunks": 0}
            cur.execute("SELECT count(*), count(*) FILTER (WHERE quarantined) FROM gg_chunks"
                        " WHERE tenant_id=%s", (tenant_id,))
            n, q = cur.fetchone()
            cur.execute("SELECT coalesce(sum(c - 1), 0) FROM (SELECT count(*) c FROM gg_chunks"
                        " WHERE tenant_id=%s GROUP BY doc_id, content_hash HAVING count(*) > 1) d",
                        (tenant_id,))
            dup = cur.fetchone()[0]
        # Orphans are impossible here: the foreign key cascades deletes.
        return {"documents": docs, "chunks": n, "duplicate_chunks": int(dup), "orphan_chunks": 0,
                "docs_with_stale_embedding_model": stale, "quarantined_chunks": q}
