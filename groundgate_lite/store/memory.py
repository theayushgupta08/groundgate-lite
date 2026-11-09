"""In-memory store: numpy cosine search + BM25. Zero setup; good to ~100k chunks.
Optionally persists to a JSON file so the dev server survives restarts."""

from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..retrieval.bm25 import BM25Index
from ..types import Chunk
from .base import DocState, VectorStore


class MemoryStore(VectorStore):
    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._lock = threading.RLock()
        self.docs: dict[tuple[str, str], DocState] = {}
        self.chunks: dict[str, dict[str, Chunk]] = defaultdict(dict)  # tenant -> id -> chunk
        self._versions: Counter = Counter()
        self._mat: dict[str, tuple[list[str], np.ndarray]] = {}
        self._bm25: dict[str, BM25Index] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()

    # --- writes ---------------------------------------------------------------
    def get_doc_state(self, tenant_id: str, doc_id: str) -> DocState | None:
        return self.docs.get((tenant_id, doc_id))

    def replace_document(self, state: DocState, chunks: list[Chunk]) -> None:
        with self._lock:
            t = state.tenant_id
            for cid in [c for c, ch in self.chunks[t].items() if ch.doc_id == state.doc_id]:
                del self.chunks[t][cid]
            for ch in chunks:
                self.chunks[t][ch.chunk_id] = ch
            state.chunk_count = len(chunks)
            state.updated_at = datetime.now(timezone.utc)
            self.docs[(t, state.doc_id)] = state
            self._dirty(t)

    def delete_document(self, tenant_id: str, doc_id: str) -> bool:
        with self._lock:
            existed = self.docs.pop((tenant_id, doc_id), None) is not None
            for cid in [c for c, ch in self.chunks[tenant_id].items() if ch.doc_id == doc_id]:
                del self.chunks[tenant_id][cid]
            self._dirty(tenant_id)
            return existed

    def list_documents(self, tenant_id: str) -> list[DocState]:
        return sorted((d for (t, _), d in self.docs.items() if t == tenant_id), key=lambda d: d.doc_id)

    def index_version(self, tenant_id: str) -> int:
        return self._versions[tenant_id]

    def all_chunks(self, tenant_id: str) -> list[Chunk]:
        return sorted(self.chunks[tenant_id].values(), key=lambda c: (c.doc_id, c.ordinal))

    def _dirty(self, tenant_id: str) -> None:
        self._versions[tenant_id] += 1
        self._mat.pop(tenant_id, None)
        self._bm25.pop(tenant_id, None)
        if self.persist_path:
            self._save()

    # --- reads ----------------------------------------------------------------
    def _visible(self, tenant_id: str, include_flagged: bool) -> dict[str, Chunk]:
        return {cid: c for cid, c in self.chunks[tenant_id].items()
                if include_flagged or not c.metadata.get("injection_risk")}

    def vector_search(self, tenant_id: str, qvec: list[float], k: int,
                      include_flagged: bool = False) -> list[tuple[Chunk, float]]:
        with self._lock:
            if tenant_id not in self._mat:
                ids = [cid for cid, c in self.chunks[tenant_id].items() if c.embedding]
                m = np.array([self.chunks[tenant_id][i].embedding for i in ids], dtype=np.float32) \
                    if ids else np.zeros((0, len(qvec)), dtype=np.float32)
                if len(m):
                    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
                self._mat[tenant_id] = (ids, m)
            ids, m = self._mat[tenant_id]
            if not ids:
                return []
            q = np.asarray(qvec, dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-9
            sims = m @ q
            order = np.argsort(-sims)
            out = []
            for i in order:
                c = self.chunks[tenant_id][ids[i]]
                if not include_flagged and c.metadata.get("injection_risk"):
                    continue
                out.append((c, float(sims[i])))
                if len(out) >= k:
                    break
            return out

    def keyword_search(self, tenant_id: str, query: str, k: int,
                       include_flagged: bool = False) -> list[tuple[Chunk, float]]:
        with self._lock:
            if tenant_id not in self._bm25:
                self._bm25[tenant_id] = BM25Index().build(
                    [(cid, c.index_text) for cid, c in self.chunks[tenant_id].items()])
            allowed = set(self._visible(tenant_id, include_flagged))
            return [(self.chunks[tenant_id][cid], s)
                    for cid, s in self._bm25[tenant_id].search(query, k, allowed)]

    def drift_report(self, tenant_id: str, active_embed_model: str) -> dict:
        chunks = list(self.chunks[tenant_id].values())
        seen: Counter = Counter((c.doc_id, c.content_hash) for c in chunks)
        doc_ids = {d for (t, d) in self.docs if t == tenant_id}
        stale = [d.doc_id for d in self.list_documents(tenant_id) if d.embed_model != active_embed_model]
        return {
            "documents": len(doc_ids),
            "chunks": len(chunks),
            "duplicate_chunks": sum(n - 1 for n in seen.values() if n > 1),
            "orphan_chunks": sum(1 for c in chunks if c.doc_id not in doc_ids),
            "docs_with_stale_embedding_model": stale,
            "quarantined_chunks": sum(1 for c in chunks if c.metadata.get("injection_risk")),
        }

    # --- persistence ----------------------------------------------------------
    def _save(self) -> None:
        data = {
            "docs": [{**asdict(d), "updated_at": d.updated_at.isoformat() if d.updated_at else None}
                     for d in self.docs.values()],
            "chunks": [asdict(c) for t in self.chunks.values() for c in t.values()],
            "versions": dict(self._versions),
        }
        tmp = self.persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self.persist_path)

    def _load(self) -> None:
        data = json.loads(self.persist_path.read_text())
        for d in data["docs"]:
            d["updated_at"] = datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else None
            st = DocState(**d)
            self.docs[(st.tenant_id, st.doc_id)] = st
        for c in data["chunks"]:
            ch = Chunk(**c)
            self.chunks[ch.tenant_id][ch.chunk_id] = ch
        self._versions.update(data.get("versions", {}))
