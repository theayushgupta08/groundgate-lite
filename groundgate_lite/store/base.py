"""Vector store interface. Two implementations ship: in-memory (dev, tests, CI) and Postgres +
pgvector (production). Both enforce tenant isolation as a *pre-filter*: other tenants' rows
never become candidates, so they can never leak into a prompt."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from ..types import Chunk


@dataclass
class DocState:
    doc_id: str
    tenant_id: str
    content_hash: str
    embed_model: str
    version: int
    title: str = ""
    source: str = ""
    updated_at: datetime | None = None
    chunk_count: int = 0
    flagged: bool = False
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def get_doc_state(self, tenant_id: str, doc_id: str) -> DocState | None: ...

    @abstractmethod
    def replace_document(self, state: DocState, chunks: list[Chunk]) -> None:
        """Atomically delete the document's old chunks and insert the new ones."""

    @abstractmethod
    def delete_document(self, tenant_id: str, doc_id: str) -> bool: ...

    @abstractmethod
    def list_documents(self, tenant_id: str) -> list[DocState]: ...

    @abstractmethod
    def vector_search(self, tenant_id: str, qvec: list[float], k: int,
                      include_flagged: bool = False) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def keyword_search(self, tenant_id: str, query: str, k: int,
                       include_flagged: bool = False) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def all_chunks(self, tenant_id: str) -> list[Chunk]:
        """Every chunk for a tenant (used by test-set synthesis and exports)."""

    @abstractmethod
    def index_version(self, tenant_id: str) -> int:
        """Monotonic counter bumped on every write. Caches key on it, so ingesting or deleting
        a document invalidates stale answers immediately."""

    @abstractmethod
    def drift_report(self, tenant_id: str, active_embed_model: str) -> dict:
        """Index-hygiene checks: duplicate chunks, chunks embedded with a stale model, orphans."""
