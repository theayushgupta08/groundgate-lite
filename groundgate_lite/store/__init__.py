from ..config import Settings
from .base import DocState, VectorStore
from .memory import MemoryStore

__all__ = ["DocState", "VectorStore", "MemoryStore", "make_store"]


def make_store(s: Settings, persist_path: str | None = None) -> VectorStore:
    if s.store == "postgres":
        from .postgres import PostgresStore

        return PostgresStore(s.database_url)
    return MemoryStore(persist_path)
