"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str  # stable id from the source system (path, URL, record id)
    text: str
    title: str = ""
    source: str = ""
    tenant_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str  # f"{doc_id}#{ordinal}@{content_hash[:12]}"
    doc_id: str
    ordinal: int
    text: str  # the raw chunk text shown to the model and the user
    index_text: str  # text that is embedded and keyword-indexed (may carry a context prefix)
    content_hash: str
    tenant_id: str = "default"
    title: str = ""
    source: str = ""
    heading_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class Hit:
    chunk: Chunk
    score: float  # stage-specific score; higher is better
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None


@dataclass
class Claim:
    text: str
    supported: bool
    support: float
    cited: list[int] = field(default_factory=list)
    best_source: int | None = None


@dataclass
class Answer:
    question: str
    answer: str
    abstained: bool
    citations: list[dict[str, Any]]
    hits: list[Hit]
    claims: list[Claim]
    grounding_score: float | None
    trace: dict[str, Any]
    cached: bool = False

    def to_dict(self, include_context: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "grounding_score": self.grounding_score,
            "cached": self.cached,
            "citations": self.citations,
            "claims": [c.__dict__ for c in self.claims],
            "trace": self.trace,
        }
        if include_context:
            out["context"] = [
                {
                    "n": i + 1,
                    "chunk_id": h.chunk.chunk_id,
                    "doc_id": h.chunk.doc_id,
                    "title": h.chunk.title,
                    "heading_path": h.chunk.heading_path,
                    "score": round(h.score, 4),
                    "rerank_score": None if h.rerank_score is None else round(h.rerank_score, 4),
                    "vector_rank": h.vector_rank,
                    "keyword_rank": h.keyword_rank,
                    "text": h.chunk.text,
                }
                for i, h in enumerate(self.hits)
            ]
        return out
