"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

score(d) = sum_i  w_i / (k + rank_i(d)),  k = 60 by convention.

RRF uses only ranks, so it needs no score normalisation between BM25 and cosine similarity,
which live on incomparable scales. Weighted RRF lets you lean on one retriever when your eval
set says so.
"""

from __future__ import annotations

from ..types import Chunk, Hit


def rrf(
    vector: list[tuple[Chunk, float]],
    keyword: list[tuple[Chunk, float]],
    *,
    k: int = 60,
    vector_weight: float = 1.0,
    keyword_weight: float = 1.0,
) -> list[Hit]:
    hits: dict[str, Hit] = {}
    for rank, (c, s) in enumerate(vector, start=1):
        h = hits.setdefault(c.chunk_id, Hit(chunk=c, score=0.0))
        h.vector_rank = rank
        h.fused_score += vector_weight / (k + rank)
    for rank, (c, s) in enumerate(keyword, start=1):
        h = hits.setdefault(c.chunk_id, Hit(chunk=c, score=0.0))
        h.keyword_rank = rank
        h.fused_score += keyword_weight / (k + rank)
    out = sorted(hits.values(), key=lambda h: -h.fused_score)
    for h in out:
        h.score = h.fused_score
    return out


def rrf_lists(lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Fuse already-fused lists (used by multi-query rewriting)."""
    merged: dict[str, Hit] = {}
    for lst in lists:
        for rank, h in enumerate(lst, start=1):
            m = merged.setdefault(h.chunk.chunk_id, Hit(chunk=h.chunk, score=0.0,
                                                         vector_rank=h.vector_rank,
                                                         keyword_rank=h.keyword_rank))
            m.fused_score += 1.0 / (k + rank)
    out = sorted(merged.values(), key=lambda h: -h.fused_score)
    for h in out:
        h.score = h.fused_score
    return out
