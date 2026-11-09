"""Lexical reranker: weighted query-term coverage + phrase matches + fused rank.

The score (0..1) doubles as the relevance signal for the "I don't know" gate. Terms present in
every candidate (usually your product name) weigh little; identifiers and numbers weigh double.
The full GroundGate kit adds cross-encoder, Cohere and LLM rerankers plus threshold calibration.
"""

from __future__ import annotations

import math

from ..text import content_terms, query_coverage, tokens
from ..types import Hit


def pool_idf(question: str, hits: list[Hit]) -> dict[str, float]:
    n = len(hits)
    if not n:
        return {}
    texts = [content_terms(h.chunk.index_text) for h in hits]
    return {t: math.log(1 + (n + 1) / (sum(t in x for x in texts) + 0.5)) for t in content_terms(question)}


def rerank(question: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return hits
    top = hits[0].fused_score or 1.0
    w = pool_idf(question, hits)
    qt = tokens(question)
    q_bigrams = set(zip(qt, qt[1:]))
    for h in hits:
        t = tokens(h.chunk.index_text)
        bigram = len(q_bigrams & set(zip(t, t[1:]))) / len(q_bigrams) if q_bigrams else 0.0
        h.rerank_score = min(1.0, 0.8 * query_coverage(question, h.chunk.index_text, w) + 0.2 * bigram)
        h.score = 0.7 * h.rerank_score + 0.3 * (h.fused_score / top)
    return sorted(hits, key=lambda h: -h.score)
