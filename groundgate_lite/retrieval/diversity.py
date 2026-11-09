"""Final-context selection: dedup, per-document cap, MMR, and position ordering.

* Exact duplicates (same content hash) are dropped first; MMR alone can let two identical
  chunks through when their metadata differs.
* A per-document cap stops one long document from filling every slot.
* MMR (Carbonell & Goldstein 1998) trades relevance against redundancy; lambda 0.7 keeps
  relevance in the lead.
* "Lost in the middle": models use the start and end of the context best, so the strongest
  chunks go first and last.
"""

from __future__ import annotations

from ..text import content_terms
from ..types import Hit


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def select(hits: list[Hit], *, k: int, mmr_lambda: float = 0.7, max_per_doc: int = 2) -> list[Hit]:
    seen_hash: set[str] = set()
    pool: list[Hit] = []
    for h in hits:
        if h.chunk.content_hash in seen_hash:
            continue
        seen_hash.add(h.chunk.content_hash)
        pool.append(h)
    if not pool:
        return []

    top = max(h.score for h in pool) or 1.0
    terms = {h.chunk.chunk_id: content_terms(h.chunk.text) for h in pool}
    chosen: list[Hit] = []
    per_doc: dict[str, int] = {}
    candidates = list(pool)
    while candidates and len(chosen) < k:
        best, best_val = None, -1e9
        for h in candidates:
            if per_doc.get(h.chunk.doc_id, 0) >= max_per_doc:
                continue
            red = max((_jaccard(terms[h.chunk.chunk_id], terms[c.chunk.chunk_id]) for c in chosen),
                      default=0.0)
            val = mmr_lambda * (h.score / top) - (1 - mmr_lambda) * red
            if val > best_val:
                best, best_val = h, val
        if best is None:
            break
        chosen.append(best)
        per_doc[best.chunk.doc_id] = per_doc.get(best.chunk.doc_id, 0) + 1
        candidates.remove(best)
    return chosen


def order_for_context(hits: list[Hit]) -> list[Hit]:
    """Place the best chunks at both ends: ranks 1,3,5,... then ...,6,4,2."""
    if len(hits) <= 2:
        return hits
    ranked = sorted(hits, key=lambda h: -h.score)
    front = ranked[0::2]
    back = ranked[1::2][::-1]
    return front + back
