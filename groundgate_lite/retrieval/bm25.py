"""Okapi BM25 (k1=1.2, b=0.75). Used by the in-memory store and as a reference implementation.

Vector search misses exact identifiers (error codes, SKUs, version numbers, names); BM25 does
not. Anthropic's contextual-retrieval study and multiple 2026 production write-ups found that
embeddings + BM25 beats embeddings alone.
"""

from __future__ import annotations

import math
from collections import Counter

from ..text import tokens


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.tf: list[Counter] = []
        self.len: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def build(self, items: list[tuple[str, str]]) -> BM25Index:
        self.ids, self.tf, self.len, self.df = [], [], [], Counter()
        for cid, text in items:
            toks = tokens(text)
            c = Counter(toks)
            self.ids.append(cid)
            self.tf.append(c)
            self.len.append(len(toks))
            self.df.update(c.keys())
        self.avgdl = (sum(self.len) / len(self.len)) if self.len else 0.0
        return self

    def idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 50, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = [t for t in dict.fromkeys(tokens(query))]
        if not q or not self.ids:
            return []
        idfs = {t: self.idf(t) for t in q if t in self.df}
        if not idfs:
            return []
        scores: list[tuple[str, float]] = []
        for i, cid in enumerate(self.ids):
            if allowed is not None and cid not in allowed:
                continue
            tf, dl = self.tf[i], self.len[i]
            s = 0.0
            for t, idf in idfs.items():
                f = tf.get(t, 0)
                if f:
                    s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            if s > 0:
                scores.append((cid, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
