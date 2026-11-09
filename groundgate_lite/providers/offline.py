"""Deterministic offline stand-ins so the whole kit (API, UI, evals, CI) runs with no API key.

They are honest baselines, not toys:
* HashingEmbeddings: hashed stemmed unigrams + bigrams + character trigrams. Lexical, not
  semantic, so it shows exactly why hybrid search and reranking matter.
* ExtractiveLLM: answers by quoting the best-supported sentences with [n] citations, and
  performs the judge/rewrite/synthesis tasks with transparent heuristics.

Swap in a real provider with one env var once you want real quality.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re

from ..text import content_terms, query_coverage, sentences, table_rows_as_text, tokens
from .base import Completion, EmbeddingProvider, LLMProvider


def _h(s: str, dim: int) -> tuple[int, float]:
    d = hashlib.blake2b(s.encode(), digest_size=8).digest()
    v = int.from_bytes(d, "little")
    return v % dim, (1.0 if (v >> 40) & 1 else -1.0)


class HashingEmbeddings(EmbeddingProvider):
    name = "offline"
    model = "hashing-v1"

    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = tokens(text)
        feats: dict[str, float] = {}
        for t in toks:
            feats["u:" + t] = feats.get("u:" + t, 0) + 1.0
        for a, b in zip(toks, toks[1:]):
            feats[f"b:{a}_{b}"] = feats.get(f"b:{a}_{b}", 0) + 0.7
        for t in set(toks):
            padded = f"#{t}#"
            for i in range(len(padded) - 2):
                k = "c:" + padded[i : i + 3]
                feats[k] = feats.get(k, 0) + 0.15
        for f, w in feats.items():
            idx, sign = _h(f, self.dim)
            v[idx] += sign * (1 + math.log(w)) if w >= 1 else sign * w
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        self.usage.add(sum(len(t) // 4 for t in texts), 0)
        return [self._vec(t) for t in texts]


_DOC_RE = re.compile(r'<document n="(\d+)"[^>]*?section="([^"]*)"[^>]*>(.*?)</document>', re.S)


class ExtractiveLLM(LLMProvider):
    name = "offline"
    model = "extractive-v1"

    def complete(self, system: str, user: str, *, max_tokens: int = 700,
                 temperature: float | None = None, json_mode: bool = False) -> Completion:
        task = re.search(r"Task:\s*([a-z_]+)", system)
        task_name = task.group(1) if task else "answer"
        handler = getattr(self, f"_t_{task_name}", self._t_answer)
        text = handler(user)
        self.usage.add(len(system + user) // 4, len(text) // 4)
        return Completion(text, len(system + user) // 4, len(text) // 4, self.model)

    # --- grounded answering -------------------------------------------------
    def _t_answer(self, user: str) -> str:
        q = html.unescape(_field(user, "question"))
        scored: list[tuple[float, int, str]] = []
        for n, section, body in _DOC_RE.findall(user):
            head = html.unescape(section).split(" / ")[-1] if section else ""
            for s in sentences(table_rows_as_text(html.unescape(body))):
                if len(content_terms(s)) < 2:
                    continue
                # Say which section a sentence comes from when the sentence leaves it implicit.
                if head and not (content_terms(head) & content_terms(s)):
                    s = f"{head}: {s}"
                scored.append((query_coverage(q, s), int(n), s))
        scored.sort(key=lambda x: -x[0])
        top = scored[0][0] if scored else 0.0
        best = [x for x in scored if x[0] >= max(0.34, 0.85 * top)][:2]
        if not best:
            return "I don't know based on the provided documents."
        seen, out = set(), []
        for _, n, s in best:
            if s in seen:
                continue
            seen.add(s)
            out.append(f"{s.rstrip().rstrip('.')} [{n}].")
        return " ".join(out)

    # --- claim verification (judge) ----------------------------------------
    def _t_verify_claims(self, user: str) -> str:
        claims = json.loads(_field(user, "claims") or "[]")
        context = _field(user, "context")
        cterms = content_terms(context)
        verdicts = []
        for c in claims:
            t = content_terms(c)
            sup = len(t & cterms) / len(t) if t else 1.0
            verdicts.append({"claim": c, "supported": sup >= 0.8, "reason": f"term overlap {sup:.2f}"})
        return json.dumps({"verdicts": verdicts})

    def _t_judge_relevance(self, user: str) -> str:
        q, a = content_terms(_field(user, "question")), content_terms(_field(user, "answer"))
        score = len(q & a) / len(q) if q else 0.0
        return json.dumps({"score": round(min(1.0, score * 1.25), 3), "reason": "term overlap"})

    def _t_judge_correctness(self, user: str) -> str:
        gt, a = content_terms(_field(user, "reference")), content_terms(_field(user, "answer"))
        score = len(gt & a) / len(gt) if gt else 0.0
        return json.dumps({"score": round(score, 3), "reason": "reference term recall"})

    # --- retrieval helpers ---------------------------------------------------
    def _t_rewrite_multi(self, user: str) -> str:
        q = _field(user, "question")
        kw = " ".join(tokens(q, stemmed=False))
        return json.dumps({"queries": [q, kw]})

    def _t_rewrite_hyde(self, user: str) -> str:
        return _field(user, "question")

    def _t_rewrite_decompose(self, user: str) -> str:
        q = _field(user, "question")
        parts = [p.strip(" ?") + "?" for p in re.split(r"\band\b|;", q) if len(p.strip()) > 8]
        return json.dumps({"queries": parts or [q]})

    def _t_rerank(self, user: str) -> str:
        q = content_terms(_field(user, "question"))
        scores = []
        for n, _, body in _DOC_RE.findall(user):
            t = content_terms(body)
            scores.append({"n": int(n), "score": round(len(q & t) / len(q), 3) if q else 0})
        return json.dumps({"scores": scores})

    def _t_contextualize(self, user: str) -> str:
        doc = _field(user, "document")
        first = sentences(doc)[:1]
        return f"This chunk is from a document that begins: {first[0][:200] if first else ''}"

    def _t_synthesize(self, user: str) -> str:
        chunk = _field(user, "chunk")
        cands = [s for s in sentences(chunk) if 6 <= len(s.split()) <= 40]
        if not cands:
            return json.dumps({"items": []})
        s = max(cands, key=lambda x: len(content_terms(x)))
        terms = [t for t in tokens(s, stemmed=False)][:6]
        return json.dumps({"items": [{
            "question": "What does the documentation say about " + " ".join(terms) + "?",
            "answer": s,
        }]})


def _field(user: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", user, re.S)
    return m.group(1).strip() if m else ""
