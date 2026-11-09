"""Retrieval evaluation: hit rate@k, MRR@10 and refusal behaviour on a golden set.

The full GroundGate kit adds faithfulness, correctness, citation precision, injection probes,
latency/cost, baselines, threshold calibration and a CI gate that fails pull requests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

from .engine import Lite


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _relevant(hit, item) -> bool:
    ev = [e for e in item.get("evidence", []) if e]
    if ev:
        return any(_norm(e) in _norm(hit.chunk.text) for e in ev)
    return hit.chunk.doc_id in set(item.get("relevant_doc_ids", []))


def evaluate(rag: Lite, dataset_path: str | Path, k: int = 5) -> dict:
    rows = [json.loads(x) for x in Path(dataset_path).read_text().splitlines() if x.strip()]
    hit, mrr, false_ref, correct_ref = [], [], [], []
    for item in rows:
        final, pool, top = rag.retrieve(item["question"])
        if item.get("type", "answerable") == "answerable":
            rel = [_relevant(h, item) for h in pool]
            hit.append(float(any(rel[:k])))
            first = next((i for i, r in enumerate(rel[:10]) if r), None)
            mrr.append(0.0 if first is None else 1 / (first + 1))
            false_ref.append(float(not final))
        elif item.get("type") == "unanswerable":
            correct_ref.append(float(not final))
    r = lambda xs: round(mean(xs), 4) if xs else None  # noqa: E731
    return {"questions": len(rows), f"hit_rate@{k}": r(hit), "mrr@10": r(mrr),
            "false_refusal_rate": r(false_ref), "correct_refusal_rate": r(correct_ref)}
