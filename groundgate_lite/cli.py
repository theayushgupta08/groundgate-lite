"""ggl ingest <dir> | ggl ask "question" | ggl eval evals/golden.jsonl"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .providers import ProviderError

INDEX = ".groundgate-lite/index.json"


def _rag():
    from .engine import Lite

    s = get_settings()
    if s.store == "postgres":
        return Lite(s)
    Path(INDEX).parent.mkdir(parents=True, exist_ok=True)
    return Lite(s, persist_path=INDEX)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ggl", description="GroundGate Lite")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").add_argument("path")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("eval").add_argument("dataset")
    a = p.parse_args(argv)
    try:
        if a.cmd == "ingest":
            print(json.dumps(_rag().ingest_dir(a.path), indent=2))
        elif a.cmd == "ask":
            out = _rag().ask(a.question)
            print(out["answer"])
            for src in out["sources"]:
                print(f"  [{src['n']}] {src['title']} ({src['doc_id']})")
        else:
            from .evals import evaluate

            print(json.dumps(evaluate(_rag(), a.dataset), indent=2))
    except (ProviderError, RuntimeError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
