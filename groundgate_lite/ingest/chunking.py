"""Structure-aware recursive chunking.

Why this design:
* Recursive splitting at ~512 tokens is the benchmark-validated default; more "advanced"
  semantic chunking often produces tiny fragments that retrieve well but answer badly.
* Never cut across a heading; every chunk remembers its heading path (title > H1 > H2).
* Enforce a minimum chunk size so we never index 40-token fragments.
* Keep fenced code blocks and tables whole when they fit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..text import approx_tokens

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)")


@dataclass
class Section:
    heading_path: list[str]
    text: str


def split_sections(text: str) -> list[Section]:
    """Split markdown-ish text on headings, keeping fenced code intact."""
    sections: list[Section] = []
    path: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section([h for _, h in path], body))
        buf.clear()

    for line in text.split("\n"):
        if _FENCE.match(line.strip()):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            path = [(lv, t) for lv, t in path if lv < level] + [(level, title)]
        else:
            buf.append(line)
    flush()
    return sections


_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", r"(?<=[;:,])\s+", " "]


def _recursive_split(text: str, max_tokens: int, seps: list[str] | None = None) -> list[str]:
    if approx_tokens(text) <= max_tokens:
        return [text]
    seps = _SEPARATORS if seps is None else seps
    if not seps:  # hard cut as a last resort
        step = max_tokens * 4
        return [text[i : i + step] for i in range(0, len(text), step)]
    sep, rest = seps[0], seps[1:]
    parts = re.split(sep, text) if sep.startswith("(?") or "\\" in sep else text.split(sep)
    joiner = "\n\n" if sep == "\n\n" else ("\n" if sep == "\n" else " ")
    out: list[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        cand = f"{cur}{joiner}{p}" if cur else p
        if approx_tokens(cand) <= max_tokens:
            cur = cand
            continue
        if cur:
            out.append(cur)
        if approx_tokens(p) > max_tokens:
            out.extend(_recursive_split(p, max_tokens, rest))
            cur = ""
        else:
            cur = p
    if cur:
        out.append(cur)
    return out


def _protect_blocks(text: str) -> list[str]:
    """Split a section into atomic blocks: fenced code and markdown tables stay together."""
    blocks: list[str] = []
    cur: list[str] = []
    mode = None  # None | "fence" | "table"
    for line in text.split("\n"):
        s = line.strip()
        if mode == "fence":
            cur.append(line)
            if _FENCE.match(s):
                blocks.append("\n".join(cur))
                cur, mode = [], None
            continue
        if _FENCE.match(s):
            if cur:
                blocks.append("\n".join(cur))
            cur, mode = [line], "fence"
            continue
        is_table = s.startswith("|") and s.endswith("|")
        if is_table and mode != "table":
            if cur:
                blocks.append("\n".join(cur))
            cur, mode = [line], "table"
            continue
        if mode == "table" and not is_table:
            blocks.append("\n".join(cur))
            cur, mode = [], None
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return [b.strip() for b in blocks if b.strip()]


def chunk_text(
    text: str, *, max_tokens: int = 512, overlap_tokens: int = 48, min_tokens: int = 64
) -> list[tuple[list[str], str]]:
    """Return [(heading_path, chunk_text)]."""
    result: list[tuple[list[str], str]] = []
    for sec in split_sections(text):
        pieces: list[str] = []
        cur = ""
        for block in _protect_blocks(sec.text):
            atomic = block.startswith(("```", "~~~", "|"))
            if atomic and approx_tokens(block) <= max_tokens:
                units = [block]
            else:
                units = _recursive_split(block, max_tokens)
            for u in units:
                cand = f"{cur}\n\n{u}" if cur else u
                if approx_tokens(cand) <= max_tokens:
                    cur = cand
                else:
                    if cur:
                        pieces.append(cur)
                    cur = u
        if cur:
            pieces.append(cur)

        # merge undersized trailing pieces into their neighbour
        merged: list[str] = []
        for p in pieces:
            if merged and (approx_tokens(p) < min_tokens or approx_tokens(merged[-1]) < min_tokens) \
                    and approx_tokens(merged[-1] + p) <= int(max_tokens * 1.25):
                merged[-1] = merged[-1] + "\n\n" + p
            else:
                merged.append(p)

        # light overlap between consecutive chunks of the same section
        for i, p in enumerate(merged):
            if i > 0 and overlap_tokens > 0:
                tail = merged[i - 1][-overlap_tokens * 4 :]
                cut = tail.find(" ")
                tail = tail[cut + 1 :] if cut >= 0 else tail
                p = f"…{tail}\n\n{p}"
            result.append((sec.heading_path, p))

    # Section-level min-size pass: fold tiny sections into the previous one if same parent.
    folded: list[tuple[list[str], str]] = []
    for path, p in result:
        if folded and approx_tokens(p) < min_tokens // 2 and folded[-1][0][:1] == path[:1] \
                and approx_tokens(folded[-1][1] + p) <= max_tokens:
            prev_path, prev = folded[-1]
            heading = " > ".join(path[len(prev_path):]) if len(path) > len(prev_path) else ""
            folded[-1] = (prev_path, prev + ("\n\n" + heading + ":\n" if heading else "\n\n") + p)
        else:
            folded.append((path, p))
    return folded
