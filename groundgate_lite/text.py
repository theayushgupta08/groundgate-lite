"""Small, dependency-free text utilities used by BM25, chunking and grounding checks."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\.]*[A-Za-z0-9]|[A-Za-z0-9]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its may me might my no not of on or our shall should so than that the their them
    then there these they this those to was we were what when where which while who whom why will
    with would you your about above after again all also am any because before being below between
    both during each few further here just more most must nor only other out over own same some
    such too under until up very""".split()
)

# Characters that render as nothing but can hide instructions inside documents.
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
     0x2066, 0x2067, 0x2068, 0x2069]
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    """NFKC-normalise, drop invisible/bidi control characters and collapse whitespace runs."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stem(word: str) -> str:
    """A deliberately tiny suffix stripper. Good enough for BM25/overlap; swap for Snowball if needed."""
    w = word
    for suf in ("ational", "ization", "fulness", "ousness", "iveness", "ations", "ments", "ingly",
                "ments", "ness", "ment", "ings", "ing", "edly", "ies", "ied", "ed", "ly", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            if suf in ("ies", "ied"):
                return w[: -len(suf)] + "y"
            return w[: -len(suf)]
    return w


def tokens(text: str, *, keep_stopwords: bool = False, stemmed: bool = True) -> list[str]:
    out = []
    for m in _WORD.finditer(text.lower()):
        t = m.group(0).strip(".-_")
        if not t:
            continue
        if not keep_stopwords and t in STOPWORDS:
            continue
        out.append(stem(t) if stemmed and t.isalpha() else t)
    return out


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English). Used for budgets, not billing."""
    return max(1, round(len(text) / 4))


_CITE_AFTER_PUNCT = re.compile(r"([.!?])\s*((?:\[\d{1,2}\]\s*)+)")
_LINE_ITEM = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|\|)")


def attach_citations(text: str) -> str:
    """'...fee. [1] Next' -> '...fee [1]. Next' so a citation stays with its own sentence."""
    return _CITE_AFTER_PUNCT.sub(lambda m: " " + m.group(2).strip() + m.group(1) + " ", text).strip()


def sentences(text: str) -> list[str]:
    """Split into sentences. Wrapped lines are re-joined; list items and table rows stay separate."""
    parts: list[str] = []
    for block in re.split(r"\n\s*\n", attach_citations(text)):
        units: list[str] = []
        for line in block.split("\n"):
            line = line.strip()
            if not line or re.fullmatch(r"\|?[\s\-:|]+\|?", line):  # blank or table separator
                continue
            if _LINE_ITEM.match(line) or not units:
                units.append(line)
            else:
                units[-1] += " " + line
        for u in units:
            parts.extend(s.strip() for s in _SENT_SPLIT.split(u) if s.strip())
    return parts


def content_terms(text: str) -> set[str]:
    """Distinct content terms, plus numbers/identifiers kept verbatim (those matter most)."""
    return set(tokens(text))


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


_IDENT = re.compile(r"\b(?=[A-Za-z0-9\-_.]*\d)(?=[A-Za-z0-9\-_.]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9\-_.]{2,}\b")


def query_coverage(question: str, text: str, weights: dict[str, float] | None = None) -> float:
    """Weighted share of the question's content terms present in `text`.

    Identifiers (LMR-4031, v2.3, ap-south-1) and numbers count double: they are what vector
    search most often gets wrong and what users care most about. Optional `weights` (e.g. IDF
    over the candidate pool) down-weight terms that appear everywhere, such as the product name.
    """
    q = content_terms(question)
    if not q:
        return 0.0
    t = content_terms(text)
    idents = {tokens(m)[0] for m in _IDENT.findall(question) if tokens(m)} | numbers_in(question)

    def w(term: str) -> float:
        return (2.0 if term in idents else 1.0) * (weights.get(term, 1.0) if weights else 1.0)

    total = sum(w(x) for x in q)
    return sum(w(x) for x in q if x in t) / total if total else 0.0


def table_rows_as_text(text: str) -> str:
    """Rewrite markdown table rows as 'Header: value; Header: value' lines, so a row read on its
    own still says what each number means."""
    out: list[str] = []
    header: list[str] | None = None
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            if header is None:
                header = cells
                continue
            pairs = [f"{h}: {c}" for h, c in zip(header, cells) if c]
            out.append("- " + "; ".join(pairs) + ".")
        else:
            header = None
            out.append(line)
    return "\n".join(out)
