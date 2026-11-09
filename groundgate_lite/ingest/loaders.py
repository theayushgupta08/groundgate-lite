"""File loaders. Plain text, Markdown, HTML and PDF out of the box.

For table-heavy or scanned PDFs, plug in Docling (best open-source table structure) or a
managed parser; see. Loader quality sits upstream of
everything else: a bad parse cannot be fixed by a better embedding model.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from ..types import Document

TEXT_EXT = {".txt", ".md", ".markdown", ".rst"}
HTML_EXT = {".html", ".htm"}
PDF_EXT = {".pdf"}


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>", " ", raw)
    # Hidden elements are a classic carrier for indirect prompt injection: drop them.
    raw = re.sub(r'(?is)<([a-z0-9]+)[^>]*(?:display\s*:\s*none|visibility\s*:\s*hidden|'
                 r'font-size\s*:\s*0|aria-hidden="true"|hidden)[^>]*>.*?</\1>', " ", raw)
    for lvl in range(1, 7):
        raw = re.sub(rf"(?is)<h{lvl}[^>]*>(.*?)</h{lvl}>",
                     lambda m, lvl=lvl: "\n\n" + "#" * lvl + " " + m.group(1) + "\n\n", raw)
    raw = re.sub(r"(?is)<(br|/p|/div|/li|/tr)[^>]*>", "\n", raw)
    raw = re.sub(r"(?is)<li[^>]*>", "- ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return html.unescape(raw)


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install pypdf to ingest PDFs") from e
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append(f"<!-- page {i} -->\n{txt}")
    return "\n\n".join(pages)


def load_file(path: Path, *, root: Path | None = None, tenant_id: str = "default") -> Document | None:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        text = path.read_text(encoding="utf-8", errors="replace")
    elif ext in HTML_EXT:
        text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
    elif ext in PDF_EXT:
        text = load_pdf(path)
    else:
        return None
    rel = str(path.relative_to(root)) if root else path.name
    title = _title_from(text) or path.stem.replace("-", " ").replace("_", " ").title()
    return Document(doc_id=rel, text=text, title=title, source=str(path), tenant_id=tenant_id,
                    metadata={"ext": ext})


def load_dir(root: Path, *, tenant_id: str = "default") -> list[Document]:
    docs = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            d = load_file(p, root=root, tenant_id=tenant_id)
            if d and d.text.strip():
                docs.append(d)
    return docs


def _title_from(text: str) -> str:
    m = re.search(r"^#\s+(.+)$", text, re.M)
    return m.group(1).strip() if m else ""
