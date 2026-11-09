"""Provider interfaces. Every LLM/embedding backend implements one of these two classes.

Adding a provider means writing ~40 lines: subclass, implement `embed` or `complete`,
register it in `providers/__init__.py`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, i: int, o: int) -> None:
        self.input_tokens += i
        self.output_tokens += o
        self.calls += 1


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    raw: dict = field(default_factory=dict)


class EmbeddingProvider(ABC):
    name: str = "base"
    model: str = ""

    def __init__(self) -> None:
        self.usage = Usage()

    @abstractmethod
    def embed(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        """Return one vector per input. `kind` is 'document' or 'query' (some models want a prefix)."""

    def embed_one(self, text: str, *, kind: str = "query") -> list[float]:
        return self.embed([text], kind=kind)[0]


class LLMProvider(ABC):
    name: str = "base"
    model: str = ""

    def __init__(self) -> None:
        self.usage = Usage()

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 700,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """Single-turn completion. Keep it single-turn: RAG prompts are rebuilt every request."""


class ProviderError(RuntimeError):
    pass


def explain(url: str, status: int | None, body: str) -> str:
    """Turn a raw provider failure into a message a buyer can act on."""
    host = url.split("/")[2] if "//" in url else url
    if status is None:
        if "11434" in url:
            return ("Cannot reach Ollama at " + host + ". Start the Ollama app (or run `ollama serve`) "
                    "and check GG_OLLAMA_BASE_URL in .env.")
        return f"Cannot reach {host}. Check your internet connection or the *_BASE_URL setting in .env."
    if status in (401, 403):
        return (f"{host} rejected the API key ({status}). Check the key in .env "
                "(GG_OPENAI_API_KEY / GG_ANTHROPIC_API_KEY / GG_COHERE_API_KEY), or create a new one.")
    if status == 404 or "model" in body.lower() and "not found" in body.lower():
        return (f"{host} does not know this model ({status}). Set a current model name in .env, e.g. "
                "GG_OPENAI_CHAT_MODEL / GG_ANTHROPIC_MODEL, or for Ollama run `ollama list` and copy the "
                "exact name.")
    if status == 429:
        return f"{host} rate limit or quota exceeded (429). Wait and retry, or check billing on your account."
    return f"{host} returned {status}: {body[:300]}"


def post_json(
    client: httpx.Client, url: str, payload: dict, headers: dict, *, retries: int = 3
) -> dict:
    """POST with retry on 429/5xx and exponential backoff. Raises ProviderError with the body."""
    delay = 1.0
    last = ""
    status: int | None = None
    for attempt in range(retries + 1):
        try:
            r = client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:  # network error
            last, status = str(e), None
        else:
            if r.status_code < 400:
                return r.json()
            last, status = r.text[:500], r.status_code
            if r.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                break
        if attempt < retries:
            time.sleep(delay)
            delay *= 2
    raise ProviderError(explain(url, status, last))
