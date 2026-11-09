"""Ollama: fully local chat and embeddings. Free to run, no data leaves the machine."""

from __future__ import annotations

import httpx

from .base import Completion, EmbeddingProvider, LLMProvider, post_json

# nomic-embed-text was trained with task prefixes; using them measurably helps retrieval.
_PREFIXES = {
    "nomic-embed-text": {"document": "search_document: ", "query": "search_query: "},
}


class OllamaEmbeddings(EmbeddingProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 120.0, batch_size: int = 64) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self._client = httpx.Client(timeout=timeout)
        self._prefix = next((v for k, v in _PREFIXES.items() if model.startswith(k)), None)

    def embed(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        pre = self._prefix[kind] if self._prefix else ""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [pre + t for t in texts[i : i + self.batch_size]]
            data = post_json(self._client, f"{self.base_url}/api/embed",
                             {"model": self.model, "input": batch}, {})
            out.extend(data["embeddings"])
            self.usage.add(data.get("prompt_eval_count", 0), 0)
        return out


class OllamaChat(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 180.0) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str, *, max_tokens: int = 700,
                 temperature: float | None = None, json_mode: bool = False) -> Completion:
        payload: dict = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"num_predict": max_tokens},
        }
        if temperature is not None:
            payload["options"]["temperature"] = temperature
        if json_mode:
            payload["format"] = "json"
        data = post_json(self._client, f"{self.base_url}/api/chat", payload, {})
        text = data.get("message", {}).get("content", "").strip()
        i, o = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        self.usage.add(i, o)
        return Completion(text, i, o, self.model, data)
