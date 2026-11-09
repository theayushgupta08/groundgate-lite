"""OpenAI (and any OpenAI-compatible API: OpenRouter, Groq, Together, vLLM, LM Studio...)."""

from __future__ import annotations

import httpx

from .base import Completion, EmbeddingProvider, LLMProvider, ProviderError, post_json


class OpenAIEmbeddings(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str | None, model: str, base_url: str, timeout: float = 60.0,
                 batch_size: int = 128) -> None:
        super().__init__()
        if not api_key:
            raise ProviderError("GG_OPENAI_API_KEY is not set")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(timeout=timeout)

    def embed(self, texts: list[str], *, kind: str = "document") -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            data = post_json(
                self._client, f"{self.base_url}/embeddings",
                {"model": self.model, "input": batch}, self._headers,
            )
            rows = sorted(data["data"], key=lambda d: d["index"])
            out.extend(r["embedding"] for r in rows)
            self.usage.add(data.get("usage", {}).get("prompt_tokens", 0), 0)
        return out


class OpenAIChat(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None, model: str, base_url: str, timeout: float = 60.0) -> None:
        super().__init__()
        if not api_key:
            raise ProviderError("GG_OPENAI_API_KEY is not set")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str, *, max_tokens: int = 700,
                 temperature: float | None = None, json_mode: bool = False) -> Completion:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # GPT-5-class reasoning models count hidden reasoning against this budget,
            # so give them headroom.
            "max_completion_tokens": max_tokens * 4,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = post_json(self._client, f"{self.base_url}/chat/completions", payload, self._headers)
        text = (data["choices"][0]["message"].get("content") or "").strip()
        u = data.get("usage", {})
        self.usage.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return Completion(text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                          data.get("model", self.model), data)
