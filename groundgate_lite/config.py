"""Settings for GroundGate Lite (env vars with prefix GG_, or a .env file)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GG_", env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "ollama", "offline"] = "offline"
    embed_provider: Literal["openai", "ollama", "offline"] = "offline"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-5-mini"
    openai_embed_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"
    request_timeout_s: float = 60.0

    store: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://groundgate:groundgate@localhost:5432/groundgate"

    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 48
    chunk_min_tokens: int = 64
    candidates_per_retriever: int = 50
    rrf_k: int = 60
    rerank_top_n: int = 20
    final_k: int = 5
    # Below this relevance the answer is "I don't know" and the LLM is never called.
    abstain_min_score: float = 0.30
    max_answer_tokens: int = 600


@lru_cache
def get_settings() -> Settings:
    return Settings()
