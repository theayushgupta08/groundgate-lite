from ..config import Settings
from .base import EmbeddingProvider, LLMProvider, ProviderError
from .offline import ExtractiveLLM, HashingEmbeddings
from .ollama import OllamaChat, OllamaEmbeddings
from .openai import OpenAIChat, OpenAIEmbeddings

__all__ = ["EmbeddingProvider", "LLMProvider", "ProviderError", "make_embedder", "make_llm"]


def make_embedder(s: Settings) -> EmbeddingProvider:
    if s.embed_provider == "openai":
        return OpenAIEmbeddings(s.openai_api_key, s.openai_embed_model, s.openai_base_url, s.request_timeout_s)
    if s.embed_provider == "ollama":
        return OllamaEmbeddings(s.ollama_base_url, s.ollama_embed_model)
    return HashingEmbeddings()


def make_llm(s: Settings) -> LLMProvider:
    if s.llm_provider == "openai":
        return OpenAIChat(s.openai_api_key, s.openai_chat_model, s.openai_base_url, s.request_timeout_s)
    if s.llm_provider == "ollama":
        return OllamaChat(s.ollama_base_url, s.ollama_chat_model)
    return ExtractiveLLM()
