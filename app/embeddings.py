"""
Rate-limited Azure OpenAI embeddings to stay under Azure tier limits (e.g. 429).
Configurable delay via EMBEDDING_RATE_LIMIT_DELAY_SECONDS (default 0.2).
"""
import os
import time
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import AzureOpenAIEmbeddings

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)


def _get_delay() -> float:
    # Default 0.5s to stay under Azure S0 embedding limits; increase if you still see 429
    return float(os.getenv("EMBEDDING_RATE_LIMIT_DELAY_SECONDS", "0.5"))


class RateLimitedAzureOpenAIEmbeddings(Embeddings):
    """Azure OpenAI embeddings with a short delay after each API call to avoid 429."""

    def __init__(self, delay_seconds: float | None = None, **kwargs: Any) -> None:
        # Unset OPENAI_API_BASE so AzureOpenAIEmbeddings uses only azure_endpoint + azure_deployment
        # (agents.py sets OPENAI_API_BASE for CrewAI chat; embedding client must not use it)
        saved_base = os.environ.pop("OPENAI_API_BASE", None)
        try:
            self._client = AzureOpenAIEmbeddings(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                **kwargs,
            )
        finally:
            if saved_base is not None:
                os.environ["OPENAI_API_BASE"] = saved_base
        self._delay = delay_seconds if delay_seconds is not None else _get_delay()

    def _throttle(self) -> None:
        if self._delay > 0:
            time.sleep(self._delay)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._throttle()
        result = self._client.embed_documents(texts)
        return result

    def embed_query(self, text: str) -> list[float]:
        self._throttle()
        return self._client.embed_query(text)


def get_embeddings() -> Embeddings:
    """Return rate-limited embeddings for ingest and retriever."""
    return RateLimitedAzureOpenAIEmbeddings()
