"""Embeddings schema utils."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingsSchema(Protocol):
    """EmbeddingsSchema."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed query."""
        ...
