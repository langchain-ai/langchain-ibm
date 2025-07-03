from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingsSchema(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
