from abc import ABC, abstractmethod

from rag.types import Chunk, EmbeddedChunk

__all__ = ["Embedder"]


class Embedder(ABC):
    """Produces vector embeddings for chunks (ingest) and queries (search).

    Note: the Orchestration Layer never calls an `Embedder` directly. Embedders
    are injected into `VectorStore` adapters that need external embedding
    (Chroma, pgvector). Cortex embeds internally and does not use one. See
    ADR-0005.
    """

    @abstractmethod
    async def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        ...
