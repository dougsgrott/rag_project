from abc import ABC, abstractmethod

from rag.types import Chunk, SearchResult

__all__ = ["VectorStore"]


class VectorStore(ABC):
    """Indexes chunks and serves nearest-neighbour search by query.

    `index()` accepts raw `Chunk`s — the VectorStore owns its embedding
    strategy (ADR-0005). Adapters that need an external embedder receive one
    via constructor injection in `compose.py`.
    """

    @abstractmethod
    async def index(self, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        ...
