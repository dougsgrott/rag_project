from abc import ABC, abstractmethod

from rag.types import SearchResult

__all__ = ["Reranker"]


class Reranker(ABC):
    """Filters a wide candidate set down to a high-quality top-k.

    Sits between `VectorStore.search()` (e.g. top 150) and
    `Generator.generate()` (e.g. top 5). Adapters that score with a
    Cross-Encoder reflect those scores in the returned `SearchResult.score`.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        ...
