from rag.stages.reranker import Reranker
from rag.types import SearchResult

__all__ = ["NoOpReranker"]


class NoOpReranker(Reranker):
    """Passthrough: truncates the candidate list to `top_k` with no scoring."""

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        return list(candidates[:top_k])
