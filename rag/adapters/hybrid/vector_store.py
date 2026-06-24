"""Hybrid VectorStore — dense + sparse retrieval fused with RRF.

Wraps two inner `VectorStore`s (a dense/vector store and a sparse/BM25 store),
runs both for each query, and merges their ranked lists with Reciprocal Rank
Fusion. This is technique A1 in `docs/MULTIHOP_TECHNIQUES.md`: dense search finds
semantically similar passages, BM25 finds exact entity/date/ticker matches, and
fusing the two lifts single-shot recall so more of a multi-hop query's evidence
lands in one wide net.

The `VectorStore` interface is unchanged — `index`/`search` have the same
signatures — so the query and evaluation pipelines call this polymorphically with
no awareness that retrieval is now hybrid.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from types import TracebackType

from rag.adapters._fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag.stages.vector_store import VectorStore
from rag.types import Chunk, SearchResult

__all__ = ["HybridVectorStore"]


class HybridVectorStore(VectorStore):
    """Fuses a dense and a sparse `VectorStore` via RRF.

    Owns neither inner store's lifecycle in construction — it enters and exits
    whatever it is given through its own `AsyncExitStack`, so both inner stores
    are set up and torn down with this one. `index()` fans the same chunks out to
    both; `search()` queries both concurrently and fuses the results.
    """

    def __init__(
        self,
        *,
        dense: VectorStore,
        sparse: VectorStore,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "HybridVectorStore":
        # Enter each inner store that manages a resource. The `VectorStore`
        # contract does not mandate the async-CM protocol (NoOp stores have no
        # lifecycle), so only stores that implement it are entered — and they are
        # torn down with this wrapper.
        stack = AsyncExitStack()
        await stack.__aenter__()
        for store in (self._dense, self._sparse):
            if isinstance(store, AbstractAsyncContextManager):
                await stack.enter_async_context(store)
        self._stack = stack
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
            self._stack = None

    async def index(self, chunks: list[Chunk]) -> None:
        await asyncio.gather(
            self._dense.index(chunks),
            self._sparse.index(chunks),
        )

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []
        dense_results, sparse_results = await asyncio.gather(
            self._dense.search(query, top_k),
            self._sparse.search(query, top_k),
        )
        return reciprocal_rank_fusion(
            [dense_results, sparse_results], top_k=top_k, rrf_k=self._rrf_k
        )
