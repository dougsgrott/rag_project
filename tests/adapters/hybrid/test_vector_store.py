from types import TracebackType
from typing import AsyncIterator

import pytest

from rag.adapters.hybrid.vector_store import HybridVectorStore
from rag.adapters.local.bm25_vector_store import BM25VectorStore
from rag.stages.vector_store import VectorStore
from rag.types import Chunk, SearchResult

from tests.stages.vector_store_conformance import VectorStoreConformance


def _result(position: int, score: float, source: str = "d") -> SearchResult:
    return SearchResult(
        chunk=Chunk(content=f"c{position}", document_source=source, position=position),
        score=score,
    )


class _StubStore(VectorStore):
    """A VectorStore that returns a fixed result list and records lifecycle.

    Lets the unit tests drive RRF fusion with deterministic per-store rankings,
    and assert that the hybrid wrapper enters/exits and fans indexing out to both
    inner stores.
    """

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []
        self.indexed: list[Chunk] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "_StubStore":
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.exited = True

    async def index(self, chunks: list[Chunk]) -> None:
        self.indexed.extend(chunks)

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        return list(self._results[:top_k])


class TestHybridVectorStoreConformance(VectorStoreConformance):
    @pytest.fixture
    async def store(self) -> AsyncIterator[HybridVectorStore]:  # type: ignore[override]
        async with HybridVectorStore(
            dense=BM25VectorStore(), sparse=BM25VectorStore()
        ) as s:
            yield s


class TestHybridVectorStoreUnit:
    async def test_rrf_combines_and_orders_both_lists(self) -> None:
        # dense ranks A(0) then B(1); sparse ranks B(0) then C(1).
        # RRF: B is hit by both → highest; A (1/61) edges out C (1/62).
        dense = _StubStore([_result(0, 0.9), _result(1, 0.8)])  # A, B
        sparse = _StubStore([_result(1, 5.0), _result(2, 4.0)])  # B, C
        async with HybridVectorStore(dense=dense, sparse=sparse) as store:
            results = await store.search("q", top_k=10)
        assert [r.chunk.position for r in results] == [1, 0, 2]  # B, A, C

    async def test_score_is_rrf_not_passthrough(self) -> None:
        dense = _StubStore([_result(0, 0.9)])
        sparse = _StubStore([_result(0, 7.0)])
        async with HybridVectorStore(dense=dense, sparse=sparse) as store:
            results = await store.search("q", top_k=10)
        # One chunk hit at rank 0 by both lists: 2 * 1/(60+1).
        assert results[0].score == pytest.approx(2.0 / 61.0)

    async def test_top_k_caps_fused_results(self) -> None:
        dense = _StubStore([_result(0, 0.9), _result(1, 0.8)])
        sparse = _StubStore([_result(2, 5.0), _result(3, 4.0)])
        async with HybridVectorStore(dense=dense, sparse=sparse) as store:
            results = await store.search("q", top_k=2)
        assert len(results) == 2

    async def test_top_k_zero_short_circuits(self) -> None:
        dense = _StubStore([_result(0, 0.9)])
        sparse = _StubStore([_result(1, 5.0)])
        async with HybridVectorStore(dense=dense, sparse=sparse) as store:
            assert await store.search("q", top_k=0) == []

    async def test_one_empty_store_degrades_to_other(self) -> None:
        dense = _StubStore([_result(0, 0.9), _result(1, 0.8)])
        sparse = _StubStore([])
        async with HybridVectorStore(dense=dense, sparse=sparse) as store:
            results = await store.search("q", top_k=10)
        assert [r.chunk.position for r in results] == [0, 1]

    async def test_index_fans_out_to_both_stores(self) -> None:
        dense, sparse = _StubStore(), _StubStore()
        chunks = [_result(i, 0.0).chunk for i in range(3)]
        async with HybridVectorStore(dense=dense, sparse=sparse) as store:
            await store.index(chunks)
        assert dense.indexed == chunks
        assert sparse.indexed == chunks

    async def test_delegates_lifecycle_to_inner_stores(self) -> None:
        dense, sparse = _StubStore(), _StubStore()
        store = HybridVectorStore(dense=dense, sparse=sparse)
        assert not dense.entered and not sparse.entered
        async with store:
            assert dense.entered and sparse.entered
            assert not dense.exited and not sparse.exited
        assert dense.exited and sparse.exited
