import pytest

from rag.adapters.local.bm25_vector_store import BM25VectorStore
from rag.types import Chunk, SearchResult

from tests.stages._helpers import make_chunks
from tests.stages.vector_store_conformance import VectorStoreConformance


def _chunk(position: int, content: str, source: str = "doc.txt") -> Chunk:
    return Chunk(content=content, document_source=source, position=position)


class TestBM25VectorStoreConformance(VectorStoreConformance):
    @pytest.fixture
    def store(self) -> BM25VectorStore:  # type: ignore[override]
        return BM25VectorStore()


class TestBM25VectorStoreUnit:
    async def test_search_ranks_lexical_match_first(self) -> None:
        store = BM25VectorStore()
        await store.index(
            [
                _chunk(0, "the central bank raised interest rates"),
                _chunk(1, "a recipe for banana bread and butter"),
                _chunk(2, "interest rates and the bank decision"),
            ]
        )
        results = await store.search("bank interest rates", top_k=3)
        assert results, "expected at least one lexical match"
        # The banana-bread chunk shares no query terms and must not appear.
        assert all("banana" not in r.chunk.content for r in results)
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0.0

    async def test_no_term_overlap_returns_empty(self) -> None:
        store = BM25VectorStore()
        await store.index(make_chunks(5))  # content "chunk-0".."chunk-4"
        assert await store.search("alpha beta gamma", top_k=3) == []

    async def test_top_k_caps_result_count(self) -> None:
        store = BM25VectorStore()
        await store.index(
            [_chunk(i, f"shared token number {i}") for i in range(5)]
        )
        results = await store.search("shared token", top_k=2)
        assert len(results) == 2

    async def test_empty_index_search_returns_empty(self) -> None:
        store = BM25VectorStore()
        assert await store.search("anything", top_k=3) == []

    async def test_index_empty_is_noop(self) -> None:
        store = BM25VectorStore()
        await store.index([])
        assert await store.search("anything", top_k=3) == []

    async def test_reindex_same_chunks_does_not_duplicate(self) -> None:
        store = BM25VectorStore()
        chunks = [_chunk(0, "shared token here"), _chunk(1, "another shared token")]
        await store.index(chunks)
        await store.index(chunks)  # upsert: must not duplicate
        results = await store.search("shared token", top_k=10)
        ids = [(r.chunk.document_source, r.chunk.position) for r in results]
        assert len(ids) == len(set(ids)) == 2

    async def test_top_k_zero_returns_empty(self) -> None:
        store = BM25VectorStore()
        await store.index([_chunk(0, "shared token")])
        assert await store.search("shared", top_k=0) == []

    async def test_is_trivial_async_context_manager(self) -> None:
        async with BM25VectorStore() as store:
            await store.index([_chunk(0, "shared token")])
            results = await store.search("shared", top_k=1)
        assert len(results) == 1
