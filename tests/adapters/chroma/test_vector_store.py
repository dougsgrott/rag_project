from typing import AsyncIterator

import pytest

from rag.adapters.chroma.vector_store import ChromaVectorStore
from rag.types import SearchResult

from tests.stages._helpers import StubEmbedder, make_chunks
from tests.stages.vector_store_conformance import VectorStoreConformance


class TestChromaVectorStoreConformance(VectorStoreConformance):
    @pytest.fixture
    async def store(self) -> AsyncIterator[ChromaVectorStore]:  # type: ignore[override]
        async with ChromaVectorStore(
            embedder=StubEmbedder(),
            collection_name="conformance",
        ) as s:
            yield s


class TestChromaVectorStoreUnit:
    async def test_search_returns_indexed_chunk_metadata(self) -> None:
        async with ChromaVectorStore(
            embedder=StubEmbedder(),
            collection_name="unit-meta",
        ) as store:
            chunks = make_chunks(3, source="abc.txt")
            await store.index(chunks)
            results = await store.search("alpha", top_k=3)
        assert len(results) == 3
        sources = {r.chunk.document_source for r in results}
        positions = {r.chunk.position for r in results}
        assert sources == {"abc.txt"}
        assert positions == {0, 1, 2}

    async def test_index_empty_is_noop(self) -> None:
        async with ChromaVectorStore(
            embedder=StubEmbedder(),
            collection_name="unit-empty",
        ) as store:
            await store.index([])
            results = await store.search("alpha", top_k=3)
        assert results == []

    async def test_upsert_avoids_duplicate_ids(self) -> None:
        async with ChromaVectorStore(
            embedder=StubEmbedder(),
            collection_name="unit-upsert",
        ) as store:
            chunks = make_chunks(2, source="x.txt")
            await store.index(chunks)
            await store.index(chunks)  # second pass would explode without upsert
            results = await store.search("anything", top_k=10)
        assert len(results) == 2

    async def test_score_is_float(self) -> None:
        async with ChromaVectorStore(
            embedder=StubEmbedder(),
            collection_name="unit-score",
        ) as store:
            await store.index(make_chunks(1))
            results = await store.search("alpha", top_k=1)
        assert isinstance(results[0], SearchResult)
        assert isinstance(results[0].score, float)

    async def test_use_outside_context_raises(self) -> None:
        store = ChromaVectorStore(embedder=StubEmbedder(), collection_name="unit")
        with pytest.raises(RuntimeError):
            await store.search("x", top_k=1)
