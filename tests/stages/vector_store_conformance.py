"""Conformance tests every `VectorStore` adapter must pass."""

import pytest

from rag.stages.vector_store import VectorStore
from rag.types import SearchResult

from tests.stages._helpers import make_chunks


class VectorStoreConformance:
    @pytest.fixture
    def store(self) -> VectorStore:
        raise NotImplementedError("subclass must provide a `store` fixture")

    async def test_index_then_search_returns_results(self, store: VectorStore) -> None:
        await store.index(make_chunks(5))
        results = await store.search("alpha", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3
        for r in results:
            assert isinstance(r, SearchResult)
            assert isinstance(r.score, float)

    async def test_search_top_k_caps_result_count(self, store: VectorStore) -> None:
        await store.index(make_chunks(5))
        results = await store.search("alpha", top_k=2)
        assert len(results) <= 2
