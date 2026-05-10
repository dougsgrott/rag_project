"""Conformance tests every `Reranker` adapter must pass.

Contract:
- Returned list length is at most `top_k` and at most `len(candidates)`.
- Each returned `SearchResult.chunk` is one of the input candidates'
  chunks — rerankers score, they do not generate.
"""

import pytest

from rag.stages.reranker import Reranker
from rag.types import SearchResult

from tests.stages._helpers import make_search_results


class RerankerConformance:
    @pytest.fixture
    def reranker(self) -> Reranker:
        raise NotImplementedError("subclass must provide a `reranker` fixture")

    async def test_rerank_caps_at_top_k(self, reranker: Reranker) -> None:
        candidates = make_search_results(10)
        result = await reranker.rerank("query", candidates, top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3
        for r in result:
            assert isinstance(r, SearchResult)

    async def test_rerank_caps_at_candidate_count(self, reranker: Reranker) -> None:
        candidates = make_search_results(2)
        result = await reranker.rerank("query", candidates, top_k=10)
        assert len(result) <= len(candidates)

    async def test_rerank_empty_candidates(self, reranker: Reranker) -> None:
        result = await reranker.rerank("query", [], top_k=5)
        assert result == []

    async def test_rerank_returns_subset_of_input_chunks(self, reranker: Reranker) -> None:
        candidates = make_search_results(5)
        input_keys = {(c.chunk.document_source, c.chunk.position) for c in candidates}
        result = await reranker.rerank("query", candidates, top_k=3)
        for r in result:
            assert (r.chunk.document_source, r.chunk.position) in input_keys
