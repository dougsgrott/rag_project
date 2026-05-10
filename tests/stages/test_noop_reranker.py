import pytest

from rag.adapters.local.reranker import NoOpReranker

from tests.stages._helpers import make_search_results
from tests.stages.reranker_conformance import RerankerConformance


class TestNoOpReranker(RerankerConformance):
    @pytest.fixture
    def reranker(self) -> NoOpReranker:
        return NoOpReranker()

    async def test_truncates_in_input_order(self, reranker: NoOpReranker) -> None:
        candidates = make_search_results(5)
        result = await reranker.rerank("query", candidates, top_k=3)
        assert result == candidates[:3]

    async def test_does_not_modify_scores(self, reranker: NoOpReranker) -> None:
        candidates = make_search_results(3)
        result = await reranker.rerank("query", candidates, top_k=3)
        assert [r.score for r in result] == [c.score for c in candidates]
