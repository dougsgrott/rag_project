import pytest

from rag.adapters.local.retrieval_evaluator import NoOpRetrievalEvaluator

from tests.stages._helpers import make_search_results
from tests.stages.retrieval_evaluator_conformance import RetrievalEvaluatorConformance


class TestNoOpRetrievalEvaluator(RetrievalEvaluatorConformance):
    @pytest.fixture
    def retrieval_evaluator(self) -> NoOpRetrievalEvaluator:
        return NoOpRetrievalEvaluator()

    async def test_returns_zeroed_metrics(
        self, retrieval_evaluator: NoOpRetrievalEvaluator
    ) -> None:
        result = await retrieval_evaluator.evaluate(
            make_search_results(5), {"doc.txt"}, None, k=5
        )
        assert result.recall_at_k == 0.0
        assert result.precision_at_k == 0.0
        assert result.mrr == 0.0
        assert result.ndcg == 0.0
        assert result.hit_rate == 0.0
        assert result.k == 5
