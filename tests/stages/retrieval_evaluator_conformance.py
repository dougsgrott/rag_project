"""Conformance tests every `RetrievalEvaluator` adapter must pass."""

import pytest

from rag.stages.retrieval_evaluator import RetrievalEvaluator
from rag.types import RetrievalResult

from tests.stages._helpers import make_search_results


class RetrievalEvaluatorConformance:
    @pytest.fixture
    def retrieval_evaluator(self) -> RetrievalEvaluator:
        raise NotImplementedError(
            "subclass must provide a `retrieval_evaluator` fixture"
        )

    @staticmethod
    def _assert_in_range(result: RetrievalResult, k: int) -> None:
        assert isinstance(result, RetrievalResult)
        for value in (
            result.recall_at_k,
            result.precision_at_k,
            result.mrr,
            result.ndcg,
            result.hit_rate,
        ):
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
        assert result.k == k

    async def test_returns_retrieval_result_in_range(
        self, retrieval_evaluator: RetrievalEvaluator
    ) -> None:
        result = await retrieval_evaluator.evaluate(
            make_search_results(5), {"doc.txt"}, None, k=5
        )
        self._assert_in_range(result, k=5)

    async def test_chunk_level_qrels_in_range(
        self, retrieval_evaluator: RetrievalEvaluator
    ) -> None:
        result = await retrieval_evaluator.evaluate(
            make_search_results(5), None, {"doc.txt::0", "doc.txt::3"}, k=5
        )
        self._assert_in_range(result, k=5)

    async def test_empty_ranking_in_range(
        self, retrieval_evaluator: RetrievalEvaluator
    ) -> None:
        result = await retrieval_evaluator.evaluate([], {"doc.txt"}, None, k=10)
        self._assert_in_range(result, k=10)
