import pytest

from rag.adapters.local.evaluator import NoOpEvaluator

from tests.stages._helpers import make_message, make_search_results
from tests.stages.evaluator_conformance import EvaluatorConformance


class TestNoOpEvaluator(EvaluatorConformance):
    @pytest.fixture
    def evaluator(self) -> NoOpEvaluator:
        return NoOpEvaluator()

    async def test_returns_zeroed_metrics_with_no_recall(
        self, evaluator: NoOpEvaluator
    ) -> None:
        result = await evaluator.evaluate(
            query="q",
            context=make_search_results(2),
            answer=make_message("assistant", "a"),
            reference=None,
        )
        assert result.faithfulness == 0.0
        assert result.answer_relevancy == 0.0
        assert result.context_precision == 0.0
        assert result.context_recall is None

    async def test_ignores_reference_when_provided(
        self, evaluator: NoOpEvaluator
    ) -> None:
        result = await evaluator.evaluate(
            query="q",
            context=make_search_results(1),
            answer=make_message("assistant", "a"),
            reference="gold answer",
        )
        assert result.context_recall is None
