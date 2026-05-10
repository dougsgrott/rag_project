"""Conformance tests every `Evaluator` adapter must pass."""

import pytest

from rag.stages.evaluator import Evaluator
from rag.types import EvaluationResult

from tests.stages._helpers import make_message, make_search_results


class EvaluatorConformance:
    @pytest.fixture
    def evaluator(self) -> Evaluator:
        raise NotImplementedError("subclass must provide an `evaluator` fixture")

    async def test_evaluate_without_reference_returns_evaluation_result(
        self, evaluator: Evaluator
    ) -> None:
        result = await evaluator.evaluate(
            query="what is alpha?",
            context=make_search_results(3),
            answer=make_message("assistant", "alpha is the first letter"),
            reference=None,
        )
        assert isinstance(result, EvaluationResult)
        assert isinstance(result.faithfulness, float)
        assert isinstance(result.answer_relevancy, float)
        assert isinstance(result.context_precision, float)
        assert result.context_recall is None or isinstance(result.context_recall, float)

    async def test_evaluate_metric_fields_are_floats(self, evaluator: Evaluator) -> None:
        result = await evaluator.evaluate(
            query="what is alpha?",
            context=make_search_results(2),
            answer=make_message("assistant", "alpha"),
            reference="alpha is the first letter",
        )
        assert isinstance(result, EvaluationResult)
        for field_value in (
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision,
        ):
            assert isinstance(field_value, float)
