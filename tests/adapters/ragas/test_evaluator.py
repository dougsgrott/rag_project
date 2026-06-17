"""Tests for the RAGAS Evaluator adapter.

The RAGAS framework is heavy and LLM-backed, so the unit and conformance
suites mock the single backend seam (`RagasEvaluator._run_ragas`) and assert
the adapter's *own* logic: how it maps RAGAS scores onto `EvaluationResult`
fields and how it honours the reference-presence contract. The real framework
is exercised only by the integration test, gated behind `INTEGRATION=true`.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import pytest

from rag.adapters.ragas.evaluator import RagasEvaluator
from rag.errors import ConfigurationError
from rag.types import Chunk, EvaluationResult, Message, SearchResult

from tests.stages._helpers import make_message, make_search_results
from tests.stages.evaluator_conformance import EvaluatorConformance


# ---------------------------------------------------------------------------
# Helpers — stub the `_run_ragas` seam with a recorded fake.
# ---------------------------------------------------------------------------


def _with_fake_backend(
    evaluator: RagasEvaluator,
    fake: Callable[..., dict[str, float]],
) -> list[dict[str, object]]:
    """Replace `_run_ragas` with `fake`, recording each call's kwargs.

    Returns the list that accumulates call kwargs so tests can assert on what
    the adapter passed to the backend.
    """
    calls: list[dict[str, object]] = []

    def recorder(**kwargs: object) -> dict[str, float]:
        calls.append(kwargs)
        return fake(**kwargs)

    # Instance attribute shadows the bound method; `asyncio.to_thread` calls it
    # with the same keyword arguments.
    evaluator._run_ragas = recorder  # type: ignore[method-assign]
    return calls


def _constant(value: float) -> Callable[..., dict[str, float]]:
    def fake(*, metric_names: list[str], **_: object) -> dict[str, float]:
        return {name: value for name in metric_names}

    return fake


# ---------------------------------------------------------------------------
# Conformance — every Evaluator adapter must pass this suite.
# ---------------------------------------------------------------------------


class TestRagasEvaluatorConformance(EvaluatorConformance):
    @pytest.fixture
    def evaluator(self) -> RagasEvaluator:
        ev = RagasEvaluator(api_key="sk-test")
        _with_fake_backend(ev, _constant(0.5))
        return ev


# ---------------------------------------------------------------------------
# Unit tests — mapping + contract.
# ---------------------------------------------------------------------------


class TestRagasEvaluatorUnit:
    async def test_constructor_rejects_empty_api_key(self) -> None:
        with pytest.raises(ConfigurationError):
            RagasEvaluator(api_key="")

    async def test_maps_all_five_metrics_when_reference_present(self) -> None:
        ev = RagasEvaluator(api_key="k")

        def distinct(*, metric_names: list[str], **_: object) -> dict[str, float]:
            return {
                "faithfulness": 0.11,
                "answer_relevancy": 0.22,
                "context_precision": 0.33,
                "context_recall": 0.44,
                "answer_correctness": 0.55,
            }

        _with_fake_backend(ev, distinct)
        result = await ev.evaluate(
            query="q",
            context=make_search_results(3),
            answer=make_message("assistant", "a"),
            reference="gold",
        )
        assert isinstance(result, EvaluationResult)
        assert result.faithfulness == pytest.approx(0.11)
        assert result.answer_relevancy == pytest.approx(0.22)
        assert result.context_precision == pytest.approx(0.33)
        assert result.context_recall == pytest.approx(0.44)
        assert result.answer_correctness == pytest.approx(0.55)

    async def test_reference_metrics_are_none_without_reference(self) -> None:
        ev = RagasEvaluator(api_key="k")
        _with_fake_backend(ev, _constant(0.7))
        result = await ev.evaluate(
            query="q",
            context=make_search_results(2),
            answer=make_message("assistant", "a"),
            reference=None,
        )
        assert result.faithfulness == pytest.approx(0.7)
        assert result.answer_relevancy == pytest.approx(0.7)
        assert result.context_precision == pytest.approx(0.7)
        assert result.context_recall is None
        assert result.answer_correctness is None

    async def test_reference_dependent_metrics_requested_only_with_reference(self) -> None:
        ev = RagasEvaluator(api_key="k")
        calls = _with_fake_backend(ev, _constant(0.5))

        await ev.evaluate("q", make_search_results(1), make_message(), reference=None)
        await ev.evaluate("q", make_search_results(1), make_message(), reference="gold")

        without_ref = calls[0]["metric_names"]
        with_ref = calls[1]["metric_names"]
        assert without_ref == ["faithfulness", "answer_relevancy", "context_precision"]
        assert "context_recall" in with_ref
        assert "answer_correctness" in with_ref

    async def test_contexts_are_extracted_from_search_result_chunks(self) -> None:
        ev = RagasEvaluator(api_key="k")
        calls = _with_fake_backend(ev, _constant(0.5))
        context = make_search_results(3)

        await ev.evaluate("q", context, make_message("assistant", "ans"), reference="gold")

        passed = calls[0]
        assert passed["contexts"] == [r.chunk.content for r in context]
        assert passed["answer"] == "ans"
        assert passed["query"] == "q"
        assert passed["reference"] == "gold"

    async def test_nan_scores_coerced_to_zero(self) -> None:
        ev = RagasEvaluator(api_key="k")

        def with_nan(*, metric_names: list[str], **_: object) -> dict[str, float]:
            scores = {name: 0.6 for name in metric_names}
            scores["faithfulness"] = math.nan
            return scores

        _with_fake_backend(ev, with_nan)
        result = await ev.evaluate(
            "q", make_search_results(2), make_message("assistant", "a"), reference="gold"
        )
        assert result.faithfulness == 0.0
        assert result.answer_relevancy == pytest.approx(0.6)

    async def test_context_manager_is_a_noop(self) -> None:
        ev = RagasEvaluator(api_key="k")
        async with ev as entered:
            assert entered is ev


# ---------------------------------------------------------------------------
# Integration — real RAGAS call. Gated; never runs in the default suite.
# ---------------------------------------------------------------------------

INTEGRATION = os.environ.get("INTEGRATION", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(
    not INTEGRATION or not os.environ.get("OPENAI_API_KEY"),
    reason="Set INTEGRATION=true and OPENAI_API_KEY (and `uv sync --group ragas`) to run.",
)
class TestRagasEvaluatorIntegration:
    async def test_real_ragas_call_returns_bounded_scores(self) -> None:
        context = [
            SearchResult(
                chunk=Chunk(
                    content="The Eiffel Tower is located in Paris, France.",
                    document_source="paris.txt",
                    position=0,
                ),
                score=1.0,
            )
        ]
        async with RagasEvaluator(api_key=os.environ["OPENAI_API_KEY"]) as ev:
            result = await ev.evaluate(
                query="Where is the Eiffel Tower?",
                context=context,
                answer=Message(role="assistant", content="It is in Paris, France."),
                reference="The Eiffel Tower is in Paris.",
            )
        for value in (
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision,
            result.context_recall,
            result.answer_correctness,
        ):
            assert value is not None
            assert 0.0 <= value <= 1.0
