"""Tests for the offline evaluation pipeline."""

import pytest

from rag.adapters.local.evaluator import NoOpEvaluator
from rag.adapters.local.retrieval_evaluator import LocalRetrievalEvaluator
from rag.pipeline.evaluate import (
    EvalCase,
    EvalRecord,
    EvaluationReport,
    aggregate,
    evaluate_test_set,
)
from rag.types import EvaluationResult, Message, RetrievalResult

from tests.pipeline.test_query import (
    _Recorder,
    _StubConversationStore,
    _StubGenerator,
    _StubPromptStore,
    _StubRewriter,
    _StubReranker,
    _StubVectorStore,
)
from tests.stages._helpers import make_search_results


# --- aggregate(): pure unit ------------------------------------------------


def _record(
    faithfulness=0.0,
    answer_relevancy=0.0,
    context_precision=0.0,
    context_recall=None,
    answer_correctness=None,
):
    has_reference = context_recall is not None or answer_correctness is not None
    return EvalRecord(
        case=EvalCase(query="q", reference="ref" if has_reference else None),
        answer=Message("assistant", "a"),
        context=[],
        result=EvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            answer_correctness=answer_correctness,
        ),
    )


def test_aggregate_empty_returns_zeroed_report() -> None:
    report = aggregate([])
    assert isinstance(report, EvaluationReport)
    assert report.records == []
    assert report.mean_faithfulness == 0.0
    assert report.mean_answer_relevancy == 0.0
    assert report.mean_context_precision == 0.0
    assert report.mean_context_recall is None
    assert report.mean_answer_correctness is None


def test_aggregate_means_per_metric() -> None:
    records = [
        _record(0.8, 0.9, 0.7, 0.6),
        _record(0.6, 0.7, 0.5, 0.4),
    ]
    report = aggregate(records)
    assert report.mean_faithfulness == pytest.approx(0.7)
    assert report.mean_answer_relevancy == pytest.approx(0.8)
    assert report.mean_context_precision == pytest.approx(0.6)
    assert report.mean_context_recall == pytest.approx(0.5)


def test_aggregate_context_recall_only_averages_referenced_cases() -> None:
    # Two cases supply a reference (recall set), one does not (recall None).
    records = [
        _record(0.0, 0.0, 0.0, 0.8),
        _record(0.0, 0.0, 0.0, None),
        _record(0.0, 0.0, 0.0, 0.4),
    ]
    report = aggregate(records)
    # Mean over [0.8, 0.4] only.
    assert report.mean_context_recall == pytest.approx(0.6)


def test_aggregate_all_missing_recall_yields_none() -> None:
    records = [_record(0.5, 0.5, 0.5, None), _record(0.5, 0.5, 0.5, None)]
    report = aggregate(records)
    assert report.mean_context_recall is None


def test_report_str_shows_dash_for_missing_recall() -> None:
    report = aggregate([_record(0.5, 0.5, 0.5, None)])
    text = str(report)
    assert "Context Recall      —" in text
    assert "Faithfulness        0.500" in text


def test_aggregate_answer_correctness_averages_referenced_subset() -> None:
    records = [
        _record(answer_correctness=0.9),
        _record(answer_correctness=None),
        _record(answer_correctness=0.5),
    ]
    report = aggregate(records)
    assert report.mean_answer_correctness == pytest.approx(0.7)


def test_aggregate_all_missing_correctness_yields_none() -> None:
    records = [_record(answer_correctness=None), _record(answer_correctness=None)]
    report = aggregate(records)
    assert report.mean_answer_correctness is None


def test_aggregate_recall_and_correctness_are_independent() -> None:
    # A case can supply correctness but not recall, or vice versa — they
    # each average over their own non-None subset.
    records = [
        _record(context_recall=0.8, answer_correctness=None),
        _record(context_recall=None, answer_correctness=0.6),
        _record(context_recall=0.4, answer_correctness=0.4),
    ]
    report = aggregate(records)
    assert report.mean_context_recall == pytest.approx(0.6)        # mean of 0.8, 0.4
    assert report.mean_answer_correctness == pytest.approx(0.5)    # mean of 0.6, 0.4


def test_report_str_shows_correctness_line() -> None:
    report = aggregate([_record(answer_correctness=0.75)])
    text = str(report)
    assert "Answer Correctness  0.750" in text


def test_report_str_shows_dash_for_missing_correctness() -> None:
    report = aggregate([_record(0.1, 0.1, 0.1, None, None)])
    assert "Answer Correctness  —" in str(report)


# --- evaluate_test_set(): end-to-end with stubs ---------------------------


def _make_stubs(rec: _Recorder, generator_answer: str = "stub-answer"):
    return dict(
        prompt_store=_StubPromptStore(rec),
        conversation_store=_StubConversationStore(rec),
        query_rewriter=_StubRewriter(rec, prefix=""),
        vector_store=_StubVectorStore(rec, make_search_results(5)),
        reranker=_StubReranker(rec),
        generator=_StubGenerator(rec, answer=generator_answer),
        retrieval_evaluator=LocalRetrievalEvaluator(),
    )


async def test_evaluate_with_noop_evaluator_returns_zeroed_report() -> None:
    rec = _Recorder()
    cases = [EvalCase(query="q1"), EvalCase(query="q2", reference="gold")]
    report = await evaluate_test_set(
        **_make_stubs(rec),
        evaluator=NoOpEvaluator(),
        domain="default",
        cases=cases,
        search_top_k=4,
        final_top_k=2,
    )
    assert len(report.records) == 2
    assert report.mean_faithfulness == 0.0
    assert report.mean_answer_relevancy == 0.0
    assert report.mean_context_precision == 0.0
    # NoOpEvaluator leaves the reference-dependent metrics as None regardless
    # of whether a reference is supplied, so both means are None.
    assert report.mean_context_recall is None
    assert report.mean_answer_correctness is None


async def test_evaluate_records_carry_answer_and_context() -> None:
    rec = _Recorder()
    report = await evaluate_test_set(
        **_make_stubs(rec, generator_answer="A"),
        evaluator=NoOpEvaluator(),
        domain="default",
        cases=[EvalCase("q1"), EvalCase("q2")],
        search_top_k=10,
        final_top_k=3,
    )
    for r in report.records:
        assert r.answer == Message("assistant", "A")
        assert len(r.context) == 3  # truncated by reranker to final_top_k


async def test_evaluate_runs_query_pipeline_in_correct_order_per_case() -> None:
    rec = _Recorder()
    await evaluate_test_set(
        **_make_stubs(rec),
        evaluator=NoOpEvaluator(),
        domain="default",
        cases=[EvalCase("q1"), EvalCase("q2")],
        search_top_k=5,
        final_top_k=2,
    )
    names = [c[0] for c in rec.calls]
    # For each case: get_prompt, get_history, rewrite, search, rerank, generate.
    # No append_message — execute_query does not persist.
    expected = ["get_prompt", "get_history", "rewrite", "search", "rerank", "generate"] * 2
    assert names == expected


async def test_evaluate_uses_unique_conversation_id_per_case() -> None:
    rec = _Recorder()
    await evaluate_test_set(
        **_make_stubs(rec),
        evaluator=NoOpEvaluator(),
        domain="default",
        cases=[EvalCase("q1"), EvalCase("q2")],
    )
    history_calls = [c for c in rec.calls if c[0] == "get_history"]
    conv_ids = [c[1] for c in history_calls]
    assert conv_ids == ["__eval__:0", "__eval__:1"]


async def test_evaluate_passes_reference_through_to_evaluator() -> None:
    """A custom evaluator captures the reference; verify it receives the right one."""
    received_refs: list[str | None] = []

    class _CapturingEvaluator:
        async def evaluate(self, query, context, answer, reference=None):
            received_refs.append(reference)
            return EvaluationResult(
                faithfulness=0.5,
                answer_relevancy=0.5,
                context_precision=0.5,
                context_recall=0.5 if reference else None,
                answer_correctness=0.5 if reference else None,
            )

    rec = _Recorder()
    cases = [
        EvalCase("q1", reference="gold-1"),
        EvalCase("q2", reference=None),
        EvalCase("q3", reference="gold-3"),
    ]
    report = await evaluate_test_set(
        **_make_stubs(rec),
        evaluator=_CapturingEvaluator(),
        domain="default",
        cases=cases,
    )
    assert received_refs == ["gold-1", None, "gold-3"]
    # Two cases supply a reference; recall and correctness average over them only.
    assert report.mean_context_recall == pytest.approx(0.5)
    assert report.mean_answer_correctness == pytest.approx(0.5)


async def test_evaluate_empty_test_set_returns_empty_report() -> None:
    rec = _Recorder()
    report = await evaluate_test_set(
        **_make_stubs(rec),
        evaluator=NoOpEvaluator(),
        domain="default",
        cases=[],
    )
    assert report.records == []
    assert rec.calls == []  # nothing ran


# --- retrieval metrics -----------------------------------------------------


def _retrieval_record(retrieval=None, rerank=None) -> EvalRecord:
    return EvalRecord(
        case=EvalCase(query="q"),
        answer=Message("assistant", "a"),
        context=[],
        result=EvaluationResult(0.0, 0.0, 0.0, None, None),
        retrieval=retrieval,
        rerank=rerank,
    )


def _ret(recall: float, k: int = 10) -> RetrievalResult:
    return RetrievalResult(
        recall_at_k=recall, precision_at_k=recall, mrr=recall, ndcg=recall,
        hit_rate=1.0, k=k,
    )


def test_aggregate_retrieval_means_over_qrel_subset() -> None:
    # Two cases carry qrels (retrieval scored), one does not (None).
    records = [
        _retrieval_record(retrieval=_ret(0.8), rerank=_ret(1.0, k=5)),
        _retrieval_record(retrieval=None, rerank=None),
        _retrieval_record(retrieval=_ret(0.4), rerank=_ret(0.6, k=5)),
    ]
    report = aggregate(records)
    assert report.mean_retrieval is not None
    assert report.mean_retrieval.recall_at_k == pytest.approx(0.6)  # mean(0.8, 0.4)
    assert report.mean_retrieval.k == 10
    assert report.mean_rerank is not None
    assert report.mean_rerank.recall_at_k == pytest.approx(0.8)  # mean(1.0, 0.6)
    assert report.mean_rerank.k == 5


def test_aggregate_no_qrels_yields_none_retrieval_means() -> None:
    report = aggregate([_retrieval_record(), _retrieval_record()])
    assert report.mean_retrieval is None
    assert report.mean_rerank is None


def test_report_str_shows_dash_for_missing_retrieval() -> None:
    report = aggregate([_retrieval_record()])
    text = str(report)
    assert "Retrieval (search)" in text
    assert "Retrieval (rerank)" in text
    assert "no qrels in test set" in text


async def test_evaluate_scores_retrieval_only_for_cases_with_qrels() -> None:
    rec = _Recorder()
    # Stub vector store returns chunks from source "doc.txt"; label it relevant.
    cases = [
        EvalCase(query="q1", relevant_sources={"doc.txt"}),
        EvalCase(query="q2"),  # no qrels
    ]
    report = await evaluate_test_set(
        **_make_stubs(rec),
        evaluator=NoOpEvaluator(),
        domain="default",
        cases=cases,
        search_top_k=5,
        final_top_k=2,
        retrieval_k=5,
    )
    q1, q2 = report.records
    assert q1.retrieval is not None and q1.rerank is not None
    assert q1.retrieval.hit_rate == 1.0
    assert q1.retrieval.recall_at_k == pytest.approx(1.0)
    assert q1.rerank.k == 2
    assert q2.retrieval is None and q2.rerank is None
    # Aggregate means come only from q1.
    assert report.mean_retrieval is not None
    assert report.mean_retrieval.recall_at_k == pytest.approx(1.0)
