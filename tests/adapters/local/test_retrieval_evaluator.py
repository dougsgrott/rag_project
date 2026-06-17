"""Metric-math tests for LocalRetrievalEvaluator against hand-computed values."""

import math

import pytest

from rag.adapters.local.retrieval_evaluator import LocalRetrievalEvaluator
from rag.types import Chunk, SearchResult

from tests.stages._helpers import make_search_results
from tests.stages.retrieval_evaluator_conformance import RetrievalEvaluatorConformance


def _results(*specs: tuple[str, int]) -> list[SearchResult]:
    """Build a ranked list from (source, position) pairs, best-first."""
    return [
        SearchResult(
            chunk=Chunk(content=f"{src}-{pos}", document_source=src, position=pos),
            score=1.0 - 0.01 * i,
        )
        for i, (src, pos) in enumerate(specs)
    ]


class TestLocalRetrievalEvaluatorConformance(RetrievalEvaluatorConformance):
    @pytest.fixture
    def retrieval_evaluator(self) -> LocalRetrievalEvaluator:
        return LocalRetrievalEvaluator()


class TestLocalRetrievalEvaluatorMath:
    async def test_perfect_ranking_scores_one(self) -> None:
        ev = LocalRetrievalEvaluator()
        ranked = make_search_results(3)  # ids doc.txt::0,1,2
        result = await ev.evaluate(
            ranked, None, {"doc.txt::0", "doc.txt::1", "doc.txt::2"}, k=3
        )
        assert result.recall_at_k == pytest.approx(1.0)
        assert result.precision_at_k == pytest.approx(1.0)
        assert result.mrr == pytest.approx(1.0)
        assert result.ndcg == pytest.approx(1.0)
        assert result.hit_rate == 1.0

    async def test_partial_recall_missing_chunk_not_retrieved(self) -> None:
        ev = LocalRetrievalEvaluator()
        ranked = make_search_results(3)  # doc.txt::0,1,2
        # Two gold chunks; only ::0 is retrieved.
        result = await ev.evaluate(ranked, None, {"doc.txt::0", "doc.txt::9"}, k=3)
        assert result.recall_at_k == pytest.approx(0.5)  # 1 of 2 relevant found
        assert result.precision_at_k == pytest.approx(1 / 3)
        assert result.mrr == pytest.approx(1.0)  # first result is relevant
        assert result.hit_rate == 1.0
        # DCG = 1/log2(2)=1 ; IDCG over min(2,3)=2 ideal positions.
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert result.ndcg == pytest.approx(1.0 / idcg)

    async def test_no_hits_scores_zero(self) -> None:
        ev = LocalRetrievalEvaluator()
        result = await ev.evaluate(make_search_results(3), None, {"other.txt::0"}, k=3)
        assert result.recall_at_k == 0.0
        assert result.precision_at_k == 0.0
        assert result.mrr == 0.0
        assert result.ndcg == 0.0
        assert result.hit_rate == 0.0

    async def test_first_relevant_lower_in_ranking_lowers_mrr_and_ndcg(self) -> None:
        ev = LocalRetrievalEvaluator()
        ranked = make_search_results(3)  # only ::2 is relevant -> rank 3
        result = await ev.evaluate(ranked, None, {"doc.txt::2"}, k=3)
        assert result.mrr == pytest.approx(1 / 3)
        assert result.recall_at_k == pytest.approx(1.0)
        assert result.precision_at_k == pytest.approx(1 / 3)
        # DCG = 1/log2(4)=0.5 ; IDCG over 1 ideal position = 1.
        assert result.ndcg == pytest.approx(0.5)

    async def test_document_level_qrels_collapse_to_one_target(self) -> None:
        ev = LocalRetrievalEvaluator()
        ranked = make_search_results(3)  # all from doc.txt
        result = await ev.evaluate(ranked, {"doc.txt"}, None, k=3)
        # One relevant *document*; all three chunks count as relevant positions
        # (precision) but recall is over the single document target.
        assert result.recall_at_k == pytest.approx(1.0)
        assert result.precision_at_k == pytest.approx(1.0)
        assert result.hit_rate == 1.0
        assert result.ndcg == pytest.approx(1.0)

    async def test_mixed_document_and_chunk_level_qrels(self) -> None:
        ev = LocalRetrievalEvaluator()
        ranked = _results(("a.txt", 0), ("b.txt", 0), ("c.txt", 0))
        result = await ev.evaluate(
            ranked, relevant_sources={"a.txt"}, relevant_chunks={"b.txt::0"}, k=3
        )
        # Two distinct gold targets (source a.txt, chunk b.txt::0), both hit.
        assert result.recall_at_k == pytest.approx(1.0)
        assert result.precision_at_k == pytest.approx(2 / 3)
        assert result.mrr == pytest.approx(1.0)
        assert result.hit_rate == 1.0
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        dcg = 1 / math.log2(2) + 1 / math.log2(3)  # both at ranks 1 and 2
        assert result.ndcg == pytest.approx(dcg / idcg)

    async def test_k_larger_than_result_list(self) -> None:
        ev = LocalRetrievalEvaluator()
        ranked = make_search_results(2)  # doc.txt::0,1
        result = await ev.evaluate(ranked, None, {"doc.txt::0", "doc.txt::1"}, k=10)
        assert result.recall_at_k == pytest.approx(1.0)
        # Precision divides by results examined (2), not by k.
        assert result.precision_at_k == pytest.approx(1.0)
        assert result.k == 10

    async def test_k_truncates_ranking(self) -> None:
        ev = LocalRetrievalEvaluator()
        ranked = make_search_results(5)  # doc.txt::0..4
        # Only ::3 is relevant, but k=2 cuts the ranking before it.
        result = await ev.evaluate(ranked, None, {"doc.txt::3"}, k=2)
        assert result.recall_at_k == 0.0
        assert result.hit_rate == 0.0
        assert result.mrr == 0.0
