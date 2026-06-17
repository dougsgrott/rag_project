"""Offline evaluation pipeline.

Runs the live query pipeline against a test set of `(query, reference?)`
pairs, then asks an injected `Evaluator` to score each output. Aggregates
per-case `EvaluationResult`s into a summary report.

Reuses `execute_query` from `query.py` so adapter wiring and pipeline shape
stay in lockstep with the live path. Each case runs under a unique synthetic
`conversation_id` (`__eval__:<n>`) so it sees an empty history and the test
set never pollutes real conversations.
"""

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from rag.pipeline.query import execute_query
from rag.stages.conversation_store import ConversationStore
from rag.stages.evaluator import Evaluator
from rag.stages.generator import Generator
from rag.stages.prompt_store import PromptStore
from rag.stages.query_rewriter import QueryRewriter
from rag.stages.reranker import Reranker
from rag.stages.retrieval_evaluator import RetrievalEvaluator
from rag.stages.vector_store import VectorStore
from rag.types import EvaluationResult, Message, RetrievalResult, SearchResult

__all__ = [
    "EvalCase",
    "EvalRecord",
    "EvaluationReport",
    "evaluate_test_set",
    "aggregate",
]


@dataclass(frozen=True)
class EvalCase:
    query: str
    reference: str | None = None
    # Gold relevance labels (qrels) for retrieval scoring. A case with neither
    # set is not scored for retrieval — its retrieval metrics stay None.
    relevant_sources: set[str] | None = None
    relevant_chunks: set[str] | None = None

    @property
    def has_qrels(self) -> bool:
        return bool(self.relevant_sources or self.relevant_chunks)


@dataclass(frozen=True)
class EvalRecord:
    case: EvalCase
    answer: Message
    context: list[SearchResult]
    result: EvaluationResult
    # Retrieval metrics for the two IR stages — None when the case has no qrels.
    retrieval: RetrievalResult | None = None  # scored on the wide candidate set
    rerank: RetrievalResult | None = None  # scored on the reranked context


@dataclass(frozen=True)
class EvaluationReport:
    records: list[EvalRecord] = field(default_factory=list)
    mean_faithfulness: float = 0.0
    mean_answer_relevancy: float = 0.0
    mean_context_precision: float = 0.0
    # Reference-dependent metrics: averaged only over the subset of cases
    # whose evaluator returned a non-None value. None when no case did.
    mean_context_recall: float | None = None
    mean_answer_correctness: float | None = None
    # Retrieval metrics, averaged only over cases that supplied qrels. None
    # when no case did. `mean_retrieval` scores the wide candidate set;
    # `mean_rerank` scores the reranked context.
    mean_retrieval: RetrievalResult | None = None
    mean_rerank: RetrievalResult | None = None

    def __str__(self) -> str:
        n = len(self.records)

        def _line(value: float | None) -> str:
            if value is not None:
                return f"{value:.3f}"
            return "—   (no reference answers in test set)"

        def _retrieval_line(label: str, r: RetrievalResult | None) -> str:
            if r is None:
                return f"  {label:<22}—   (no qrels in test set)"
            return (
                f"  {label:<22}"
                f"R@{r.k}={r.recall_at_k:.3f} P@{r.k}={r.precision_at_k:.3f} "
                f"MRR={r.mrr:.3f} nDCG={r.ndcg:.3f} Hit={r.hit_rate:.3f}"
            )

        return "\n".join(
            [
                f"Evaluation report — {n} case{'s' if n != 1 else ''}",
                f"  Faithfulness        {self.mean_faithfulness:.3f}",
                f"  Answer Relevancy    {self.mean_answer_relevancy:.3f}",
                f"  Context Precision   {self.mean_context_precision:.3f}",
                f"  Context Recall      {_line(self.mean_context_recall)}",
                f"  Answer Correctness  {_line(self.mean_answer_correctness)}",
                _retrieval_line("Retrieval (search)", self.mean_retrieval),
                _retrieval_line("Retrieval (rerank)", self.mean_rerank),
            ]
        )


async def evaluate_test_set(
    *,
    prompt_store: PromptStore,
    conversation_store: ConversationStore,
    query_rewriter: QueryRewriter,
    vector_store: VectorStore,
    reranker: Reranker,
    generator: Generator,
    evaluator: Evaluator,
    retrieval_evaluator: RetrievalEvaluator,
    domain: str,
    cases: Iterable[EvalCase],
    search_top_k: int = 150,
    final_top_k: int = 5,
    retrieval_k: int = 10,
) -> EvaluationReport:
    records: list[EvalRecord] = []
    for index, case in enumerate(cases):
        conv_id = f"__eval__:{index}"
        answer, candidates, context = await execute_query(
            prompt_store=prompt_store,
            conversation_store=conversation_store,
            query_rewriter=query_rewriter,
            vector_store=vector_store,
            reranker=reranker,
            generator=generator,
            conversation_id=conv_id,
            domain=domain,
            query=case.query,
            search_top_k=search_top_k,
            final_top_k=final_top_k,
        )
        result = await evaluator.evaluate(case.query, context, answer, case.reference)

        retrieval: RetrievalResult | None = None
        rerank: RetrievalResult | None = None
        if case.has_qrels:
            # Score the retriever (wide candidate set, at retrieval_k) and the
            # reranker (final context, at final_top_k) as separate IR stages.
            retrieval = await retrieval_evaluator.evaluate(
                candidates, case.relevant_sources, case.relevant_chunks, retrieval_k
            )
            rerank = await retrieval_evaluator.evaluate(
                context, case.relevant_sources, case.relevant_chunks, final_top_k
            )

        records.append(
            EvalRecord(
                case=case,
                answer=answer,
                context=context,
                result=result,
                retrieval=retrieval,
                rerank=rerank,
            )
        )
    return aggregate(records)


def aggregate(records: Sequence[EvalRecord]) -> EvaluationReport:
    if not records:
        return EvaluationReport(records=[])
    n = len(records)
    mf = sum(r.result.faithfulness for r in records) / n
    mar = sum(r.result.answer_relevancy for r in records) / n
    mcp = sum(r.result.context_precision for r in records) / n
    mcr = _mean_optional(r.result.context_recall for r in records)
    mac = _mean_optional(r.result.answer_correctness for r in records)
    mret = _mean_retrieval(r.retrieval for r in records)
    mrer = _mean_retrieval(r.rerank for r in records)
    return EvaluationReport(
        records=list(records),
        mean_faithfulness=mf,
        mean_answer_relevancy=mar,
        mean_context_precision=mcp,
        mean_context_recall=mcr,
        mean_answer_correctness=mac,
        mean_retrieval=mret,
        mean_rerank=mrer,
    )


def _mean_optional(values: Iterable[float | None]) -> float | None:
    """Mean over the subset of values that are not None. Returns None if empty."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _mean_retrieval(
    values: Iterable[RetrievalResult | None],
) -> RetrievalResult | None:
    """Per-field mean over the non-None RetrievalResults. None if none present.

    `k` is taken from the first present result — all results for a given stage
    are scored at the same cutoff.
    """
    present = [v for v in values if v is not None]
    if not present:
        return None
    n = len(present)
    return RetrievalResult(
        recall_at_k=sum(r.recall_at_k for r in present) / n,
        precision_at_k=sum(r.precision_at_k for r in present) / n,
        mrr=sum(r.mrr for r in present) / n,
        ndcg=sum(r.ndcg for r in present) / n,
        hit_rate=sum(r.hit_rate for r in present) / n,
        k=present[0].k,
    )
