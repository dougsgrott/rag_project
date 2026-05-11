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
from rag.stages.vector_store import VectorStore
from rag.types import EvaluationResult, Message, SearchResult

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


@dataclass(frozen=True)
class EvalRecord:
    case: EvalCase
    answer: Message
    context: list[SearchResult]
    result: EvaluationResult


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

    def __str__(self) -> str:
        n = len(self.records)

        def _line(value: float | None) -> str:
            if value is not None:
                return f"{value:.3f}"
            return "—   (no reference answers in test set)"

        return "\n".join(
            [
                f"Evaluation report — {n} case{'s' if n != 1 else ''}",
                f"  Faithfulness        {self.mean_faithfulness:.3f}",
                f"  Answer Relevancy    {self.mean_answer_relevancy:.3f}",
                f"  Context Precision   {self.mean_context_precision:.3f}",
                f"  Context Recall      {_line(self.mean_context_recall)}",
                f"  Answer Correctness  {_line(self.mean_answer_correctness)}",
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
    domain: str,
    cases: Iterable[EvalCase],
    search_top_k: int = 150,
    final_top_k: int = 5,
) -> EvaluationReport:
    records: list[EvalRecord] = []
    for index, case in enumerate(cases):
        conv_id = f"__eval__:{index}"
        answer, context = await execute_query(
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
        records.append(EvalRecord(case=case, answer=answer, context=context, result=result))
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
    return EvaluationReport(
        records=list(records),
        mean_faithfulness=mf,
        mean_answer_relevancy=mar,
        mean_context_precision=mcp,
        mean_context_recall=mcr,
        mean_answer_correctness=mac,
    )


def _mean_optional(values: Iterable[float | None]) -> float | None:
    """Mean over the subset of values that are not None. Returns None if empty."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
