"""RAGAS-backed Evaluator adapter.

Scores a single `(query, context, answer, reference?)` tuple with the RAGAS
framework. Three metrics are reference-free and always computed — Faithfulness,
Answer Relevancy, Context Precision; two require the gold reference answer —
Context Recall and Answer Correctness — and are set on the returned
`EvaluationResult` only when a reference is supplied (otherwise left `None`,
honouring the `Evaluator` contract).

RAGAS is a synchronous, LLM-backed library, so the blocking call is run in a
worker thread (`asyncio.to_thread`) to avoid stalling the event loop. The heavy
`ragas`/`datasets`/`langchain-openai` imports are deferred to first use (inside
`_run_ragas`) so the module — and the unit/conformance suites that mock the
backend — do not depend on the `ragas` group being installed. Install it with
`uv sync --group ragas` to exercise the real path.
"""

from __future__ import annotations

import asyncio
import math
from types import TracebackType
from typing import Any

from rag.errors import BackendCommunicationError, ConfigurationError
from rag.stages.evaluator import Evaluator
from rag.types import EvaluationResult, Message, SearchResult

__all__ = ["RagasEvaluator"]

# RAGAS metric names — these double as the `ragas.metrics` instance names and
# the column labels in the result frame, so they are the single source of truth
# for the mapping between RAGAS output and `EvaluationResult` fields.
_FAITHFULNESS = "faithfulness"
_ANSWER_RELEVANCY = "answer_relevancy"
_CONTEXT_PRECISION = "context_precision"
_CONTEXT_RECALL = "context_recall"
_ANSWER_CORRECTNESS = "answer_correctness"

# Always computed (reference-free) vs. computed only when a reference is given.
_REFERENCE_FREE = (_FAITHFULNESS, _ANSWER_RELEVANCY, _CONTEXT_PRECISION)
_REFERENCE_DEPENDENT = (_CONTEXT_RECALL, _ANSWER_CORRECTNESS)


class RagasEvaluator(Evaluator):
    """Evaluator backed by the RAGAS framework.

    Use as an async context manager for lifecycle symmetry with the other
    Backend Adapters (RAGAS holds no persistent client of its own, so enter/
    exit are no-ops). The OpenAI credentials drive the LLM judge and the
    embedding model RAGAS uses internally.
    """

    def __init__(
        self,
        *,
        api_key: str,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        if not api_key:
            raise ConfigurationError("RagasEvaluator requires a non-empty api_key")
        self._api_key = api_key
        self._llm_model = llm_model
        self._embedding_model = embedding_model

    async def __aenter__(self) -> "RagasEvaluator":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def evaluate(
        self,
        query: str,
        context: list[SearchResult],
        answer: Message,
        reference: str | None = None,
    ) -> EvaluationResult:
        contexts = [r.chunk.content for r in context]
        metric_names = list(_REFERENCE_FREE)
        if reference is not None:
            metric_names += list(_REFERENCE_DEPENDENT)

        scores = await asyncio.to_thread(
            self._run_ragas,
            query=query,
            answer=answer.content,
            contexts=contexts,
            reference=reference,
            metric_names=metric_names,
        )

        return EvaluationResult(
            faithfulness=_score(scores, _FAITHFULNESS),
            answer_relevancy=_score(scores, _ANSWER_RELEVANCY),
            context_precision=_score(scores, _CONTEXT_PRECISION),
            # Reference-dependent metrics stay None when no reference is given,
            # regardless of what the backend returned.
            context_recall=_score(scores, _CONTEXT_RECALL) if reference is not None else None,
            answer_correctness=(
                _score(scores, _ANSWER_CORRECTNESS) if reference is not None else None
            ),
        )

    def _run_ragas(
        self,
        *,
        query: str,
        answer: str,
        contexts: list[str],
        reference: str | None,
        metric_names: list[str],
    ) -> dict[str, float]:
        """Run RAGAS for the given metrics; return {metric_name: score}.

        This is the single seam that touches the RAGAS framework. It runs in a
        worker thread (see `evaluate`) and is patched out wholesale by the unit
        and conformance suites — only the gated integration test exercises the
        real framework.
        """
        try:
            from datasets import Dataset  # type: ignore[import-untyped]
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from pydantic import SecretStr
            from ragas import evaluate as ragas_evaluate
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import (
                answer_correctness,
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as e:  # pragma: no cover - exercised only without the group
            raise ConfigurationError(
                "RagasEvaluator requires the 'ragas' dependency group "
                "(run: uv sync --group ragas). Underlying import error: " + str(e)
            ) from e

        by_name = {
            _FAITHFULNESS: faithfulness,
            _ANSWER_RELEVANCY: answer_relevancy,
            _CONTEXT_PRECISION: context_precision,
            _CONTEXT_RECALL: context_recall,
            _ANSWER_CORRECTNESS: answer_correctness,
        }
        metrics = [by_name[name] for name in metric_names]

        data: dict[str, list[object]] = {
            "question": [query],
            "answer": [answer],
            "contexts": [list(contexts)],
        }
        if reference is not None:
            data["ground_truth"] = [reference]
        dataset = Dataset.from_dict(data)

        api_key = SecretStr(self._api_key)
        llm = LangchainLLMWrapper(
            ChatOpenAI(model=self._llm_model, api_key=api_key, temperature=0.0)
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(model=self._embedding_model, api_key=api_key)
        )

        try:
            result: Any = ragas_evaluate(
                dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings
            )
        except Exception as e:  # noqa: BLE001 - normalise any backend failure
            raise BackendCommunicationError(f"RAGAS evaluation failed: {e}") from e

        row = result.to_pandas().iloc[0]
        scores: dict[str, float] = {}
        for name, metric in zip(metric_names, metrics, strict=True):
            column = getattr(metric, "name", name)
            value = row[column] if column in row else row[name]
            scores[name] = float(value)
        return scores


def _score(scores: dict[str, float], name: str) -> float:
    """Read a metric, coercing a missing or non-finite (NaN) value to 0.0.

    RAGAS occasionally emits NaN when the LLM judge fails to produce a parsable
    verdict; a measured-but-unparsable score is treated as 0.0 rather than
    propagating NaN into the aggregate report.
    """
    value = scores.get(name, 0.0)
    return value if math.isfinite(value) else 0.0
