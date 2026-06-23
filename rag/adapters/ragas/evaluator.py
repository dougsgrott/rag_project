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
import inspect
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
        # RAGAS LLM/embeddings wrappers, built once on first use and reused
        # across calls. The raw langchain objects are kept so their OpenAI
        # clients can be closed on exit.
        self._llm: Any = None
        self._embeddings: Any = None
        self._chat: Any = None
        self._raw_embeddings: Any = None

    async def __aenter__(self) -> "RagasEvaluator":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._aclose_backends()

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
            from ragas import evaluate as ragas_evaluate
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

        llm, embeddings = self._ensure_backends()

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

    def _ensure_backends(self) -> tuple[Any, Any]:
        """Build and cache the RAGAS LLM + embeddings wrappers.

        Built once and reused across calls: constructing a fresh
        ChatOpenAI/OpenAIEmbeddings per `evaluate()` spins up a new pair of
        OpenAI HTTP clients each time, wasting connections and leaving clients
        to be finalised after the event loop closes (the "Event loop is closed"
        noise). The raw langchain objects are retained for teardown.
        """
        if self._llm is not None and self._embeddings is not None:
            return self._llm, self._embeddings
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from pydantic import SecretStr
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
        except ImportError as e:  # pragma: no cover - exercised only without the group
            raise ConfigurationError(
                "RagasEvaluator requires the 'ragas' dependency group "
                "(run: uv sync --group ragas). Underlying import error: " + str(e)
            ) from e

        api_key = SecretStr(self._api_key)
        self._chat = ChatOpenAI(
            model=self._llm_model, api_key=api_key, temperature=0.0
        )
        self._raw_embeddings = OpenAIEmbeddings(
            model=self._embedding_model, api_key=api_key
        )
        self._llm = LangchainLLMWrapper(self._chat)
        self._embeddings = LangchainEmbeddingsWrapper(self._raw_embeddings)
        return self._llm, self._embeddings

    async def _aclose_backends(self) -> None:
        """Close the OpenAI HTTP clients the langchain backends opened.

        Runs from `__aexit__`, while the event loop is still open, so the
        async clients are torn down cleanly instead of being finalised against
        a closed loop. Best-effort and defensive across langchain/openai
        versions; a no-op when the backends were never built (e.g. mocked).
        """
        # pragma: no cover - real clients only exist on the live RAGAS path.
        clients: list[Any] = []
        for obj in (self._chat, self._raw_embeddings):
            if obj is None:
                continue
            clients.append(getattr(obj, "root_async_client", None))
            clients.append(getattr(obj, "root_client", None))
            for attr in ("async_client", "client"):
                resource = getattr(obj, attr, None)
                clients.append(getattr(resource, "_client", None))

        for client in clients:
            closer = getattr(client, "close", None)
            if closer is None:
                continue
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass

        self._llm = self._embeddings = self._chat = self._raw_embeddings = None


def _score(scores: dict[str, float], name: str) -> float:
    """Read a metric, coercing a missing or non-finite (NaN) value to 0.0.

    RAGAS occasionally emits NaN when the LLM judge fails to produce a parsable
    verdict; a measured-but-unparsable score is treated as 0.0 rather than
    propagating NaN into the aggregate report.
    """
    value = scores.get(name, 0.0)
    return value if math.isfinite(value) else 0.0
