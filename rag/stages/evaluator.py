from abc import ABC, abstractmethod

from rag.types import EvaluationResult, Message, SearchResult

__all__ = ["Evaluator"]


class Evaluator(ABC):
    """Measures RAG output quality for a single (query, context, answer) tuple.

    Run offline by `pipeline/evaluate.py` — never on the live query path.
    `reference` is the optional gold answer; `context_recall` in the returned
    `EvaluationResult` is set only when a reference is provided.
    """

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        context: list[SearchResult],
        answer: Message,
        reference: str | None = None,
    ) -> EvaluationResult:
        ...
