from abc import ABC, abstractmethod

from rag.types import RetrievalResult, SearchResult

__all__ = ["RetrievalEvaluator"]


class RetrievalEvaluator(ABC):
    """Scores a ranked retrieval result as an IR system against gold labels.

    Deterministic and qrels-based — a distinct concern from the LLM-judged
    generation `Evaluator`. Run offline by `pipeline/evaluate.py` (never on the
    live query path), once per retrieval stage: the wide vector-search
    candidate set and, separately, the post-rerank context.

    A `SearchResult` counts as relevant when its `"<document_source>::<position>"`
    chunk ID is in `relevant_chunks`, or its `document_source` is in
    `relevant_sources`. Callers pass the qrels for the case; a case that
    supplies neither set has no qrels and should not be scored at all (the
    metrics are undefined without a gold set).
    """

    @abstractmethod
    async def evaluate(
        self,
        ranked: list[SearchResult],
        relevant_sources: set[str] | None,
        relevant_chunks: set[str] | None,
        k: int,
    ) -> RetrievalResult:
        ...
