"""Shared types flowing through the RAG pipeline.

Defined once and imported by every stage interface and Backend Adapter — never
redefined per-stage. See CONTEXT.md for the data-flow contracts that bind these
types to specific pipeline stages.
"""

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Document",
    "Chunk",
    "EmbeddedChunk",
    "Message",
    "SearchResult",
    "EvaluationResult",
    "RetrievalResult",
]


@dataclass(frozen=True)
class Document:
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    content: str
    document_source: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    vector: list[float]


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class EvaluationResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float | None = None
    answer_correctness: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    """Rank-aware IR metrics for one ranked retrieval result scored against
    gold relevance labels (qrels). Produced by the `RetrievalEvaluator`. All
    rate metrics are in [0, 1]; `k` records the cutoff the metrics were
    computed at.
    """

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float
    hit_rate: float
    k: int
