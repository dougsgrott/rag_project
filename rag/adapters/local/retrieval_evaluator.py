"""Local, dependency-free retrieval evaluators.

`LocalRetrievalEvaluator` computes rank-aware IR metrics in pure Python — no
external service or model — so it is cheap and deterministic enough to be the
default retrieval scorer. `NoOpRetrievalEvaluator` is the passthrough for
stacks where retrieval scoring is not wanted (e.g. Cortex, which ranks
internally).

Relevance is binary, decided per retrieved `SearchResult` by the gold sets
(see `_match`). Metrics over the top-`k` slice of the ranking:

- **recall@k**   — distinct gold targets hit / total gold targets.
- **precision@k**— relevant results / results examined (`min(k, len(ranked))`).
- **MRR**        — reciprocal rank of the first relevant result.
- **nDCG**       — binary-gain DCG over distinct-target hits, normalised by the
  ideal DCG; de-duplicating per gold target keeps the score in [0, 1] even when
  a single relevant *document* is split across many retrieved chunks.
- **hit rate**   — 1.0 if any relevant result appears, else 0.0.
"""

from __future__ import annotations

import math

from rag.stages.retrieval_evaluator import RetrievalEvaluator
from rag.types import RetrievalResult, SearchResult

__all__ = ["LocalRetrievalEvaluator", "NoOpRetrievalEvaluator"]


def _chunk_id(result: SearchResult) -> str:
    # Mirrors the chunk-ID convention used by the vector-store adapters.
    return f"{result.chunk.document_source}::{result.chunk.position}"


def _match(
    result: SearchResult,
    relevant_sources: set[str],
    relevant_chunks: set[str],
) -> tuple[str, str] | None:
    """Return the gold target a result satisfies, or None if irrelevant.

    A target is identified so multiple chunks of one relevant document collapse
    to a single target. Chunk-level labels take precedence over document-level.
    """
    if _chunk_id(result) in relevant_chunks:
        return ("chunk", _chunk_id(result))
    if result.chunk.document_source in relevant_sources:
        return ("source", result.chunk.document_source)
    return None


class LocalRetrievalEvaluator(RetrievalEvaluator):
    async def evaluate(
        self,
        ranked: list[SearchResult],
        relevant_sources: set[str] | None,
        relevant_chunks: set[str] | None,
        k: int,
    ) -> RetrievalResult:
        sources = relevant_sources or set()
        chunks = relevant_chunks or set()
        total_relevant = len(sources) + len(chunks)
        top = ranked[:k]

        flags: list[bool] = []  # raw per-position relevance
        gains: list[int] = []  # 1 only on the first hit of each distinct target
        seen: set[tuple[str, str]] = set()
        for result in top:
            target = _match(result, sources, chunks)
            flags.append(target is not None)
            if target is not None and target not in seen:
                seen.add(target)
                gains.append(1)
            else:
                gains.append(0)

        num_hits = sum(flags)
        distinct_hits = sum(gains)

        recall = distinct_hits / total_relevant if total_relevant else 0.0
        precision = num_hits / len(top) if top else 0.0
        mrr = next((1.0 / (i + 1) for i, hit in enumerate(flags) if hit), 0.0)
        hit_rate = 1.0 if num_hits else 0.0

        dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
        ideal_n = min(total_relevant, len(top))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
        ndcg = dcg / idcg if idcg else 0.0

        return RetrievalResult(
            recall_at_k=recall,
            precision_at_k=precision,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate=hit_rate,
            k=k,
        )


class NoOpRetrievalEvaluator(RetrievalEvaluator):
    """Passthrough: returns a zeroed `RetrievalResult` without scoring."""

    async def evaluate(
        self,
        ranked: list[SearchResult],
        relevant_sources: set[str] | None,
        relevant_chunks: set[str] | None,
        k: int,
    ) -> RetrievalResult:
        return RetrievalResult(
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg=0.0,
            hit_rate=0.0,
            k=k,
        )
