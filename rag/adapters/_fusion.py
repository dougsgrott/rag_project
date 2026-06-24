"""Reciprocal Rank Fusion (RRF) — a rank-based result merge.

Shared by retrievers that combine several ranked `SearchResult` lists into one:
`HybridVectorStore` (dense + BM25) and, later, the multi-query retriever (#016).
RRF fuses by *rank*, not score, so it sidesteps the incomparable-scale problem of
mixing cosine similarity with BM25 scores.

The fused `SearchResult.score` is the RRF score (`Σ 1/(rrf_k + rank)`), **not** a
similarity — downstream consumers (the reranker, IR metrics) treat it as an
opaque ranking signal.
"""

from __future__ import annotations

from typing import Sequence

from rag.types import Chunk, SearchResult

__all__ = ["reciprocal_rank_fusion", "DEFAULT_RRF_K"]

# Conventional RRF constant (Cormack et al., 2009). Dampens the weight of top
# ranks so a single list can't dominate the fusion.
DEFAULT_RRF_K = 60


def _chunk_id(chunk: Chunk) -> str:
    """Stable identity for a chunk across retrievers.

    Mirrors `ChromaVectorStore._chunk_id` so a chunk found by both the dense and
    the sparse store collapses to one fused entry.
    """
    return f"{chunk.document_source}::{chunk.position}"


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[SearchResult]],
    *,
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SearchResult]:
    """Fuse several ranked lists into one top-`top_k` list by RRF.

    Each list's item at 0-based `rank` contributes `1 / (rrf_k + rank + 1)` to its
    chunk's fused score; contributions sum across lists, so a chunk surfaced by
    several retrievers outranks one surfaced by a single retriever. Ties break by
    first-seen order (stable), keeping the output deterministic.
    """
    if top_k <= 0:
        return []

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    order: list[str] = []  # first-seen order → deterministic tie-breaking
    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            cid = _chunk_id(result.chunk)
            if cid not in chunks:
                chunks[cid] = result.chunk
                order.append(cid)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked_ids = sorted(order, key=lambda cid: scores[cid], reverse=True)
    return [SearchResult(chunk=chunks[cid], score=scores[cid]) for cid in ranked_ids[:top_k]]
