"""BM25 (lexical) VectorStore.

A pure-Python sparse retriever over chunk text, backed by `rank_bm25`. It has no
external service dependency, so per the adapter-layering rule it lives in
`local/` alongside the other no-backend adapters.

It implements the `VectorStore` interface (`index` / `search`) without an
`Embedder` — BM25 ranks by term overlap, not vectors. Paired with a dense store
under `HybridVectorStore`, it recovers the named entities, dates, and tickers
that dense embeddings blur — the bridge terms multi-hop queries hinge on
(technique A1, see `docs/MULTIHOP_TECHNIQUES.md`).
"""

from __future__ import annotations

import re
from types import TracebackType

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from rag.stages.vector_store import VectorStore
from rag.types import Chunk, SearchResult

__all__ = ["BM25VectorStore"]

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-token split — the lexical unit BM25 scores over."""
    return _TOKEN_RE.findall(text.lower())


class BM25VectorStore(VectorStore):
    """In-memory BM25 retriever.

    `index()` has upsert semantics — chunks are keyed by
    ``"<document_source>::<position>"`` (the shared chunk-ID convention), so
    re-indexing the same chunk updates it rather than duplicating, and the BM25
    index is rebuilt from the full chunk set on each call.

    Implements the async context-manager protocol trivially (it owns no external
    resource) so it can be entered uniformly alongside resource-backed stores —
    e.g. by `HybridVectorStore`.
    """

    def __init__(self) -> None:
        self._chunks_by_id: dict[str, Chunk] = {}
        self._corpus: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    async def __aenter__(self) -> "BM25VectorStore":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for chunk in chunks:
            self._chunks_by_id[self._chunk_id(chunk)] = chunk
        self._corpus = list(self._chunks_by_id.values())
        self._tokenized = [_tokenize(c.content) for c in self._corpus]
        self._bm25 = BM25Okapi(self._tokenized)

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        if top_k <= 0 or self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        query_terms = set(tokens)
        scores = self._bm25.get_scores(tokens)
        # Keep only genuine lexical matches — chunks sharing at least one query
        # term. We test overlap rather than `score > 0` because BM25's IDF is
        # negative for terms present in every document, so a real match can
        # score below zero; RRF ranks by position, not score, so the sign is
        # immaterial to fusion.
        scored = [
            (float(score), chunk)
            for score, chunk, doc_tokens in zip(
                scores, self._corpus, self._tokenized, strict=True
            )
            if query_terms.intersection(doc_tokens)
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [SearchResult(chunk=chunk, score=score) for score, chunk in scored[:top_k]]

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        return f"{chunk.document_source}::{chunk.position}"
