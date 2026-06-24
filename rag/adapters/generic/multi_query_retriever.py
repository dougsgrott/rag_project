"""Multi-query expansion retriever.

Wraps a `VectorStore` and, for each query, asks an injected `Generator` for
several alternative phrasings / sub-facets of the query, searches the inner store
for each, and fuses the results with Reciprocal Rank Fusion. A single embedding
of a multi-hop query covers only part of its evidence set; expanding into several
sub-queries and fusing pulls more of that set into one wide net in a single shot
(technique A2, see `docs/MULTIHOP_TECHNIQUES.md`). Best on `comparison_query`,
where the sub-facts are independently retrievable.

Why a `VectorStore` wrapper and not a `QueryRewriter`: `QueryRewriter.rewrite()`
returns a single `str`, but expansion needs `list[str]`. Wrapping the store keeps
the expansion internal and the query/evaluation pipelines unchanged — they call
`search(query, top_k)` exactly as before.

Stack-agnostic: the `Generator` is injected, exactly like `LLMQueryRewriter` and
`LLMContextEnricher`, so no backend-specific imports live here.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from types import TracebackType

from rag.adapters._fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag.stages.generator import Generator
from rag.stages.vector_store import VectorStore
from rag.types import Chunk, SearchResult

__all__ = ["MultiQueryRetriever"]


_DEFAULT_SYSTEM_PROMPT = (
    "You expand a search query into alternative queries that capture different "
    "phrasings and sub-aspects of the user's information need, to improve "
    "retrieval recall. Output only the alternative queries, one per line, with "
    "no numbering, preamble, or explanation."
)

_USER_TEMPLATE = """Original query: {query}

Write {n} alternative search queries that rephrase or decompose the original so \
that, together, they retrieve all the documents needed to answer it. One query \
per line, no numbering."""

# Strips a leading list marker ("1. ", "- ", "* ", "• ") a model may add despite
# the instruction not to number.
_LIST_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")


class MultiQueryRetriever(VectorStore):
    """Expands a query into several and fuses the per-query results with RRF.

    Owns the inner store's lifecycle (enters/exits it if it manages a resource),
    mirroring `HybridVectorStore`. The injected `Generator` is *not* entered here —
    it is constructed and managed by the composition root.
    """

    def __init__(
        self,
        *,
        inner: VectorStore,
        generator: Generator,
        n_queries: int = 4,
        include_original: bool = True,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if n_queries < 1:
            raise ValueError("n_queries must be >= 1")
        self._inner = inner
        self._generator = generator
        self._n_queries = n_queries
        self._include_original = include_original
        self._system_prompt = system_prompt
        self._rrf_k = rrf_k
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "MultiQueryRetriever":
        stack = AsyncExitStack()
        await stack.__aenter__()
        if isinstance(self._inner, AbstractAsyncContextManager):
            await stack.enter_async_context(self._inner)
        self._stack = stack
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
            self._stack = None

    async def index(self, chunks: list[Chunk]) -> None:
        await self._inner.index(chunks)

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []
        queries = await self._build_queries(query)
        result_lists = await asyncio.gather(
            *(self._inner.search(q, top_k) for q in queries)
        )
        return reciprocal_rank_fusion(result_lists, top_k=top_k, rrf_k=self._rrf_k)

    async def _build_queries(self, query: str) -> list[str]:
        """Original (optionally) plus the LLM-generated variants, de-duplicated.

        Falls back to ``[query]`` when the Generator yields nothing usable, so the
        inner store is never asked to search an empty string.
        """
        variants = self._parse_variants(await self._generate_variants(query))
        queries: list[str] = []
        seen: set[str] = set()
        if self._include_original:
            queries.append(query)
            seen.add(query.strip().lower())
        for variant in variants:
            key = variant.lower()
            if key not in seen:
                seen.add(key)
                queries.append(variant)
        return queries or [query]

    async def _generate_variants(self, query: str) -> str:
        answer = await self._generator.generate(
            query=_USER_TEMPLATE.format(query=query, n=self._n_queries),
            context=[],
            system_prompt=self._system_prompt,
            history=[],
        )
        return answer.content

    def _parse_variants(self, text: str) -> list[str]:
        variants: list[str] = []
        for line in text.splitlines():
            cleaned = _LIST_PREFIX.sub("", line).strip()
            if cleaned:
                variants.append(cleaned)
            if len(variants) >= self._n_queries:
                break
        return variants
