"""HyDE — Hypothetical Document Embeddings retrieval.

Wraps a `VectorStore`. For each query, it asks an injected `Generator` to draft a
*hypothetical answer passage*, then retrieves with that passage instead of the raw
query. A question and the document that answers it are often far apart in
embedding space; a hypothetical answer looks much more like the target document,
closing the query↔document gap and lifting recall (technique A3, see
`docs/MULTIHOP_TECHNIQUES.md`).

The inner store embeds whatever text it is handed in `search()`, so HyDE only
changes *which string* gets embedded — no `Embedder` is touched here and ADR-0005
(VectorStore owns embedding) holds by construction.

Note: `CONTEXT.md` lists "HyDE" as an *avoid* synonym under **Query Rewriter**.
HyDE is a distinct retrieval technique, not a rewriter — it produces a document to
embed, not a reformulated query — and lives here as a `VectorStore` wrapper.

Stack-agnostic: the `Generator` is injected, like `LLMQueryRewriter` and
`MultiQueryRetriever`; no backend-specific imports here.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from types import TracebackType

from rag.adapters._fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from rag.stages.generator import Generator
from rag.stages.vector_store import VectorStore
from rag.types import Chunk, SearchResult

__all__ = ["HyDERetriever"]


_DEFAULT_SYSTEM_PROMPT = (
    "Write a brief, plausible passage that directly answers the question, as if "
    "excerpted from a source document. State it factually and confidently; do not "
    "hedge, qualify, or say you are unsure. Reply with the passage only."
)

_USER_TEMPLATE = "Question: {query}\n\nWrite a short passage that answers it."


class HyDERetriever(VectorStore):
    """Retrieves on an LLM-drafted hypothetical answer instead of the raw query.

    With ``fuse_with_query=True`` it searches on *both* the raw query and the
    hypothetical passage and RRF-fuses the two — a hedge against a misleading
    hypothesis. Owns the inner store's lifecycle (mirrors `HybridVectorStore` /
    `MultiQueryRetriever`); the injected `Generator` is managed by the composition
    root.
    """

    def __init__(
        self,
        *,
        inner: VectorStore,
        generator: Generator,
        fuse_with_query: bool = False,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self._inner = inner
        self._generator = generator
        self._fuse_with_query = fuse_with_query
        self._system_prompt = system_prompt
        self._rrf_k = rrf_k
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "HyDERetriever":
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
        hypothesis = await self._generate_hypothesis(query)

        if not self._fuse_with_query:
            # Pure HyDE: embed the hypothesis. Fall back to the raw query if the
            # Generator returned nothing usable (never search an empty string).
            return await self._inner.search(hypothesis or query, top_k)

        # Fuse mode: search on the raw query and (when present) the hypothesis,
        # then RRF-merge — guards against a hypothesis that points the wrong way.
        queries = [query]
        if hypothesis:
            queries.append(hypothesis)
        result_lists = await asyncio.gather(
            *(self._inner.search(q, top_k) for q in queries)
        )
        return reciprocal_rank_fusion(result_lists, top_k=top_k, rrf_k=self._rrf_k)

    async def _generate_hypothesis(self, query: str) -> str:
        answer = await self._generator.generate(
            query=_USER_TEMPLATE.format(query=query),
            context=[],
            system_prompt=self._system_prompt,
            history=[],
        )
        return answer.content.strip()
