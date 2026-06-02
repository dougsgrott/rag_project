"""LLM-based Context Enricher (Anthropic's Contextual Retrieval).

For each chunk, asks an injected `Generator` to produce a short summary that
situates the chunk within the parent document. The summary is prepended to
the chunk's content before indexing, so a chunk that would otherwise read
"This is a known issue and is being investigated." carries enough context
("Section: Error code E-217 in the SICRO manual...") to be retrievable.

Stack-agnostic: the `Generator` is injected. The simple stack supplies
`OpenAIGenerator`; the Cortex stack supplies `CortexGenerator`. No
backend-specific imports here.
"""

from __future__ import annotations

import asyncio

from rag.stages.context_enricher import ContextEnricher
from rag.stages.generator import Generator
from rag.types import Chunk, Document

__all__ = ["LLMContextEnricher"]


_DEFAULT_SYSTEM_PROMPT = (
    "You situate document chunks within their parent document for retrieval. "
    "Given a document and one of its chunks, respond with a short, succinct "
    "context that locates the chunk within the overall document. "
    "Reply with the context only, no preamble."
)


_USER_TEMPLATE = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:

<chunk>
{chunk}
</chunk>

Please give a short, succinct context to situate this chunk within the overall \
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""


class LLMContextEnricher(ContextEnricher):
    """Contextual Retrieval enricher driven by an injected `Generator`.

    Concurrency is bounded by a semaphore — `max_concurrency` in-flight
    generator calls at a time. Default 5; set to 1 for strictly sequential.

    Cost note: each chunk's prompt embeds the *full* document. For a doc
    with N chunks, the Generator sees N × document_len tokens. Real
    deployments should use a Generator adapter that supports prompt
    caching (e.g. Anthropic's cache-control) — this reference enricher
    just calls `generate()` and trusts the adapter's policy.
    """

    def __init__(
        self,
        *,
        generator: Generator,
        max_concurrency: int = 5,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self._generator = generator
        self._system_prompt = system_prompt
        self._max_concurrency = max_concurrency

    async def enrich(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        semaphore = asyncio.Semaphore(self._max_concurrency)
        tasks = [self._enrich_one(semaphore, document, chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)

    async def _enrich_one(
        self,
        semaphore: asyncio.Semaphore,
        document: Document,
        chunk: Chunk,
    ) -> Chunk:
        async with semaphore:
            answer = await self._generator.generate(
                query=_USER_TEMPLATE.format(
                    document=document.content,
                    chunk=chunk.content,
                ),
                context=[],
                system_prompt=self._system_prompt,
                history=[],
            )
        summary = answer.content.strip()
        if not summary:
            return chunk
        return Chunk(
            content=f"{summary}\n\n{chunk.content}",
            document_source=chunk.document_source,
            position=chunk.position,
            metadata={**chunk.metadata, "contextual_summary": summary},
        )
