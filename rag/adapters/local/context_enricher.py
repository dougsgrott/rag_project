from rag.stages.context_enricher import ContextEnricher
from rag.types import Chunk, Document

__all__ = ["NoOpContextEnricher"]


class NoOpContextEnricher(ContextEnricher):
    """Passthrough: returns the input chunks unchanged."""

    async def enrich(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        return list(chunks)
