from abc import ABC, abstractmethod

from rag.types import Chunk, Document

__all__ = ["ContextEnricher"]


class ContextEnricher(ABC):
    """Enriches `Chunk`s with parent-document context before indexing.

    The canonical implementation prepends an LLM-generated contextual summary
    to each chunk (Anthropic's Contextual Retrieval). The contract preserves
    chunk count and per-chunk `document_source`; only `content` is augmented.
    """

    @abstractmethod
    async def enrich(self, document: Document, chunks: list[Chunk]) -> list[Chunk]:
        ...
