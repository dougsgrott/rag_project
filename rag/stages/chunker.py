from abc import ABC, abstractmethod

from rag.types import Chunk, Document

__all__ = ["Chunker"]


class Chunker(ABC):
    """Splits a `Document` into retrieval-sized `Chunk`s.

    The chunking strategy (fixed-size, sentence-boundary, structure-aware) is
    an adapter concern.
    """

    @abstractmethod
    async def chunk(self, document: Document) -> list[Chunk]:
        ...
