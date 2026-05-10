from rag.stages.chunker import Chunker
from rag.types import Chunk, Document

__all__ = ["FixedSizeChunker"]


class FixedSizeChunker(Chunker):
    """Character-based fixed-size sliding window chunker.

    `chunk_size` and `overlap` are in characters. Token-aware chunking is the
    job of `StructureAwareChunker` (issue #008).
    """

    def __init__(self, *, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self._chunk_size = chunk_size
        self._overlap = overlap

    async def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []
        step = self._chunk_size - self._overlap
        chunks: list[Chunk] = []
        position = 0
        for start in range(0, len(text), step):
            end = min(start + self._chunk_size, len(text))
            chunks.append(
                Chunk(
                    content=text[start:end],
                    document_source=document.source,
                    position=position,
                    metadata={"start": start, "end": end},
                )
            )
            position += 1
            if end == len(text):
                break
        return chunks
