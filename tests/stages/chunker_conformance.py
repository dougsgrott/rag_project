"""Conformance tests every `Chunker` adapter must pass."""

import pytest

from rag.stages.chunker import Chunker
from rag.types import Chunk

from tests.stages._helpers import make_document


class ChunkerConformance:
    @pytest.fixture
    def chunker(self) -> Chunker:
        raise NotImplementedError("subclass must provide a `chunker` fixture")

    async def test_chunk_returns_chunks_carrying_document_source(self, chunker: Chunker) -> None:
        document = make_document(content="alpha beta gamma delta epsilon", source="example.txt")
        chunks = await chunker.chunk(document)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.document_source == document.source
            assert isinstance(c.content, str) and c.content

    async def test_chunk_positions_are_non_negative_and_unique(self, chunker: Chunker) -> None:
        chunks = await chunker.chunk(make_document(content="a b c d e f g h"))
        positions = [c.position for c in chunks]
        assert all(p >= 0 for p in positions)
        assert len(set(positions)) == len(positions)
