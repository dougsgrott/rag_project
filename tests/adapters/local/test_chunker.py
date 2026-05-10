import pytest

from rag.adapters.local.chunker import FixedSizeChunker
from rag.types import Document

from tests.stages.chunker_conformance import ChunkerConformance


class TestFixedSizeChunkerConformance(ChunkerConformance):
    @pytest.fixture
    def chunker(self) -> FixedSizeChunker:
        return FixedSizeChunker(chunk_size=4, overlap=1)


class TestFixedSizeChunkerUnit:
    async def test_splits_with_expected_overlap(self) -> None:
        chunker = FixedSizeChunker(chunk_size=5, overlap=2)
        document = Document(content="abcdefghij", source="t.txt")
        chunks = await chunker.chunk(document)
        contents = [c.content for c in chunks]
        # step = 5 - 2 = 3; the third window (start=6, end=10) hits end-of-text
        # and the loop breaks rather than producing a trailing fragment.
        assert contents == ["abcde", "defgh", "ghij"]
        assert [c.position for c in chunks] == [0, 1, 2]
        assert all(c.document_source == "t.txt" for c in chunks)

    async def test_short_document_yields_single_chunk(self) -> None:
        chunker = FixedSizeChunker(chunk_size=128, overlap=16)
        document = Document(content="short", source="t.txt")
        chunks = await chunker.chunk(document)
        assert len(chunks) == 1
        assert chunks[0].content == "short"

    async def test_empty_document_yields_no_chunks(self) -> None:
        chunker = FixedSizeChunker()
        chunks = await chunker.chunk(Document(content="", source="t.txt"))
        assert chunks == []

    @pytest.mark.parametrize(
        "chunk_size,overlap",
        [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)],
    )
    def test_invalid_config_raises(self, chunk_size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
