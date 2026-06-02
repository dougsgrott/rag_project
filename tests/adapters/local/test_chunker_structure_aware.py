import pytest

from rag.adapters.local.chunker_structure_aware import StructureAwareChunker
from rag.types import Document

from tests.stages.chunker_conformance import ChunkerConformance


# --- Conformance ----------------------------------------------------------


class TestStructureAwareChunkerConformance(ChunkerConformance):
    @pytest.fixture
    def chunker(self) -> StructureAwareChunker:
        return StructureAwareChunker(chunk_size=4, overlap=1)


# --- Unit -----------------------------------------------------------------


class TestStructureAwareChunkerUnit:
    async def test_empty_document_yields_no_chunks(self) -> None:
        chunker = StructureAwareChunker()
        chunks = await chunker.chunk(Document(content="", source="t.txt"))
        assert chunks == []

    async def test_plain_paragraph_falls_back_to_size_split(self) -> None:
        # No headers, no tables — a single long paragraph. With chunk_size=10,
        # overlap=2, step=8, content "abcdefghij...t" (20 chars) → 3 chunks.
        chunker = StructureAwareChunker(chunk_size=10, overlap=2)
        doc = Document(content="abcdefghijklmnopqrst", source="t.txt")
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2
        assert all(len(c.content) <= 10 for c in chunks)
        assert all(c.document_source == "t.txt" for c in chunks)
        assert [c.position for c in chunks] == list(range(len(chunks)))

    async def test_table_stays_atomic(self) -> None:
        chunker = StructureAwareChunker(chunk_size=20, overlap=4)
        table = (
            "| col1 | col2 |\n"
            "| ---- | ---- |\n"
            "| a    | b    |\n"
            "| c    | d    |\n"
        )
        chunks = await chunker.chunk(Document(content=table, source="t.md"))
        # Table totals ~64 chars > chunk_size=20, but must remain one chunk.
        assert len(chunks) == 1
        assert chunks[0].content == table.rstrip("\n")
        assert chunks[0].metadata["block_types"] == ["table"]

    async def test_no_table_row_split_across_chunks(self) -> None:
        """The critical contract: no row is ever cut in half."""
        chunker = StructureAwareChunker(chunk_size=20, overlap=4)
        table = (
            "| col1 | col2 |\n"
            "| ---- | ---- |\n"
            "| alpha | bravo |\n"
            "| charlie | delta |\n"
        )
        chunks = await chunker.chunk(Document(content=table, source="t.md"))
        for c in chunks:
            for line in c.content.splitlines():
                stripped = line.strip()
                if stripped.startswith("|"):
                    # A row that starts with `|` must end with `|` — otherwise
                    # it was cut mid-line.
                    assert stripped.endswith("|"), (
                        f"table row was split inside a chunk:\n{c.content!r}"
                    )

    async def test_oversized_table_still_one_chunk(self) -> None:
        chunker = StructureAwareChunker(chunk_size=30, overlap=4)
        rows = ["| col1 | col2 |", "| ---- | ---- |"] + [
            f"| row{i:02d} | val{i:02d} |" for i in range(10)
        ]
        table = "\n".join(rows) + "\n"
        chunks = await chunker.chunk(Document(content=table, source="t.md"))
        assert len(chunks) == 1
        assert chunks[0].content == table.rstrip("\n")

    async def test_headers_start_new_chunks(self) -> None:
        chunker = StructureAwareChunker(chunk_size=80, overlap=8)
        text = (
            "# Section 1\n"
            "\n"
            "First section content.\n"
            "\n"
            "# Section 2\n"
            "\n"
            "Second section content.\n"
        )
        chunks = await chunker.chunk(Document(content=text, source="t.md"))
        assert len(chunks) == 2
        assert chunks[0].content.startswith("# Section 1")
        assert "First section content." in chunks[0].content
        assert chunks[1].content.startswith("# Section 2")
        assert "Second section content." in chunks[1].content

    async def test_consecutive_headers_stack_into_one_chunk(self) -> None:
        """`# H1` immediately followed by `## H2` shouldn't orphan H1."""
        chunker = StructureAwareChunker(chunk_size=200, overlap=20)
        text = (
            "# Top\n"
            "\n"
            "## Sub\n"
            "\n"
            "Body of the subsection.\n"
        )
        chunks = await chunker.chunk(Document(content=text, source="t.md"))
        assert len(chunks) == 1
        assert "# Top" in chunks[0].content
        assert "## Sub" in chunks[0].content
        assert "Body of the subsection." in chunks[0].content

    async def test_header_prepended_when_paragraph_triggers_size_split(self) -> None:
        chunker = StructureAwareChunker(chunk_size=40, overlap=4)
        long_para = "x" * 200
        text = f"# Section\n\n{long_para}\n"
        chunks = await chunker.chunk(Document(content=text, source="t.md"))
        # First chunk should still start with the section header, not lose it.
        assert chunks[0].content.startswith("# Section")
        # The header survives despite the paragraph spilling into multiple chunks.
        assert len(chunks) >= 2

    async def test_mixed_structure_preserves_table_inside_section(self) -> None:
        chunker = StructureAwareChunker(chunk_size=200, overlap=20)
        text = (
            "# Introduction\n"
            "\n"
            "SICRO computes road costs.\n"
            "\n"
            "## Cost components\n"
            "\n"
            "| Item       | Description    |\n"
            "| ---------- | -------------- |\n"
            "| Materials  | Raw inputs     |\n"
            "| Labor      | Worker time    |\n"
            "\n"
            "Each component has a unit price.\n"
        )
        chunks = await chunker.chunk(Document(content=text, source="t.md"))
        # Exactly one chunk should carry the table, and it must include every
        # row whole (`Materials` and `Labor` rows on the same chunk).
        table_chunks = [c for c in chunks if "table" in c.metadata["block_types"]]
        assert len(table_chunks) == 1
        assert "Materials" in table_chunks[0].content
        assert "Labor" in table_chunks[0].content
        # And the introductory header survived as the start of some chunk.
        assert any(c.content.startswith("# Introduction") for c in chunks)

    @pytest.mark.parametrize(
        "chunk_size,overlap",
        [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)],
    )
    def test_invalid_config_raises(self, chunk_size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            StructureAwareChunker(chunk_size=chunk_size, overlap=overlap)
