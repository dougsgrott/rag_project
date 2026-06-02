"""Structure-aware chunker.

Splits a `Document` on structural boundaries (headers, paragraph breaks,
table boundaries) and only falls back to size-based slicing for paragraphs
that exceed `chunk_size` on their own. Addresses the "Fixed-Size Chunker
Massacre" pre-mortem failure mode where tabular rows were cut in half and
headers divorced from their paragraphs.

Drop-in for `FixedSizeChunker` — same constructor signature.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rag.stages.chunker import Chunker
from rag.types import Chunk, Document

__all__ = ["StructureAwareChunker"]


BlockType = Literal["header", "table", "paragraph"]

_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


@dataclass(frozen=True)
class _Block:
    type: BlockType
    text: str            # original lines with trailing newlines preserved
    start_line: int      # 0-indexed line number in the source document


def _is_header(line: str) -> bool:
    return bool(_HEADER_RE.match(line))


def _is_table_row(line: str) -> bool:
    return bool(_TABLE_ROW_RE.match(line))


def _parse_blocks(text: str) -> list[_Block]:
    """Split text on blank-line boundaries; classify each segment.

    - Segment composed entirely of `|...|` lines (≥2) → ``table``.
    - Segment whose first line is a markdown header → ``header`` block for
      that line, plus a ``paragraph`` block for the rest (if any).
    - Everything else → ``paragraph``.
    """
    lines = text.splitlines(keepends=True)
    n = len(lines)
    blocks: list[_Block] = []
    i = 0
    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        start = i
        while i < n and lines[i].strip():
            i += 1
        segment = lines[start:i]
        seg_text = "".join(segment)
        if len(segment) >= 2 and all(_is_table_row(line) for line in segment):
            blocks.append(_Block("table", seg_text, start))
        elif _is_header(segment[0]):
            blocks.append(_Block("header", segment[0], start))
            if len(segment) > 1:
                blocks.append(
                    _Block("paragraph", "".join(segment[1:]), start + 1)
                )
        else:
            blocks.append(_Block("paragraph", seg_text, start))
    return blocks


class StructureAwareChunker(Chunker):
    """Markdown-structure-aware chunker.

    Behaviour:

    - **Tables** are atomic. A `|...|`-style markdown table emits as one
      chunk, even if it exceeds `chunk_size` — never split a row.
    - **Headers** start a new chunk whenever the buffer already contains
      content. Stacked headers (header followed immediately by another
      header) accumulate so a section title never gets a chunk of its own
      without any content.
    - **Paragraphs** are packed greedily up to `chunk_size`. A paragraph
      that on its own exceeds `chunk_size` is split character-wise with
      `overlap` (the `FixedSizeChunker` fallback); any header buffered
      immediately above it is prepended to the first split chunk so the
      section's title isn't orphaned.
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
        blocks = _parse_blocks(text)
        if not blocks:
            return []
        packed = self._pack(blocks)
        return [
            Chunk(
                content=content,
                document_source=document.source,
                position=index,
                metadata=meta,
            )
            for index, (content, meta) in enumerate(packed)
        ]

    # ------------------------------------------------------------------
    # Packing
    # ------------------------------------------------------------------

    def _pack(self, blocks: list[_Block]) -> list[tuple[str, dict]]:
        chunks: list[tuple[str, dict]] = []
        buf: list[_Block] = []
        buf_len = 0

        def flush() -> None:
            nonlocal buf, buf_len
            if not buf:
                return
            content = "\n\n".join(b.text.rstrip("\n") for b in buf)
            chunks.append(
                (
                    content,
                    {
                        "start_line": buf[0].start_line,
                        "block_types": [b.type for b in buf],
                    },
                )
            )
            buf = []
            buf_len = 0

        def buf_has_content() -> bool:
            return any(b.type != "header" for b in buf)

        for block in blocks:
            if block.type == "table":
                # Fold any buffered header-only prefix into the table chunk
                # so a section title isn't orphaned just because the next
                # block is atomic. The table itself is still emitted whole.
                if buf and not buf_has_content():
                    prefix = "\n\n".join(b.text.rstrip("\n") for b in buf) + "\n\n"
                    types_prefix = [b.type for b in buf]
                    start_line = buf[0].start_line
                    buf = []
                    buf_len = 0
                else:
                    flush()
                    prefix = ""
                    types_prefix = []
                    start_line = block.start_line
                chunks.append(
                    (
                        prefix + block.text.rstrip("\n"),
                        {"start_line": start_line, "block_types": types_prefix + ["table"]},
                    )
                )
                continue

            if block.type == "header":
                if buf_has_content():
                    flush()
                buf.append(block)
                buf_len += len(block.text)
                continue

            # Paragraph.
            block_len = len(block.text)
            if block_len > self._chunk_size:
                # Size-based fallback. If only headers are buffered, fold
                # them into the first split chunk so the section title
                # isn't orphaned.
                if buf and not buf_has_content():
                    prefix = "\n\n".join(b.text.rstrip("\n") for b in buf) + "\n\n"
                    start_line = buf[0].start_line
                    buf = []
                    buf_len = 0
                else:
                    flush()
                    prefix = ""
                    start_line = block.start_line
                chunks.extend(self._size_split(prefix + block.text, start_line))
                continue

            if buf_has_content() and buf_len + block_len > self._chunk_size:
                flush()

            buf.append(block)
            buf_len += block_len

        flush()
        return chunks

    def _size_split(self, text: str, start_line: int) -> list[tuple[str, dict]]:
        """Character-window split with overlap. Mirrors `FixedSizeChunker`."""
        step = self._chunk_size - self._overlap
        out: list[tuple[str, dict]] = []
        for start in range(0, len(text), step):
            end = min(start + self._chunk_size, len(text))
            piece = text[start:end].rstrip("\n")
            if piece:
                out.append(
                    (
                        piece,
                        {"start_line": start_line, "block_types": ["paragraph"]},
                    )
                )
            if end == len(text):
                break
        return out
