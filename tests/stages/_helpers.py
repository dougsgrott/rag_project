"""Shared fixture builders for stage conformance suites."""

from rag.types import Chunk, Document, Message, SearchResult


def make_document(content: str = "alpha beta gamma", source: str = "doc.txt") -> Document:
    return Document(content=content, source=source, metadata={"title": "Sample"})


def make_chunk(position: int = 0, content: str | None = None, source: str = "doc.txt") -> Chunk:
    return Chunk(
        content=content if content is not None else f"chunk-{position}",
        document_source=source,
        position=position,
    )


def make_chunks(n: int, source: str = "doc.txt") -> list[Chunk]:
    return [make_chunk(position=i, source=source) for i in range(n)]


def make_search_results(n: int, source: str = "doc.txt") -> list[SearchResult]:
    return [
        SearchResult(chunk=make_chunk(position=i, source=source), score=1.0 - 0.01 * i)
        for i in range(n)
    ]


def make_message(role: str = "user", content: str = "hello") -> Message:
    return Message(role=role, content=content)
