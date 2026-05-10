"""Shared fixture builders for stage conformance suites."""

from rag.stages.embedder import Embedder
from rag.types import Chunk, Document, EmbeddedChunk, Message, SearchResult


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


class StubEmbedder(Embedder):
    """Deterministic 4-dimensional embedder for tests.

    Maps text into a normalized vector of `[alpha, digit, space, other]`
    character-class counts. Cheap, deterministic, and produces non-zero
    vectors for any non-empty string — enough to exercise vector-store and
    ingest-pipeline paths without touching a real embedding API.
    """

    DIM = 4

    async def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        return [EmbeddedChunk(chunk=c, vector=_text_to_vec(c.content)) for c in chunks]

    async def embed_query(self, query: str) -> list[float]:
        return _text_to_vec(query)


def _text_to_vec(text: str) -> list[float]:
    counts = [0.0, 0.0, 0.0, 0.0]
    for ch in text or " ":
        if ch.isalpha():
            counts[0] += 1
        elif ch.isdigit():
            counts[1] += 1
        elif ch.isspace():
            counts[2] += 1
        else:
            counts[3] += 1
    norm = sum(c * c for c in counts) ** 0.5 or 1.0
    return [c / norm for c in counts]
