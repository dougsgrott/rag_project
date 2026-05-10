"""Conformance tests every `Embedder` adapter must pass."""

import pytest

from rag.stages.embedder import Embedder
from rag.types import EmbeddedChunk

from tests.stages._helpers import make_chunks


class EmbedderConformance:
    @pytest.fixture
    def embedder(self) -> Embedder:
        raise NotImplementedError("subclass must provide an `embedder` fixture")

    async def test_embed_returns_one_vector_per_chunk(self, embedder: Embedder) -> None:
        chunks = make_chunks(3)
        embedded = await embedder.embed(chunks)
        assert isinstance(embedded, list)
        assert len(embedded) == len(chunks)
        for original, ec in zip(chunks, embedded):
            assert isinstance(ec, EmbeddedChunk)
            assert ec.chunk == original
            assert isinstance(ec.vector, list) and len(ec.vector) > 0
            assert all(isinstance(v, float) for v in ec.vector)

    async def test_embed_query_returns_vector(self, embedder: Embedder) -> None:
        vector = await embedder.embed_query("what is alpha?")
        assert isinstance(vector, list) and len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    async def test_embed_query_dimension_matches_chunks(self, embedder: Embedder) -> None:
        chunks = make_chunks(1)
        embedded = await embedder.embed(chunks)
        query_vector = await embedder.embed_query("hello")
        assert len(query_vector) == len(embedded[0].vector)
