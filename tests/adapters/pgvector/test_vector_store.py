from typing import AsyncIterator

import pytest

from rag.adapters.pgvector.vector_store import PgVectorStore
from rag.types import SearchResult

from tests._postgres import TEST_POSTGRES_DSN, requires_integration, unique_table
from tests.stages._helpers import StubEmbedder, make_chunks
from tests.stages.vector_store_conformance import VectorStoreConformance


pytestmark = requires_integration


class TestPgVectorStoreConformance(VectorStoreConformance):
    @pytest.fixture
    async def store(self) -> AsyncIterator[PgVectorStore]:  # type: ignore[override]
        table = unique_table("test_pgv")
        async with PgVectorStore(
            embedder=StubEmbedder(),
            dsn=TEST_POSTGRES_DSN,
            dim=StubEmbedder.DIM,
            table_name=table,
        ) as s:
            try:
                yield s
            finally:
                async with s._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    await conn.execute(f"DROP TABLE IF EXISTS {table}")


class TestPgVectorStoreUnit:
    async def test_search_returns_indexed_metadata(self) -> None:
        table = unique_table("test_pgv")
        try:
            async with PgVectorStore(
                embedder=StubEmbedder(),
                dsn=TEST_POSTGRES_DSN,
                dim=StubEmbedder.DIM,
                table_name=table,
            ) as store:
                await store.index(make_chunks(3, source="abc.txt"))
                results = await store.search("alpha", top_k=3)
            assert len(results) == 3
            sources = {r.chunk.document_source for r in results}
            positions = {r.chunk.position for r in results}
            assert sources == {"abc.txt"}
            assert positions == {0, 1, 2}
            for r in results:
                assert isinstance(r, SearchResult)
                assert isinstance(r.score, float)
        finally:
            async with PgVectorStore(
                embedder=StubEmbedder(),
                dsn=TEST_POSTGRES_DSN,
                dim=StubEmbedder.DIM,
                table_name=table,
            ) as store:
                async with store._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    await conn.execute(f"DROP TABLE IF EXISTS {table}")

    async def test_upsert_avoids_duplicates(self) -> None:
        table = unique_table("test_pgv")
        try:
            async with PgVectorStore(
                embedder=StubEmbedder(),
                dsn=TEST_POSTGRES_DSN,
                dim=StubEmbedder.DIM,
                table_name=table,
            ) as store:
                chunks = make_chunks(2)
                await store.index(chunks)
                await store.index(chunks)  # second pass should upsert, not duplicate
                results = await store.search("alpha", top_k=10)
            assert len(results) == 2
        finally:
            async with PgVectorStore(
                embedder=StubEmbedder(),
                dsn=TEST_POSTGRES_DSN,
                dim=StubEmbedder.DIM,
                table_name=table,
            ) as store:
                async with store._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    await conn.execute(f"DROP TABLE IF EXISTS {table}")

    async def test_use_outside_context_raises(self) -> None:
        store = PgVectorStore(
            embedder=StubEmbedder(),
            dsn=TEST_POSTGRES_DSN,
            dim=StubEmbedder.DIM,
            table_name=unique_table("x"),
        )
        with pytest.raises(RuntimeError):
            await store.search("anything", top_k=1)
