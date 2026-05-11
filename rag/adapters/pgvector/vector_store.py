from __future__ import annotations

from types import TracebackType

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from rag.stages.embedder import Embedder
from rag.stages.vector_store import VectorStore
from rag.types import Chunk, SearchResult

__all__ = ["PgVectorStore"]


# Cosine distance via the `<=>` operator; smaller distance ⇒ more similar.
# We surface similarity (`1 - distance`) so adapters report higher-is-better
# scores consistently across the project (see ChromaVectorStore).
_SEARCH_SQL = """
SELECT content, document_source, position, (embedding <=> %s) AS distance
FROM {table}
ORDER BY embedding <=> %s
LIMIT %s
"""

_INSERT_SQL = """
INSERT INTO {table} (id, document_source, position, content, embedding)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE
    SET content = EXCLUDED.content,
        embedding = EXCLUDED.embedding
"""


def _create_table_sql(table: str, dim: int) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        id              TEXT PRIMARY KEY,
        document_source TEXT NOT NULL,
        position        INTEGER NOT NULL,
        content         TEXT NOT NULL,
        embedding       vector({dim}) NOT NULL,
        metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb
    )
    """


class PgVectorStore(VectorStore):
    """pgvector-backed VectorStore.

    Embedding is delegated to an injected `Embedder` (ADR-0005). The
    embedding column dimensionality must be known at table-creation time;
    pass it via `dim` (default 1536 — `text-embedding-3-small`).

    The `vector` extension is created on entry; if the connecting role
    lacks `CREATE EXTENSION` privilege, ask your DBA to run
    `CREATE EXTENSION vector` once and the adapter will pick up the
    already-installed extension.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        dsn: str,
        dim: int = 1536,
        table_name: str = "rag_chunks",
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self._embedder = embedder
        self._dsn = dsn
        self._dim = dim
        self._table = table_name
        self._min_size = min_size
        self._max_size = max_size
        self._pool: AsyncConnectionPool | None = None

    async def __aenter__(self) -> "PgVectorStore":
        pool = AsyncConnectionPool(
            conninfo=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
            configure=_configure_connection,
        )
        await pool.open()
        self._pool = pool
        await self._ensure_schema()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._pool is not None:
            pool = self._pool
            self._pool = None
            await pool.close()

    async def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embedded = await self._embedder.embed(chunks)
        rows = [
            (
                self._chunk_id(ec.chunk),
                ec.chunk.document_source,
                ec.chunk.position,
                ec.chunk.content,
                ec.vector,
            )
            for ec in embedded
        ]
        async with self._require_pool().connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(_INSERT_SQL.format(table=self._table), rows)

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query_vector = await self._embedder.embed_query(query)
        async with self._require_pool().connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SEARCH_SQL.format(table=self._table),
                    (query_vector, query_vector, top_k),
                )
                rows = await cur.fetchall()
        results: list[SearchResult] = []
        for content, source, position, distance in rows:
            chunk = Chunk(
                content=content,
                document_source=source,
                position=int(position),
            )
            results.append(SearchResult(chunk=chunk, score=1.0 - float(distance)))
        return results

    async def _ensure_schema(self) -> None:
        async with self._require_pool().connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(_create_table_sql(self._table, self._dim))
            # Re-register the vector type now that the extension and table
            # exist on this connection (no-op if already registered).
            await register_vector_async(conn)

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        return f"{chunk.document_source}::{chunk.position}"

    def _require_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("PgVectorStore used outside its async context manager")
        return self._pool


async def _configure_connection(conn: psycopg.AsyncConnection) -> None:
    """Register pgvector type adapters on each pooled connection.

    Safe to call before the extension is installed: it falls back silently
    so the bootstrap `CREATE EXTENSION` in `_ensure_schema` can run on a
    connection that hasn't yet had vector types registered. Connections
    handed out *after* schema setup get the registration during
    `_ensure_schema`'s explicit call.
    """
    try:
        await register_vector_async(conn)
    except psycopg.errors.UndefinedObject:
        # `vector` type not yet installed in this database. The next time
        # the pool hands out a connection (after _ensure_schema runs once),
        # registration will succeed.
        pass
