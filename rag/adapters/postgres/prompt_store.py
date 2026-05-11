from __future__ import annotations

from types import TracebackType

from psycopg_pool import AsyncConnectionPool

from rag.errors import ConfigurationError
from rag.stages.prompt_store import PromptStore

__all__ = ["PostgresPromptStore"]


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    id          BIGSERIAL PRIMARY KEY,
    domain      TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    author      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_{table}_domain_id ON {table} (domain, id DESC)
"""


class PostgresPromptStore(PromptStore):
    """Postgres-backed versioned prompt store.

    Append-only: every `save_prompt` inserts a new row; `get_prompt` returns
    the most recently-saved entry for a domain. Same contract as
    `SQLitePromptStore`.
    """

    def __init__(
        self,
        *,
        dsn: str,
        table_name: str = "prompts",
        min_size: int = 1,
        max_size: int = 5,
    ) -> None:
        self._dsn = dsn
        self._table = table_name
        self._min_size = min_size
        self._max_size = max_size
        self._pool: AsyncConnectionPool | None = None

    async def __aenter__(self) -> "PostgresPromptStore":
        pool = AsyncConnectionPool(
            conninfo=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
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

    async def _ensure_schema(self) -> None:
        async with self._require_pool().connection() as conn:
            await conn.execute(_CREATE_TABLE.format(table=self._table))
            await conn.execute(_CREATE_INDEX.format(table=self._table))

    async def get_prompt(self, domain: str) -> str:
        async with self._require_pool().connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT prompt FROM {self._table} "
                    "WHERE domain = %s ORDER BY id DESC LIMIT 1",
                    (domain,),
                )
                row = await cur.fetchone()
        if row is None:
            raise ConfigurationError(f"no prompt saved for domain '{domain}'")
        return str(row[0])

    async def save_prompt(self, domain: str, prompt: str, author: str) -> None:
        async with self._require_pool().connection() as conn:
            await conn.execute(
                f"INSERT INTO {self._table} (domain, prompt, author) VALUES (%s, %s, %s)",
                (domain, prompt, author),
            )

    def _require_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("PostgresPromptStore used outside its async context manager")
        return self._pool
