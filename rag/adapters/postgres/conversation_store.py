from __future__ import annotations

from types import TracebackType

from psycopg_pool import AsyncConnectionPool

from rag.stages.conversation_store import ConversationStore
from rag.types import Message

__all__ = ["PostgresConversationStore"]


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    conversation_id TEXT NOT NULL,
    position        INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (conversation_id, position)
)
"""


class PostgresConversationStore(ConversationStore):
    """Postgres-backed conversation history.

    Same schema and contract as `SQLiteConversationStore` — same conformance
    suite passes against both. The connection pool is opened in
    `__aenter__` and closed in `__aexit__` (ADR-0007).
    """

    def __init__(
        self,
        *,
        dsn: str,
        table_name: str = "conversation_messages",
        min_size: int = 1,
        max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._table = table_name
        self._min_size = min_size
        self._max_size = max_size
        self._pool: AsyncConnectionPool | None = None

    async def __aenter__(self) -> "PostgresConversationStore":
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

    async def get_history(self, conversation_id: str) -> list[Message]:
        async with self._require_pool().connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT role, content FROM {self._table} "
                    "WHERE conversation_id = %s ORDER BY position",
                    (conversation_id,),
                )
                rows = await cur.fetchall()
        return [Message(role=role, content=content) for role, content in rows]

    async def append_message(self, conversation_id: str, message: Message) -> None:
        # Position is the next ordinal for this conversation. Computed and
        # inserted within a single transaction so concurrent appends to the
        # same conversation_id serialize via the primary key.
        async with self._require_pool().connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        f"SELECT COALESCE(MAX(position), -1) + 1 FROM {self._table} "
                        "WHERE conversation_id = %s",
                        (conversation_id,),
                    )
                    row = await cur.fetchone()
                    next_pos = int(row[0]) if row else 0
                    await cur.execute(
                        f"INSERT INTO {self._table} "
                        "(conversation_id, position, role, content) "
                        "VALUES (%s, %s, %s, %s)",
                        (conversation_id, next_pos, message.role, message.content),
                    )

    async def list_conversations(self) -> list[str]:
        async with self._require_pool().connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT conversation_id FROM {self._table} "
                    "GROUP BY conversation_id ORDER BY MAX(created_at) DESC"
                )
                rows = await cur.fetchall()
        return [row[0] for row in rows]

    def _require_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError(
                "PostgresConversationStore used outside its async context manager"
            )
        return self._pool
