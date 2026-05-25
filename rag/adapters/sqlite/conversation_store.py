from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import TracebackType

from rag.stages.conversation_store import ConversationStore
from rag.types import Message

__all__ = ["SQLiteConversationStore"]


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    conversation_id TEXT NOT NULL,
    position        INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (conversation_id, position)
)
"""


class SQLiteConversationStore(ConversationStore):
    """SQLite-backed conversation history.

    One row per turn, keyed by `(conversation_id, position)`. Position is
    assigned monotonically on insert. Uses stdlib `sqlite3` with
    `check_same_thread=False` and an `asyncio.Lock` to serialize access from
    the asyncio thread pool used by `asyncio.to_thread`.
    """

    def __init__(self, *, path: str) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "SQLiteConversationStore":
        await asyncio.to_thread(self._connect)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._conn is not None:
            conn = self._conn
            self._conn = None
            await asyncio.to_thread(conn.close)

    def _connect(self) -> None:
        if self._path != ":memory:":
            Path(self._path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    async def get_history(self, conversation_id: str) -> list[Message]:
        async with self._lock:
            rows = await asyncio.to_thread(self._fetch, conversation_id)
        return [Message(role=role, content=content) for role, content in rows]

    async def append_message(self, conversation_id: str, message: Message) -> None:
        async with self._lock:
            await asyncio.to_thread(self._insert, conversation_id, message)

    async def list_conversations(self) -> list[str]:
        async with self._lock:
            return await asyncio.to_thread(self._list)

    def _list(self) -> list[str]:
        conn = self._require_conn()
        cur = conn.execute(
            "SELECT conversation_id FROM messages "
            "GROUP BY conversation_id ORDER BY MAX(created_at) DESC"
        )
        return [row[0] for row in cur.fetchall()]

    def _fetch(self, conversation_id: str) -> list[tuple[str, str]]:
        conn = self._require_conn()
        cur = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? ORDER BY position",
            (conversation_id,),
        )
        return cur.fetchall()

    def _insert(self, conversation_id: str, message: Message) -> None:
        conn = self._require_conn()
        cur = conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        next_pos = int(cur.fetchone()[0])
        conn.execute(
            "INSERT INTO messages (conversation_id, position, role, content) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, next_pos, message.role, message.content),
        )
        conn.commit()

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "SQLiteConversationStore used outside its async context manager"
            )
        return self._conn
