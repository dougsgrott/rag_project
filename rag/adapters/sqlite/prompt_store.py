from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import TracebackType

from rag.errors import ConfigurationError
from rag.stages.prompt_store import PromptStore

__all__ = ["SQLitePromptStore"]


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    domain      TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    author      TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_prompts_domain_id ON prompts (domain, id DESC)
"""


class SQLitePromptStore(PromptStore):
    """SQLite-backed versioned prompt store.

    `save_prompt` always appends a new row — old versions are retained for
    audit. `get_prompt` returns the most recently saved prompt for a domain.
    """

    def __init__(self, *, path: str) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "SQLitePromptStore":
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
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

    async def get_prompt(self, domain: str) -> str:
        async with self._lock:
            row = await asyncio.to_thread(self._fetch_latest, domain)
        if row is None:
            raise ConfigurationError(f"no prompt saved for domain '{domain}'")
        return str(row[0])

    async def save_prompt(self, domain: str, prompt: str, author: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._insert, domain, prompt, author)

    def _fetch_latest(self, domain: str) -> tuple[str] | None:
        conn = self._require_conn()
        cur = conn.execute(
            "SELECT prompt FROM prompts WHERE domain = ? ORDER BY id DESC LIMIT 1",
            (domain,),
        )
        return cur.fetchone()

    def _insert(self, domain: str, prompt: str, author: str) -> None:
        conn = self._require_conn()
        conn.execute(
            "INSERT INTO prompts (domain, prompt, author) VALUES (?, ?, ?)",
            (domain, prompt, author),
        )
        conn.commit()

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "SQLitePromptStore used outside its async context manager"
            )
        return self._conn
