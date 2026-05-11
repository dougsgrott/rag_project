from pathlib import Path
from typing import AsyncIterator

import pytest

from rag.adapters.sqlite.prompt_store import SQLitePromptStore
from rag.errors import ConfigurationError

from tests.stages.prompt_store_conformance import PromptStoreConformance


class TestSQLitePromptStoreConformance(PromptStoreConformance):
    @pytest.fixture
    async def store(self, tmp_path: Path) -> AsyncIterator[SQLitePromptStore]:  # type: ignore[override]
        async with SQLitePromptStore(path=str(tmp_path / "prompts.db")) as s:
            yield s


class TestSQLitePromptStoreUnit:
    async def test_get_prompt_unknown_domain_raises(self, tmp_path: Path) -> None:
        async with SQLitePromptStore(path=str(tmp_path / "p.db")) as s:
            with pytest.raises(ConfigurationError):
                await s.get_prompt("missing")

    async def test_save_retains_history(self, tmp_path: Path) -> None:
        db = str(tmp_path / "p.db")
        async with SQLitePromptStore(path=db) as s:
            await s.save_prompt("d", "v1", author="alice")
            await s.save_prompt("d", "v2", author="bob")
            assert await s.get_prompt("d") == "v2"

        # `get_prompt` returns the latest; older rows are retained for audit.
        import sqlite3
        with sqlite3.connect(db) as conn:
            rows = list(conn.execute("SELECT prompt, author FROM prompts WHERE domain='d' ORDER BY id"))
        assert rows == [("v1", "alice"), ("v2", "bob")]

    async def test_persistence_across_context_re_enter(self, tmp_path: Path) -> None:
        db = str(tmp_path / "p.db")
        async with SQLitePromptStore(path=db) as s:
            await s.save_prompt("d", "prompt", author="alice")
        async with SQLitePromptStore(path=db) as s:
            assert await s.get_prompt("d") == "prompt"

    async def test_use_outside_context_raises(self, tmp_path: Path) -> None:
        store = SQLitePromptStore(path=str(tmp_path / "x.db"))
        with pytest.raises(RuntimeError):
            await store.get_prompt("d")
