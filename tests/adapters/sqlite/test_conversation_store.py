from pathlib import Path
from typing import AsyncIterator

import pytest

from rag.adapters.sqlite.conversation_store import SQLiteConversationStore
from rag.types import Message

from tests.stages.conversation_store_conformance import ConversationStoreConformance


class TestSQLiteConversationStoreConformance(ConversationStoreConformance):
    @pytest.fixture
    async def store(self, tmp_path: Path) -> AsyncIterator[SQLiteConversationStore]:  # type: ignore[override]
        async with SQLiteConversationStore(path=str(tmp_path / "conv.db")) as s:
            yield s


class TestSQLiteConversationStoreUnit:
    async def test_persistence_across_context_re_enter(self, tmp_path: Path) -> None:
        db = str(tmp_path / "conv.db")
        async with SQLiteConversationStore(path=db) as s:
            await s.append_message("conv", Message("user", "first"))
        async with SQLiteConversationStore(path=db) as s:
            history = await s.get_history("conv")
        assert history == [Message("user", "first")]

    async def test_use_outside_context_raises(self, tmp_path: Path) -> None:
        store = SQLiteConversationStore(path=str(tmp_path / "x.db"))
        with pytest.raises(RuntimeError):
            await store.get_history("conv")

    async def test_creates_missing_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c.db"
        async with SQLiteConversationStore(path=str(nested)) as s:
            await s.append_message("conv", Message("user", "ok"))
        assert nested.exists()
