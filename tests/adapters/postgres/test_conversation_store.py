from typing import AsyncIterator

import pytest

from rag.adapters.postgres.conversation_store import PostgresConversationStore
from rag.types import Message

from tests._postgres import TEST_POSTGRES_DSN, requires_integration, unique_table
from tests.stages.conversation_store_conformance import ConversationStoreConformance


pytestmark = requires_integration


class TestPostgresConversationStoreConformance(ConversationStoreConformance):
    @pytest.fixture
    async def store(self) -> AsyncIterator[PostgresConversationStore]:  # type: ignore[override]
        table = unique_table("test_conv")
        async with PostgresConversationStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
            try:
                yield s
            finally:
                # best-effort cleanup so repeated runs don't accumulate tables
                async with s._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    await conn.execute(f"DROP TABLE IF EXISTS {table}")


class TestPostgresConversationStoreUnit:
    async def test_use_outside_context_raises(self) -> None:
        store = PostgresConversationStore(dsn=TEST_POSTGRES_DSN, table_name=unique_table("x"))
        with pytest.raises(RuntimeError):
            await store.get_history("conv")

    async def test_persistence_across_context_re_enter(self) -> None:
        table = unique_table("test_conv")
        try:
            async with PostgresConversationStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
                await s.append_message("conv", Message("user", "first"))
            async with PostgresConversationStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
                history = await s.get_history("conv")
            assert history == [Message("user", "first")]
        finally:
            async with PostgresConversationStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
                async with s._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    await conn.execute(f"DROP TABLE IF EXISTS {table}")
