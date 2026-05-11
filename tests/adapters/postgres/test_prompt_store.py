from typing import AsyncIterator

import pytest

from rag.adapters.postgres.prompt_store import PostgresPromptStore
from rag.errors import ConfigurationError

from tests._postgres import TEST_POSTGRES_DSN, requires_integration, unique_table
from tests.stages.prompt_store_conformance import PromptStoreConformance


pytestmark = requires_integration


class TestPostgresPromptStoreConformance(PromptStoreConformance):
    @pytest.fixture
    async def store(self) -> AsyncIterator[PostgresPromptStore]:  # type: ignore[override]
        table = unique_table("test_prompts")
        async with PostgresPromptStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
            try:
                yield s
            finally:
                async with s._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    await conn.execute(f"DROP TABLE IF EXISTS {table}")


class TestPostgresPromptStoreUnit:
    async def test_unknown_domain_raises(self) -> None:
        table = unique_table("test_prompts")
        try:
            async with PostgresPromptStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
                with pytest.raises(ConfigurationError):
                    await s.get_prompt("missing")
        finally:
            async with PostgresPromptStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
                async with s._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    await conn.execute(f"DROP TABLE IF EXISTS {table}")

    async def test_versions_retained_for_audit(self) -> None:
        table = unique_table("test_prompts")
        try:
            async with PostgresPromptStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
                await s.save_prompt("d", "v1", author="alice")
                await s.save_prompt("d", "v2", author="bob")
                assert await s.get_prompt("d") == "v2"
                async with s._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    async with conn.cursor() as cur:
                        await cur.execute(
                            f"SELECT prompt, author FROM {table} WHERE domain = 'd' ORDER BY id"
                        )
                        rows = await cur.fetchall()
                assert rows == [("v1", "alice"), ("v2", "bob")]
        finally:
            async with PostgresPromptStore(dsn=TEST_POSTGRES_DSN, table_name=table) as s:
                async with s._require_pool().connection() as conn:  # type: ignore[attr-defined]
                    await conn.execute(f"DROP TABLE IF EXISTS {table}")
