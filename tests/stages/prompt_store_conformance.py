"""Conformance tests every `PromptStore` adapter must pass."""

import pytest

from rag.stages.prompt_store import PromptStore


class PromptStoreConformance:
    @pytest.fixture
    def store(self) -> PromptStore:
        raise NotImplementedError("subclass must provide a `store` fixture")

    async def test_save_then_get_returns_latest_prompt(self, store: PromptStore) -> None:
        await store.save_prompt("finance", "v1 prompt", author="alice")
        await store.save_prompt("finance", "v2 prompt", author="bob")
        assert await store.get_prompt("finance") == "v2 prompt"

    async def test_domains_are_isolated(self, store: PromptStore) -> None:
        await store.save_prompt("finance", "fin prompt", author="alice")
        await store.save_prompt("health", "health prompt", author="alice")
        assert await store.get_prompt("finance") == "fin prompt"
        assert await store.get_prompt("health") == "health prompt"
