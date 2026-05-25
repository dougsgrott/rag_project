"""Conformance tests every `ConversationStore` adapter must pass."""

import pytest

from rag.stages.conversation_store import ConversationStore

from tests.stages._helpers import make_message


class ConversationStoreConformance:
    @pytest.fixture
    def store(self) -> ConversationStore:
        raise NotImplementedError("subclass must provide a `store` fixture")

    async def test_get_history_for_unknown_conversation_returns_empty(
        self, store: ConversationStore
    ) -> None:
        history = await store.get_history("conv-does-not-exist")
        assert history == []

    async def test_append_then_get_history_returns_messages_in_order(
        self, store: ConversationStore
    ) -> None:
        conv_id = "conv-1"
        m1 = make_message("user", "hello")
        m2 = make_message("assistant", "hi there")
        await store.append_message(conv_id, m1)
        await store.append_message(conv_id, m2)
        history = await store.get_history(conv_id)
        assert history == [m1, m2]

    async def test_conversations_are_isolated(self, store: ConversationStore) -> None:
        await store.append_message("conv-a", make_message("user", "a"))
        await store.append_message("conv-b", make_message("user", "b"))
        history_a = await store.get_history("conv-a")
        history_b = await store.get_history("conv-b")
        assert [m.content for m in history_a] == ["a"]
        assert [m.content for m in history_b] == ["b"]

    async def test_list_conversations_empty_store_returns_empty(
        self, store: ConversationStore
    ) -> None:
        assert await store.list_conversations() == []

    async def test_list_conversations_returns_distinct_ids(
        self, store: ConversationStore
    ) -> None:
        await store.append_message("conv-a", make_message("user", "a1"))
        await store.append_message("conv-a", make_message("assistant", "a2"))
        await store.append_message("conv-b", make_message("user", "b1"))
        ids = await store.list_conversations()
        assert sorted(ids) == ["conv-a", "conv-b"]
