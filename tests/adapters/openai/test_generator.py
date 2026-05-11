from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from rag.adapters.openai.generator import OpenAIGenerator
from rag.errors import (
    BackendCommunicationError,
    ConfigurationError,
    GenerationError,
    RateLimitError,
)
from rag.types import Message

from tests.stages._helpers import make_message, make_search_results
from tests.stages.generator_conformance import GeneratorConformance


def _build_fake_client(content: str = "mock answer") -> MagicMock:
    client = MagicMock(name="AsyncOpenAI")
    message = MagicMock(); message.content = content
    choice = MagicMock(); choice.message = message
    response = MagicMock(); response.choices = [choice]
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    client.close = AsyncMock()
    return client


def _patch_async_openai(client: MagicMock) -> Any:
    return patch("rag.adapters.openai.generator.AsyncOpenAI", return_value=client)


class TestOpenAIGeneratorConformance(GeneratorConformance):
    @pytest.fixture
    async def generator(self):  # type: ignore[no-untyped-def]
        client = _build_fake_client()
        with _patch_async_openai(client):
            async with OpenAIGenerator(api_key="sk-test", model="gpt-4o-mini") as g:
                yield g


class TestOpenAIGeneratorUnit:
    async def test_constructor_rejects_empty_api_key(self) -> None:
        with pytest.raises(ConfigurationError):
            OpenAIGenerator(api_key="")

    async def test_generate_outside_context_raises(self) -> None:
        g = OpenAIGenerator(api_key="sk-test")
        with pytest.raises(ConfigurationError):
            await g.generate(
                query="q", context=[], system_prompt="be helpful", history=[]
            )

    async def test_message_payload_layout(self) -> None:
        client = _build_fake_client()
        history = [make_message("user", "prior q"), make_message("assistant", "prior a")]
        with _patch_async_openai(client):
            async with OpenAIGenerator(api_key="sk-test") as g:
                await g.generate(
                    query="new q",
                    context=make_search_results(2),
                    system_prompt="ground answers in the context",
                    history=history,
                )
        sent = client.chat.completions.create.await_args.kwargs["messages"]
        assert sent[0] == {"role": "system", "content": "ground answers in the context"}
        assert sent[1] == {"role": "user", "content": "prior q"}
        assert sent[2] == {"role": "assistant", "content": "prior a"}
        assert sent[3]["role"] == "user"
        assert "Context:" in sent[3]["content"]
        assert "new q" in sent[3]["content"]

    async def test_no_context_passes_query_verbatim(self) -> None:
        client = _build_fake_client()
        with _patch_async_openai(client):
            async with OpenAIGenerator(api_key="sk-test") as g:
                await g.generate(query="plain q", context=[], system_prompt="s", history=[])
        sent = client.chat.completions.create.await_args.kwargs["messages"]
        assert sent[-1] == {"role": "user", "content": "plain q"}

    async def test_returns_assistant_message_with_content(self) -> None:
        client = _build_fake_client(content="grounded answer")
        with _patch_async_openai(client):
            async with OpenAIGenerator(api_key="sk-test") as g:
                result = await g.generate(
                    query="q", context=[], system_prompt="s", history=[]
                )
        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert result.content == "grounded answer"

    async def test_empty_choices_raises_generation_error(self) -> None:
        client = _build_fake_client()
        client.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[]))
        with _patch_async_openai(client):
            async with OpenAIGenerator(api_key="sk-test") as g:
                with pytest.raises(GenerationError):
                    await g.generate(query="q", context=[], system_prompt="s", history=[])

    async def test_rate_limit_maps_to_rag_error(self) -> None:
        client = _build_fake_client()
        client.chat.completions.create = AsyncMock(
            side_effect=openai.RateLimitError(
                "slow down", response=MagicMock(), body=None
            )
        )
        with _patch_async_openai(client):
            async with OpenAIGenerator(api_key="sk-test") as g:
                with pytest.raises(RateLimitError):
                    await g.generate(query="q", context=[], system_prompt="s", history=[])

    async def test_auth_error_maps_to_configuration_error(self) -> None:
        client = _build_fake_client()
        client.chat.completions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                "bad", response=MagicMock(), body=None
            )
        )
        with _patch_async_openai(client):
            async with OpenAIGenerator(api_key="sk-test") as g:
                with pytest.raises(ConfigurationError):
                    await g.generate(query="q", context=[], system_prompt="s", history=[])

    async def test_connection_error_maps_to_backend_error(self) -> None:
        client = _build_fake_client()
        client.chat.completions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )
        with _patch_async_openai(client):
            async with OpenAIGenerator(api_key="sk-test") as g:
                with pytest.raises(BackendCommunicationError):
                    await g.generate(query="q", context=[], system_prompt="s", history=[])
