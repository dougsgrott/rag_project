from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from rag.adapters.openai.embedder import OpenAIEmbedder
from rag.errors import (
    BackendCommunicationError,
    ConfigurationError,
    RateLimitError,
)

from tests.stages._helpers import make_chunks
from tests.stages.embedder_conformance import EmbedderConformance


class _FakeEmbedding:
    def __init__(self, vec: list[float]) -> None:
        self.embedding = vec


class _FakeResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbedding(v) for v in vectors]


def _build_fake_client(dim: int = 8) -> MagicMock:
    client = MagicMock(name="AsyncOpenAI")

    async def fake_create(*, model: str, input: list[str]) -> _FakeResponse:  # noqa: A002
        return _FakeResponse([[float((i + 1) * 0.1)] * dim for i in range(len(input))])

    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=fake_create)
    client.close = AsyncMock()
    return client


def _patch_async_openai(client: MagicMock) -> Any:
    return patch(
        "rag.adapters.openai.embedder.AsyncOpenAI",
        return_value=client,
    )


class TestOpenAIEmbedderConformance(EmbedderConformance):
    @pytest.fixture
    async def embedder(self):  # type: ignore[no-untyped-def]
        client = _build_fake_client()
        with _patch_async_openai(client):
            async with OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small") as e:
                yield e


class TestOpenAIEmbedderUnit:
    async def test_constructor_rejects_empty_api_key(self) -> None:
        with pytest.raises(ConfigurationError):
            OpenAIEmbedder(api_key="")

    async def test_embed_outside_context_raises(self) -> None:
        emb = OpenAIEmbedder(api_key="sk-test")
        with pytest.raises(ConfigurationError):
            await emb.embed_query("hi")

    async def test_close_called_on_exit(self) -> None:
        client = _build_fake_client()
        with _patch_async_openai(client):
            async with OpenAIEmbedder(api_key="sk-test"):
                pass
        client.close.assert_awaited_once()

    async def test_embed_passes_chunk_contents(self) -> None:
        client = _build_fake_client()
        chunks = make_chunks(3)
        with _patch_async_openai(client):
            async with OpenAIEmbedder(api_key="sk-test") as e:
                result = await e.embed(chunks)
        assert len(result) == 3
        call = client.embeddings.create.await_args
        assert call.kwargs["input"] == [c.content for c in chunks]
        assert call.kwargs["model"] == "text-embedding-3-small"

    async def test_embed_empty_skips_api_call(self) -> None:
        client = _build_fake_client()
        with _patch_async_openai(client):
            async with OpenAIEmbedder(api_key="sk-test") as e:
                result = await e.embed([])
        assert result == []
        client.embeddings.create.assert_not_called()

    async def test_rate_limit_maps_to_rag_error(self) -> None:
        client = _build_fake_client()
        client.embeddings.create = AsyncMock(
            side_effect=openai.RateLimitError(
                "slow down", response=MagicMock(), body=None
            )
        )
        with _patch_async_openai(client):
            async with OpenAIEmbedder(api_key="sk-test") as e:
                with pytest.raises(RateLimitError):
                    await e.embed_query("hi")

    async def test_auth_error_maps_to_configuration_error(self) -> None:
        client = _build_fake_client()
        client.embeddings.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                "bad key", response=MagicMock(), body=None
            )
        )
        with _patch_async_openai(client):
            async with OpenAIEmbedder(api_key="sk-test") as e:
                with pytest.raises(ConfigurationError):
                    await e.embed_query("hi")

    async def test_connection_error_maps_to_backend_error(self) -> None:
        client = _build_fake_client()
        client.embeddings.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )
        with _patch_async_openai(client):
            async with OpenAIEmbedder(api_key="sk-test") as e:
                with pytest.raises(BackendCommunicationError):
                    await e.embed_query("hi")
