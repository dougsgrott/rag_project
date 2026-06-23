from __future__ import annotations

from types import TracebackType

import openai
from openai import AsyncOpenAI

from rag.errors import (
    BackendCommunicationError,
    ConfigurationError,
    RateLimitError,
)
from rag.stages.embedder import Embedder
from rag.types import Chunk, EmbeddedChunk

__all__ = ["OpenAIEmbedder"]

# The Embeddings API caps a single request at 300k tokens and 2,048 inputs.
# We batch under both, with margin on the token budget since we estimate token
# counts (~4 chars/token) rather than tokenize. Without this, indexing a large
# corpus sends every chunk in one request and the API rejects it (400).
_MAX_TOKENS_PER_BATCH = 200_000
_MAX_INPUTS_PER_BATCH = 2_048


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class OpenAIEmbedder(Embedder):
    """OpenAI Embeddings API adapter.

    Owns an `AsyncOpenAI` client; use as an async context manager so the
    underlying HTTP session is closed cleanly (see ADR-0007).
    """

    def __init__(self, *, api_key: str, model: str = "text-embedding-3-small") -> None:
        if not api_key:
            raise ConfigurationError("OpenAIEmbedder requires a non-empty api_key")
        self._api_key = api_key
        self._model = model
        self._client: AsyncOpenAI | None = None

    async def __aenter__(self) -> "OpenAIEmbedder":
        self._client = AsyncOpenAI(api_key=self._api_key)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        embedded: list[EmbeddedChunk] = []
        for batch in self._batches(chunks):
            response = await self._call(input_=[c.content for c in batch])
            embedded.extend(
                EmbeddedChunk(chunk=chunk, vector=list(item.embedding))
                for chunk, item in zip(batch, response.data, strict=True)
            )
        return embedded

    @staticmethod
    def _batches(chunks: list[Chunk]) -> list[list[Chunk]]:
        """Split chunks into request-sized batches under the API limits.

        Order is preserved; a single oversized chunk still goes out alone (the
        per-input limit is a chunker concern, not batching).
        """
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        tokens = 0
        for chunk in chunks:
            cost = _estimate_tokens(chunk.content)
            if current and (
                len(current) >= _MAX_INPUTS_PER_BATCH
                or tokens + cost > _MAX_TOKENS_PER_BATCH
            ):
                batches.append(current)
                current, tokens = [], 0
            current.append(chunk)
            tokens += cost
        if current:
            batches.append(current)
        return batches

    async def embed_query(self, query: str) -> list[float]:
        response = await self._call(input_=[query])
        return list(response.data[0].embedding)

    async def _call(self, *, input_: list[str]):  # type: ignore[no-untyped-def]
        client = self._client
        if client is None:
            raise ConfigurationError(
                "OpenAIEmbedder used outside its async context manager"
            )
        try:
            return await client.embeddings.create(model=self._model, input=input_)
        except openai.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except openai.AuthenticationError as e:
            raise ConfigurationError(str(e)) from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise BackendCommunicationError(str(e)) from e
