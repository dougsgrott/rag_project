from __future__ import annotations

from types import TracebackType

import openai
from openai import AsyncOpenAI

from rag.errors import (
    BackendCommunicationError,
    ConfigurationError,
    GenerationError,
    RateLimitError,
)
from rag.stages.generator import Generator
from rag.types import Message, SearchResult

__all__ = ["OpenAIGenerator"]


class OpenAIGenerator(Generator):
    """OpenAI Chat Completions adapter.

    Owns an `AsyncOpenAI` client; use as an async context manager so the
    HTTP session is closed cleanly (ADR-0007). The retrieved context is
    concatenated into the final user turn; conversation history is replayed
    verbatim. The system prompt is supplied by the Orchestration Layer from
    the `PromptStore`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        if not api_key:
            raise ConfigurationError("OpenAIGenerator requires a non-empty api_key")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._client: AsyncOpenAI | None = None

    async def __aenter__(self) -> "OpenAIGenerator":
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

    async def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str,
        history: list[Message],
    ) -> Message:
        client = self._client
        if client is None:
            raise ConfigurationError(
                "OpenAIGenerator used outside its async context manager"
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": self._format_user_turn(query, context)})

        try:
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
        except openai.RateLimitError as e:
            raise RateLimitError(str(e)) from e
        except openai.AuthenticationError as e:
            raise ConfigurationError(str(e)) from e
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise BackendCommunicationError(str(e)) from e

        if not response.choices:
            raise GenerationError("OpenAI returned no choices")
        content = response.choices[0].message.content or ""
        return Message(role="assistant", content=content)

    @staticmethod
    def _format_user_turn(query: str, context: list[SearchResult]) -> str:
        if not context:
            return query
        passages = "\n\n".join(
            f"[{i + 1}] {r.chunk.content}" for i, r in enumerate(context)
        )
        return f"Context:\n{passages}\n\nQuestion: {query}"
