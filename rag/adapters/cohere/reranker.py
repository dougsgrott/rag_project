"""Cohere Rerank API adapter.

Sits between VectorStore.search() (wide candidate set, e.g. top 150) and
Generator.generate() (tight context, e.g. top 5). The Cross-Encoder model
scores candidates by semantic relevance to the query rather than vector
distance, which typically yields a much higher-quality final set.

Stack-agnostic only in the sense that the constructor takes a raw API key —
no RAG pipeline abstraction is leaked here. Inject a CohereReranker into
compose.py in place of NoOpReranker with no other pipeline changes required.
"""

from __future__ import annotations

from types import TracebackType

import cohere

from rag.errors import BackendCommunicationError, ConfigurationError, RateLimitError
from rag.stages.reranker import Reranker
from rag.types import SearchResult

__all__ = ["CohereReranker"]


class CohereReranker(Reranker):
    """Cohere Rerank Cross-Encoder adapter.

    Behaviour:
    - Empty candidates → returns [] without calling the API.
    - top_k is capped at len(candidates) before the API call so the SDK
      never receives top_n > len(documents).
    - Returned SearchResult.score reflects the Cohere relevance score
      (0–1 range), not the original vector similarity score.
    - Use as an async context manager; lifecycle delegates to the cohere
      SDK's own __aenter__/__aexit__ so the underlying HTTP session is
      closed cleanly (ADR-0007).
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-english-v3.0",
    ) -> None:
        if not api_key:
            raise ConfigurationError("CohereReranker requires a non-empty api_key")
        self._api_key = api_key
        self._model = model
        self._client: cohere.AsyncClientV2 | None = None

    async def __aenter__(self) -> "CohereReranker":
        self._client = cohere.AsyncClientV2(api_key=self._api_key)
        await self._client.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc, tb)
            self._client = None

    async def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []

        client = self._client
        if client is None:
            raise ConfigurationError(
                "CohereReranker used outside its async context manager"
            )

        effective_top_k = min(top_k, len(candidates))
        documents = [r.chunk.content for r in candidates]

        try:
            response = await client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=effective_top_k,
            )
        except cohere.TooManyRequestsError as e:
            raise RateLimitError(str(e)) from e
        except (cohere.UnauthorizedError, cohere.ForbiddenError) as e:
            raise ConfigurationError(str(e)) from e
        except Exception as e:
            raise BackendCommunicationError(str(e)) from e

        return [
            SearchResult(
                chunk=candidates[item.index].chunk,
                score=item.relevance_score,
            )
            for item in response.results
        ]
