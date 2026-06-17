"""Tests for the Cohere Rerank adapter."""

from __future__ import annotations

from unittest.mock import patch

import cohere
import pytest

from rag.adapters.cohere.reranker import CohereReranker
from rag.errors import BackendCommunicationError, ConfigurationError, RateLimitError

from tests.stages._helpers import make_search_results
from tests.stages.reranker_conformance import RerankerConformance


# ---------------------------------------------------------------------------
# Fake Cohere SDK primitives
# ---------------------------------------------------------------------------


class _FakeResultItem:
    def __init__(self, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class _FakeRerankResponse:
    def __init__(self, results: list[_FakeResultItem]) -> None:
        self.results = results


class _FakeCohere:
    """Minimal stand-in for cohere.AsyncClientV2.

    Returns top_n results in descending score order, using sequential indices
    so every conformance call gets a valid, consistent response regardless of
    how many candidates are passed.
    """

    def __init__(self, *, raise_exc: Exception | None = None) -> None:
        self._raise_exc = raise_exc
        self.calls: list[dict] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "_FakeCohere":
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exited = True

    async def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **kwargs: object,
    ) -> _FakeRerankResponse:
        if self._raise_exc is not None:
            raise self._raise_exc
        self.calls.append(
            {"model": model, "query": query, "documents": documents, "top_n": top_n}
        )
        n = min(top_n or len(documents), len(documents))
        return _FakeRerankResponse(
            [_FakeResultItem(index=i, relevance_score=1.0 - 0.1 * i) for i in range(n)]
        )


# ---------------------------------------------------------------------------
# Conformance
# ---------------------------------------------------------------------------


class TestCohereRerankerConformance(RerankerConformance):
    @pytest.fixture
    async def reranker(self) -> CohereReranker:  # type: ignore[override]
        fake = _FakeCohere()
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            r = CohereReranker(api_key="test-key")
            async with r:
                yield r


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCohereRerankerUnit:
    async def test_scores_reflect_cohere_relevance_not_vector_score(self) -> None:
        fake = _FakeCohere()
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k") as r:
                candidates = make_search_results(3)
                # original scores are 1.0, 0.99, 0.98 — cohere returns 1.0, 0.9, 0.8
                results = await r.rerank("q", candidates, top_k=3)
        assert [round(res.score, 1) for res in results] == [1.0, 0.9, 0.8]

    async def test_results_ordered_by_relevance_score_descending(self) -> None:
        # Override: cohere returns results in best-first order (highest index
        # happens to be most relevant in our fake).
        class _ReverseCohere(_FakeCohere):
            async def rerank(self, *, documents, top_n, **kwargs) -> _FakeRerankResponse:  # type: ignore[override]
                n = min(top_n or len(documents), len(documents))
                # Deliberately reverse order: last doc gets highest score.
                items = [
                    _FakeResultItem(index=n - 1 - i, relevance_score=1.0 - 0.1 * i)
                    for i in range(n)
                ]
                self.calls.append({"top_n": top_n})
                return _FakeRerankResponse(items)

        fake = _ReverseCohere()
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k") as r:
                candidates = make_search_results(4)
                results = await r.rerank("q", candidates, top_k=3)
        # First result has highest score (1.0) and maps to candidate index 2.
        assert results[0].score == pytest.approx(1.0)
        assert results[0].chunk == candidates[2].chunk

    async def test_chunk_identity_preserved(self) -> None:
        fake = _FakeCohere()
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k") as r:
                candidates = make_search_results(5)
                results = await r.rerank("q", candidates, top_k=3)
        result_keys = {(res.chunk.document_source, res.chunk.position) for res in results}
        candidate_keys = {(c.chunk.document_source, c.chunk.position) for c in candidates}
        assert result_keys <= candidate_keys

    async def test_api_called_with_correct_shape(self) -> None:
        fake = _FakeCohere()
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k", model="rerank-multilingual-v3.0") as r:
                candidates = make_search_results(5)
                await r.rerank("test query", candidates, top_k=3)
        assert len(fake.calls) == 1
        call = fake.calls[0]
        assert call["model"] == "rerank-multilingual-v3.0"
        assert call["query"] == "test query"
        assert call["documents"] == [c.chunk.content for c in candidates]
        assert call["top_n"] == 3

    async def test_top_k_capped_at_candidate_count(self) -> None:
        """top_n passed to Cohere must never exceed len(documents)."""
        fake = _FakeCohere()
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k") as r:
                candidates = make_search_results(2)
                results = await r.rerank("q", candidates, top_k=100)
        assert fake.calls[0]["top_n"] == 2
        assert len(results) <= 2

    async def test_empty_candidates_skips_api_call(self) -> None:
        fake = _FakeCohere()
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k") as r:
                results = await r.rerank("q", [], top_k=5)
        assert results == []
        assert fake.calls == []

    async def test_rate_limit_error_mapped(self) -> None:
        exc = cohere.TooManyRequestsError(body="rate limited")
        fake = _FakeCohere(raise_exc=exc)
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k") as r:
                with pytest.raises(RateLimitError):
                    await r.rerank("q", make_search_results(2), top_k=1)

    async def test_unauthorized_error_mapped(self) -> None:
        exc = cohere.UnauthorizedError(body="bad key")
        fake = _FakeCohere(raise_exc=exc)
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k") as r:
                with pytest.raises(ConfigurationError):
                    await r.rerank("q", make_search_results(2), top_k=1)

    async def test_generic_error_mapped_to_backend_error(self) -> None:
        fake = _FakeCohere(raise_exc=RuntimeError("network blip"))
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            async with CohereReranker(api_key="k") as r:
                with pytest.raises(BackendCommunicationError):
                    await r.rerank("q", make_search_results(2), top_k=1)

    async def test_client_lifecycle_enter_exit(self) -> None:
        fake = _FakeCohere()
        with patch(
            "rag.adapters.cohere.reranker.cohere.AsyncClientV2", return_value=fake
        ):
            r = CohereReranker(api_key="k")
            assert r._client is None
            async with r:
                assert fake.entered
                assert r._client is not None
            assert fake.exited
            assert r._client is None

    async def test_rerank_outside_context_manager_raises(self) -> None:
        with patch("rag.adapters.cohere.reranker.cohere.AsyncClientV2"):
            r = CohereReranker(api_key="k")
            with pytest.raises(ConfigurationError, match="context manager"):
                await r.rerank("q", make_search_results(2), top_k=1)

    async def test_empty_api_key_raises_at_construction(self) -> None:
        with pytest.raises(ConfigurationError):
            CohereReranker(api_key="")
