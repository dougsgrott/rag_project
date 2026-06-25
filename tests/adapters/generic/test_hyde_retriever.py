"""Tests for the `HyDERetriever` VectorStore wrapper."""

from __future__ import annotations

from types import TracebackType
from typing import AsyncIterator

import pytest

from rag.adapters.generic.hyde_retriever import HyDERetriever
from rag.adapters.local.bm25_vector_store import BM25VectorStore
from rag.stages.generator import Generator
from rag.stages.vector_store import VectorStore
from rag.types import Chunk, Message, SearchResult

from tests.stages.vector_store_conformance import VectorStoreConformance


# --- Test doubles ----------------------------------------------------------


class _HypothesisGenerator(Generator):
    """Returns a fixed hypothetical passage for every call."""

    def __init__(self, content: str = "a hypothetical answer passage") -> None:
        self._content = content
        self.calls = 0

    async def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str,
        history: list[Message],
    ) -> Message:
        self.calls += 1
        return Message("assistant", self._content)


class _MappingStore(VectorStore):
    """Inner store returning a preset result list per query; records calls."""

    def __init__(self, mapping: dict[str, list[SearchResult]] | None = None) -> None:
        self._mapping = mapping or {}
        self.searched: list[str] = []
        self.indexed: list[Chunk] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "_MappingStore":
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.exited = True

    async def index(self, chunks: list[Chunk]) -> None:
        self.indexed.extend(chunks)

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.searched.append(query)
        return list(self._mapping.get(query, [])[:top_k])


def _result(position: int, source: str = "d") -> SearchResult:
    return SearchResult(
        chunk=Chunk(content=f"c{position}", document_source=source, position=position),
        score=1.0,
    )


# --- Conformance -----------------------------------------------------------


class TestHyDERetrieverConformance(VectorStoreConformance):
    @pytest.fixture
    async def store(self) -> AsyncIterator[HyDERetriever]:  # type: ignore[override]
        async with HyDERetriever(
            inner=BM25VectorStore(),
            generator=_HypothesisGenerator("a drafted passage"),
        ) as s:
            yield s


# --- Unit ------------------------------------------------------------------


class TestHyDERetrieverUnit:
    async def test_searches_inner_with_hypothesis_not_raw_query(self) -> None:
        gen = _HypothesisGenerator("the hypothetical answer")
        inner = _MappingStore()
        async with HyDERetriever(inner=inner, generator=gen) as r:
            await r.search("the original question", top_k=5)
        assert inner.searched == ["the hypothetical answer"]
        assert "the original question" not in inner.searched
        assert gen.calls == 1

    async def test_empty_hypothesis_falls_back_to_raw_query(self) -> None:
        gen = _HypothesisGenerator("   \n  ")
        inner = _MappingStore({"orig": [_result(0)]})
        async with HyDERetriever(inner=inner, generator=gen) as r:
            results = await r.search("orig", top_k=5)
        assert inner.searched == ["orig"]  # never an empty string
        assert len(results) == 1

    async def test_fuse_mode_searches_both_query_and_hypothesis(self) -> None:
        gen = _HypothesisGenerator("hyp")
        inner = _MappingStore({"orig": [_result(0)], "hyp": [_result(1)]})
        async with HyDERetriever(
            inner=inner, generator=gen, fuse_with_query=True
        ) as r:
            results = await r.search("orig", top_k=10)
        assert set(inner.searched) == {"orig", "hyp"}
        assert {res.chunk.position for res in results} == {0, 1}

    async def test_fuse_mode_rrf_promotes_chunk_in_both_lists(self) -> None:
        gen = _HypothesisGenerator("hyp")
        # Chunk 1 (B) is returned by both the raw query and the hypothesis.
        inner = _MappingStore(
            {"orig": [_result(0), _result(1)], "hyp": [_result(1)]}
        )
        async with HyDERetriever(
            inner=inner, generator=gen, fuse_with_query=True
        ) as r:
            results = await r.search("orig", top_k=10)
        assert [res.chunk.position for res in results] == [1, 0]  # B, A

    async def test_fuse_mode_empty_hypothesis_searches_only_raw_query(self) -> None:
        gen = _HypothesisGenerator("")
        inner = _MappingStore({"orig": [_result(0)]})
        async with HyDERetriever(
            inner=inner, generator=gen, fuse_with_query=True
        ) as r:
            results = await r.search("orig", top_k=5)
        assert inner.searched == ["orig"]
        assert len(results) == 1

    async def test_top_k_zero_short_circuits_before_generating(self) -> None:
        gen = _HypothesisGenerator("hyp")
        inner = _MappingStore()
        async with HyDERetriever(inner=inner, generator=gen) as r:
            assert await r.search("orig", top_k=0) == []
        assert gen.calls == 0
        assert inner.searched == []

    async def test_index_forwards_to_inner(self) -> None:
        inner = _MappingStore()
        chunks = [_result(i).chunk for i in range(3)]
        async with HyDERetriever(
            inner=inner, generator=_HypothesisGenerator("hyp")
        ) as r:
            await r.index(chunks)
        assert inner.indexed == chunks

    async def test_delegates_lifecycle_to_inner_store(self) -> None:
        inner = _MappingStore()
        r = HyDERetriever(inner=inner, generator=_HypothesisGenerator("hyp"))
        assert not inner.entered
        async with r:
            assert inner.entered and not inner.exited
        assert inner.exited
