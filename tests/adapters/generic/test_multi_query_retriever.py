"""Tests for the `MultiQueryRetriever` VectorStore wrapper."""

from __future__ import annotations

from types import TracebackType
from typing import AsyncIterator

import pytest

from rag.adapters.generic.multi_query_retriever import MultiQueryRetriever
from rag.adapters.local.bm25_vector_store import BM25VectorStore
from rag.stages.generator import Generator
from rag.stages.vector_store import VectorStore
from rag.types import Chunk, Message, SearchResult

from tests.stages.vector_store_conformance import VectorStoreConformance


# --- Test doubles ----------------------------------------------------------


class _VariantGenerator(Generator):
    """Returns a fixed (multi-line) block of query variants for every call."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str,
        history: list[Message],
    ) -> Message:
        self.calls += 1
        return Message("assistant", self._text)


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


class TestMultiQueryRetrieverConformance(VectorStoreConformance):
    @pytest.fixture
    async def store(self) -> AsyncIterator[MultiQueryRetriever]:  # type: ignore[override]
        async with MultiQueryRetriever(
            inner=BM25VectorStore(),
            generator=_VariantGenerator("rephrased one\nrephrased two"),
        ) as s:
            yield s


# --- Unit ------------------------------------------------------------------


class TestMultiQueryRetrieverUnit:
    async def test_searches_original_plus_each_variant(self) -> None:
        gen = _VariantGenerator("q1\nq2\nq3")
        inner = _MappingStore()
        async with MultiQueryRetriever(inner=inner, generator=gen, n_queries=3) as r:
            await r.search("orig", top_k=5)
        assert set(inner.searched) == {"orig", "q1", "q2", "q3"}
        assert gen.calls == 1

    async def test_rrf_promotes_chunk_found_by_multiple_variants(self) -> None:
        # A appears only for the original; B appears for both q1 and q2, so its
        # summed reciprocal ranks outrank A's single contribution.
        inner = _MappingStore(
            {"orig": [_result(0)], "q1": [_result(1)], "q2": [_result(1)]}
        )
        gen = _VariantGenerator("q1\nq2")
        async with MultiQueryRetriever(inner=inner, generator=gen, n_queries=2) as r:
            results = await r.search("orig", top_k=10)
        assert [res.chunk.position for res in results] == [1, 0]  # B, A

    async def test_caps_variants_to_n_queries(self) -> None:
        gen = _VariantGenerator("q1\nq2\nq3\nq4\nq5")
        inner = _MappingStore()
        async with MultiQueryRetriever(inner=inner, generator=gen, n_queries=2) as r:
            await r.search("orig", top_k=5)
        # original + exactly 2 variants
        assert set(inner.searched) == {"orig", "q1", "q2"}

    async def test_strips_list_markers_from_variants(self) -> None:
        gen = _VariantGenerator("1. first\n2. second")
        inner = _MappingStore()
        async with MultiQueryRetriever(inner=inner, generator=gen, n_queries=2) as r:
            await r.search("orig", top_k=5)
        assert set(inner.searched) == {"orig", "first", "second"}

    async def test_deduplicates_variants_and_original(self) -> None:
        gen = _VariantGenerator("dup\ndup\norig")
        inner = _MappingStore()
        async with MultiQueryRetriever(inner=inner, generator=gen, n_queries=4) as r:
            await r.search("orig", top_k=5)
        assert sorted(inner.searched) == ["dup", "orig"]

    async def test_whitespace_generation_falls_back_to_original(self) -> None:
        gen = _VariantGenerator("   \n  \n")
        inner = _MappingStore({"orig": [_result(0)]})
        async with MultiQueryRetriever(inner=inner, generator=gen, n_queries=3) as r:
            results = await r.search("orig", top_k=5)
        assert inner.searched == ["orig"]  # never an empty string
        assert len(results) == 1

    async def test_include_original_false_omits_original(self) -> None:
        gen = _VariantGenerator("q1\nq2")
        inner = _MappingStore()
        async with MultiQueryRetriever(
            inner=inner, generator=gen, n_queries=2, include_original=False
        ) as r:
            await r.search("orig", top_k=5)
        assert "orig" not in inner.searched
        assert set(inner.searched) == {"q1", "q2"}

    async def test_include_original_false_still_falls_back_when_no_variants(self) -> None:
        gen = _VariantGenerator("")
        inner = _MappingStore({"orig": [_result(0)]})
        async with MultiQueryRetriever(
            inner=inner, generator=gen, n_queries=2, include_original=False
        ) as r:
            results = await r.search("orig", top_k=5)
        assert inner.searched == ["orig"]
        assert len(results) == 1

    async def test_top_k_zero_short_circuits_before_generating(self) -> None:
        gen = _VariantGenerator("q1\nq2")
        inner = _MappingStore()
        async with MultiQueryRetriever(inner=inner, generator=gen) as r:
            assert await r.search("orig", top_k=0) == []
        assert gen.calls == 0
        assert inner.searched == []

    async def test_index_forwards_to_inner(self) -> None:
        inner = _MappingStore()
        chunks = [_result(i).chunk for i in range(3)]
        async with MultiQueryRetriever(
            inner=inner, generator=_VariantGenerator("q1")
        ) as r:
            await r.index(chunks)
        assert inner.indexed == chunks

    async def test_delegates_lifecycle_to_inner_store(self) -> None:
        inner = _MappingStore()
        r = MultiQueryRetriever(inner=inner, generator=_VariantGenerator("q1"))
        assert not inner.entered
        async with r:
            assert inner.entered and not inner.exited
        assert inner.exited

    def test_rejects_non_positive_n_queries(self) -> None:
        with pytest.raises(ValueError):
            MultiQueryRetriever(
                inner=_MappingStore(), generator=_VariantGenerator("q"), n_queries=0
            )
