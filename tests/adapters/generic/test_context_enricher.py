"""Tests for the LLM-driven `LLMContextEnricher`."""

from __future__ import annotations

import asyncio

import pytest

from rag.adapters.generic.context_enricher import LLMContextEnricher
from rag.stages.generator import Generator
from rag.types import Chunk, Document, Message, SearchResult

from tests.stages._helpers import make_chunks, make_document
from tests.stages.context_enricher_conformance import ContextEnricherConformance


# --- Test doubles ----------------------------------------------------------


class _StaticGenerator(Generator):
    """Returns the same fixed summary for every call."""

    def __init__(self, summary: str = "stub-context") -> None:
        self._summary = summary

    async def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str,
        history: list[Message],
    ) -> Message:
        return Message("assistant", self._summary)


class _CapturingGenerator(Generator):
    """Records each call's arguments and returns a per-chunk summary."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str,
        history: list[Message],
    ) -> Message:
        self.calls.append(
            {
                "query": query,
                "context": context,
                "system_prompt": system_prompt,
                "history": history,
            }
        )
        # echo the chunk content back, prefixed, so tests can correlate calls
        # to chunks.
        marker = f"[ctx for {len(self.calls)}]"
        return Message("assistant", marker)


class _EmptyGenerator(Generator):
    async def generate(self, query, context, system_prompt, history):  # type: ignore[no-untyped-def]
        return Message("assistant", "   ")  # whitespace-only


class _ConcurrencyProbeGenerator(Generator):
    """Tracks max simultaneous in-flight calls so a test can assert the cap."""

    def __init__(self, *, delay: float = 0.02) -> None:
        self._delay = delay
        self._in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def generate(self, query, context, system_prompt, history):  # type: ignore[no-untyped-def]
        async with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self._delay)
        finally:
            async with self._lock:
                self._in_flight -= 1
        return Message("assistant", "ctx")


# --- Conformance -----------------------------------------------------------


class TestLLMContextEnricherConformance(ContextEnricherConformance):
    @pytest.fixture
    def enricher(self) -> LLMContextEnricher:
        return LLMContextEnricher(generator=_StaticGenerator())


# --- Unit ------------------------------------------------------------------


class TestLLMContextEnricherUnit:
    async def test_prepends_summary_to_chunk_content(self) -> None:
        enricher = LLMContextEnricher(generator=_StaticGenerator("Section 2.1 of report"))
        document = make_document(content="full doc")
        chunks = [Chunk(content="original chunk", document_source=document.source, position=0)]
        result = await enricher.enrich(document, chunks)
        assert len(result) == 1
        assert result[0].content == "Section 2.1 of report\n\noriginal chunk"
        assert result[0].document_source == document.source
        assert result[0].position == 0

    async def test_preserves_chunk_metadata_and_records_summary(self) -> None:
        enricher = LLMContextEnricher(generator=_StaticGenerator("brief ctx"))
        document = make_document()
        chunks = [
            Chunk(
                content="alpha",
                document_source=document.source,
                position=0,
                metadata={"section": "intro"},
            )
        ]
        result = await enricher.enrich(document, chunks)
        assert result[0].metadata["section"] == "intro"
        assert result[0].metadata["contextual_summary"] == "brief ctx"

    async def test_calls_generator_once_per_chunk(self) -> None:
        cap = _CapturingGenerator()
        enricher = LLMContextEnricher(generator=cap)
        document = make_document(content="big doc")
        chunks = make_chunks(4, source=document.source)
        await enricher.enrich(document, chunks)
        assert len(cap.calls) == 4

    async def test_prompt_includes_full_document_and_chunk(self) -> None:
        cap = _CapturingGenerator()
        enricher = LLMContextEnricher(generator=cap)
        document = Document(
            content="Annual SICRO report — full text here, including tables.",
            source="report.md",
        )
        chunks = [
            Chunk(content="Section A discusses pavement costs.", document_source="report.md", position=0),
            Chunk(content="Section B covers asphalt formulations.", document_source="report.md", position=1),
        ]
        await enricher.enrich(document, chunks)
        for call, expected_chunk in zip(cap.calls, chunks, strict=True):
            assert document.content in call["query"]
            assert expected_chunk.content in call["query"]
            # No retrieval context or history flows in — this is offline ingest.
            assert call["context"] == []
            assert call["history"] == []

    async def test_empty_chunk_list_returns_empty(self) -> None:
        enricher = LLMContextEnricher(generator=_StaticGenerator())
        result = await enricher.enrich(make_document(), [])
        assert result == []

    async def test_empty_summary_returns_chunk_unchanged(self) -> None:
        enricher = LLMContextEnricher(generator=_EmptyGenerator())
        original = Chunk(content="original", document_source="d", position=0)
        result = await enricher.enrich(make_document(), [original])
        assert result == [original]

    async def test_max_concurrency_caps_in_flight_calls(self) -> None:
        probe = _ConcurrencyProbeGenerator(delay=0.02)
        enricher = LLMContextEnricher(generator=probe, max_concurrency=3)
        await enricher.enrich(make_document(), make_chunks(10))
        assert probe.max_in_flight <= 3

    async def test_max_concurrency_one_is_sequential(self) -> None:
        probe = _ConcurrencyProbeGenerator(delay=0.01)
        enricher = LLMContextEnricher(generator=probe, max_concurrency=1)
        await enricher.enrich(make_document(), make_chunks(5))
        assert probe.max_in_flight == 1

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_concurrency_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            LLMContextEnricher(generator=_StaticGenerator(), max_concurrency=bad)

    async def test_works_with_any_generator_implementation(self) -> None:
        """No backend-specific isinstance checks — any Generator must work."""

        class _AnyGenerator(Generator):
            async def generate(self, query, context, system_prompt, history):  # type: ignore[no-untyped-def]
                return Message("assistant", "from any generator")

        enricher = LLMContextEnricher(generator=_AnyGenerator())
        document = make_document()
        chunks = make_chunks(2, source=document.source)
        result = await enricher.enrich(document, chunks)
        for r, orig in zip(result, chunks, strict=True):
            assert r.content == f"from any generator\n\n{orig.content}"
