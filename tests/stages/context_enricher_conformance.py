"""Conformance tests every `ContextEnricher` adapter must pass.

Contract: count is preserved, and each output chunk's `document_source`
matches the corresponding input chunk. Adapters may augment `content`.
"""

import pytest

from rag.stages.context_enricher import ContextEnricher
from rag.types import Chunk

from tests.stages._helpers import make_chunks, make_document


class ContextEnricherConformance:
    @pytest.fixture
    def enricher(self) -> ContextEnricher:
        raise NotImplementedError("subclass must provide an `enricher` fixture")

    async def test_enrich_preserves_count_and_source(self, enricher: ContextEnricher) -> None:
        document = make_document()
        chunks = make_chunks(3, source=document.source)
        result = await enricher.enrich(document, chunks)
        assert isinstance(result, list)
        assert len(result) == len(chunks)
        for original, enriched in zip(chunks, result):
            assert isinstance(enriched, Chunk)
            assert enriched.document_source == original.document_source

    async def test_enrich_empty_chunks(self, enricher: ContextEnricher) -> None:
        result = await enricher.enrich(make_document(), [])
        assert result == []
