import pytest

from rag.adapters.local.context_enricher import NoOpContextEnricher

from tests.stages._helpers import make_chunks, make_document
from tests.stages.context_enricher_conformance import ContextEnricherConformance


class TestNoOpContextEnricher(ContextEnricherConformance):
    @pytest.fixture
    def enricher(self) -> NoOpContextEnricher:
        return NoOpContextEnricher()

    async def test_returns_input_chunks_unchanged(
        self, enricher: NoOpContextEnricher
    ) -> None:
        document = make_document()
        chunks = make_chunks(4, source=document.source)
        result = await enricher.enrich(document, chunks)
        assert result == chunks
