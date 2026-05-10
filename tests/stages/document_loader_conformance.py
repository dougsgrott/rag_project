"""Conformance tests every `DocumentLoader` adapter must pass.

Subclass and provide `loader` and `sample_source` fixtures. The
`sample_source` must point at content the loader knows how to read (e.g. a
filesystem path the loader can traverse).
"""

import pytest

from rag.stages.document_loader import DocumentLoader
from rag.types import Document


class DocumentLoaderConformance:
    @pytest.fixture
    def loader(self) -> DocumentLoader:
        raise NotImplementedError("subclass must provide a `loader` fixture")

    @pytest.fixture
    def sample_source(self) -> str:
        raise NotImplementedError("subclass must provide a `sample_source` fixture")

    async def test_load_returns_list_of_documents(
        self, loader: DocumentLoader, sample_source: str
    ) -> None:
        docs = await loader.load(sample_source)
        assert isinstance(docs, list)
        assert len(docs) > 0
        for d in docs:
            assert isinstance(d, Document)
            assert isinstance(d.content, str) and d.content
            assert isinstance(d.source, str) and d.source
