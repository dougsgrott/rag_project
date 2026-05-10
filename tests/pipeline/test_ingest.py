from pathlib import Path

import pytest

from rag.adapters.chroma.vector_store import ChromaVectorStore
from rag.adapters.local.chunker import FixedSizeChunker
from rag.adapters.local.context_enricher import NoOpContextEnricher
from rag.adapters.local.document_loader import LocalFileSystemLoader
from rag.pipeline.ingest import ingest_documents

from tests.stages._helpers import StubEmbedder


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    (tmp_path / "alpha.txt").write_text("a" * 64, encoding="utf-8")
    (tmp_path / "beta.txt").write_text("b" * 32, encoding="utf-8")
    return tmp_path


async def test_ingest_indexes_chunks(sample_dir: Path) -> None:
    async with ChromaVectorStore(
        embedder=StubEmbedder(), collection_name="ingest-pipeline"
    ) as store:
        count = await ingest_documents(
            loader=LocalFileSystemLoader(),
            chunker=FixedSizeChunker(chunk_size=16, overlap=4),
            enricher=NoOpContextEnricher(),
            store=store,
            source=str(sample_dir),
        )
        assert count > 0
        results = await store.search("anything", top_k=count)
    sources = {r.chunk.document_source for r in results}
    assert any(s.endswith("alpha.txt") for s in sources)
    assert any(s.endswith("beta.txt") for s in sources)
