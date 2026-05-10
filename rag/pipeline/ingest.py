"""Ingest pipeline: load → chunk → enrich → index.

The Orchestration Layer for the ingest path. Stays free of any backend
specifics — adapters are passed in by the composition root (`compose.py`).
"""

from rag.stages.chunker import Chunker
from rag.stages.context_enricher import ContextEnricher
from rag.stages.document_loader import DocumentLoader
from rag.stages.vector_store import VectorStore
from rag.types import Chunk

__all__ = ["ingest_documents"]


async def ingest_documents(
    *,
    loader: DocumentLoader,
    chunker: Chunker,
    enricher: ContextEnricher,
    store: VectorStore,
    source: str,
) -> int:
    """Run the ingest pipeline against `source` and return the chunk count."""
    documents = await loader.load(source)
    all_chunks: list[Chunk] = []
    for document in documents:
        raw_chunks = await chunker.chunk(document)
        enriched = await enricher.enrich(document, raw_chunks)
        all_chunks.extend(enriched)
    if all_chunks:
        await store.index(all_chunks)
    return len(all_chunks)
