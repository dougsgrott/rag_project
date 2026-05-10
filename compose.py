"""Simple stack composition root.

Wires the OpenAI + ChromaDB + local-filesystem stack with passthrough
advanced stages. This file is the *only* place adapters are constructed —
the pipeline and UI never import adapters directly (see ADR-0002).

Run as a CLI to ingest a directory of documents:

    python compose.py ingest <path>
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from rag.adapters.chroma.vector_store import ChromaVectorStore
from rag.adapters.local.chunker import FixedSizeChunker
from rag.adapters.local.context_enricher import NoOpContextEnricher
from rag.adapters.local.document_loader import LocalFileSystemLoader
from rag.adapters.openai.embedder import OpenAIEmbedder
from rag.errors import ConfigurationError
from rag.pipeline.ingest import ingest_documents
from rag.settings import Settings
from rag.stages.chunker import Chunker
from rag.stages.context_enricher import ContextEnricher
from rag.stages.document_loader import DocumentLoader
from rag.stages.embedder import Embedder
from rag.stages.vector_store import VectorStore


@dataclass
class SimpleStack:
    loader: DocumentLoader
    chunker: Chunker
    enricher: ContextEnricher
    embedder: Embedder
    vector_store: VectorStore


@asynccontextmanager
async def build_simple_stack(
    settings: Settings | None = None,
    *,
    collection_name: str = "rag_documents",
) -> AsyncIterator[SimpleStack]:
    """Build and yield a fully-wired simple stack.

    Resource-owning adapters (`OpenAIEmbedder`, `ChromaVectorStore`) are
    entered through an `AsyncExitStack` so a single `async with` cleans up
    everything in reverse order on exit (ADR-0007).
    """
    settings = settings or Settings()
    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is required for the simple stack")

    async with AsyncExitStack() as stack:
        embedder = await stack.enter_async_context(
            OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
            )
        )
        vector_store = await stack.enter_async_context(
            ChromaVectorStore(
                embedder=embedder,
                collection_name=collection_name,
                persist_dir=settings.chroma_persist_dir,
            )
        )
        yield SimpleStack(
            loader=LocalFileSystemLoader(),
            chunker=FixedSizeChunker(),
            enricher=NoOpContextEnricher(),
            embedder=embedder,
            vector_store=vector_store,
        )


async def _run_ingest(source: str) -> None:
    async with build_simple_stack() as s:
        count = await ingest_documents(
            loader=s.loader,
            chunker=s.chunker,
            enricher=s.enricher,
            store=s.vector_store,
            source=source,
        )
    print(f"indexed {count} chunks from {source}")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in {"ingest"}:
        print("usage: python compose.py ingest <path>", file=sys.stderr)
        return 2
    if argv[1] == "ingest":
        if len(argv) < 3:
            print("usage: python compose.py ingest <path>", file=sys.stderr)
            return 2
        asyncio.run(_run_ingest(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
