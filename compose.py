"""Simple stack composition root.

Wires the OpenAI + ChromaDB + SQLite + local-filesystem stack with
passthrough advanced stages (NoOp/Identity). This file is the *only* place
adapters are constructed — the pipeline and UI never import adapters
directly (see ADR-0002).

CLI:

    python compose.py set-prompt <domain> <prompt> [--author <name>]
    python compose.py ingest <path>
    python compose.py query <conversation_id> <query> [--domain <name>]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from rag.adapters.chroma.vector_store import ChromaVectorStore
from rag.adapters.local.chunker import FixedSizeChunker
from rag.adapters.local.context_enricher import NoOpContextEnricher
from rag.adapters.local.document_loader import LocalFileSystemLoader
from rag.adapters.local.evaluator import NoOpEvaluator
from rag.adapters.local.query_rewriter import IdentityQueryRewriter
from rag.adapters.local.reranker import NoOpReranker
from rag.adapters.openai.embedder import OpenAIEmbedder
from rag.adapters.openai.generator import OpenAIGenerator
from rag.adapters.sqlite.conversation_store import SQLiteConversationStore
from rag.adapters.sqlite.prompt_store import SQLitePromptStore
from rag.errors import ConfigurationError, RAGError
from rag.pipeline.ingest import ingest_documents
from rag.pipeline.query import answer_query
from rag.settings import Settings
from rag.stages.chunker import Chunker
from rag.stages.context_enricher import ContextEnricher
from rag.stages.conversation_store import ConversationStore
from rag.stages.document_loader import DocumentLoader
from rag.stages.embedder import Embedder
from rag.stages.evaluator import Evaluator
from rag.stages.generator import Generator
from rag.stages.prompt_store import PromptStore
from rag.stages.query_rewriter import QueryRewriter
from rag.stages.reranker import Reranker
from rag.stages.vector_store import VectorStore


@dataclass
class SimpleStack:
    loader: DocumentLoader
    chunker: Chunker
    enricher: ContextEnricher
    embedder: Embedder
    vector_store: VectorStore
    query_rewriter: QueryRewriter
    reranker: Reranker
    generator: Generator
    conversation_store: ConversationStore
    prompt_store: PromptStore
    evaluator: Evaluator


@asynccontextmanager
async def build_simple_stack(
    settings: Settings | None = None,
    *,
    collection_name: str = "rag_documents",
) -> AsyncIterator[SimpleStack]:
    """Build and yield the fully-wired simple stack.

    Resource-owning adapters are entered through one `AsyncExitStack` so a
    single `async with` releases everything in reverse order on exit
    (ADR-0007). NoOp/Identity adapters are stateless and constructed inline.
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
        generator = await stack.enter_async_context(
            OpenAIGenerator(
                api_key=settings.openai_api_key,
                model=settings.openai_chat_model,
            )
        )
        conversation_store = await stack.enter_async_context(
            SQLiteConversationStore(path=settings.sqlite_path)
        )
        prompt_store = await stack.enter_async_context(
            SQLitePromptStore(path=settings.sqlite_path)
        )
        yield SimpleStack(
            loader=LocalFileSystemLoader(),
            chunker=FixedSizeChunker(),
            enricher=NoOpContextEnricher(),
            embedder=embedder,
            vector_store=vector_store,
            query_rewriter=IdentityQueryRewriter(),
            reranker=NoOpReranker(),
            generator=generator,
            conversation_store=conversation_store,
            prompt_store=prompt_store,
            evaluator=NoOpEvaluator(),
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


async def _run_query(
    conversation_id: str,
    query: str,
    *,
    domain: str,
    search_top_k: int,
    final_top_k: int,
) -> None:
    async with build_simple_stack() as s:
        answer = await answer_query(
            prompt_store=s.prompt_store,
            conversation_store=s.conversation_store,
            query_rewriter=s.query_rewriter,
            vector_store=s.vector_store,
            reranker=s.reranker,
            generator=s.generator,
            conversation_id=conversation_id,
            domain=domain,
            query=query,
            search_top_k=search_top_k,
            final_top_k=final_top_k,
        )
    print(answer.content)


async def _run_set_prompt(domain: str, prompt: str, author: str) -> None:
    settings = Settings()
    async with SQLitePromptStore(path=settings.sqlite_path) as store:
        await store.save_prompt(domain, prompt, author)
    print(f"saved prompt for domain '{domain}' (author: {author})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compose.py",
        description="Simple-stack RAG composition root (OpenAI + ChromaDB + SQLite).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Load + chunk + embed + index a directory of documents")
    p_ingest.add_argument("source", help="Path to a file or directory of .txt/.md documents")

    p_query = sub.add_parser("query", help="Ask a question against indexed documents")
    p_query.add_argument("conversation_id", help="Identifier for this multi-turn conversation")
    p_query.add_argument("query", help="The user question")
    p_query.add_argument("--domain", default="default", help="Domain whose system prompt to use")
    p_query.add_argument("--search-top-k", type=int, default=150)
    p_query.add_argument("--final-top-k", type=int, default=5)

    p_prompt = sub.add_parser("set-prompt", help="Save a new versioned system prompt for a domain")
    p_prompt.add_argument("domain")
    p_prompt.add_argument("prompt")
    p_prompt.add_argument("--author", default="cli")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "ingest":
            asyncio.run(_run_ingest(args.source))
        elif args.cmd == "query":
            asyncio.run(
                _run_query(
                    args.conversation_id,
                    args.query,
                    domain=args.domain,
                    search_top_k=args.search_top_k,
                    final_top_k=args.final_top_k,
                )
            )
        elif args.cmd == "set-prompt":
            asyncio.run(_run_set_prompt(args.domain, args.prompt, args.author))
    except RAGError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
