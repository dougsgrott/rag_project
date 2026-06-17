"""Advanced-stack composition root.

The production storage layer of `compose_postgres.py` (PgVectorStore +
Postgres conversation/prompt stores + OpenAI embedder/generator), with every
advanced *processing* stage switched on:

    FixedSizeChunker        → StructureAwareChunker
    NoOpContextEnricher     → LLMContextEnricher   (injected OpenAIGenerator)
    IdentityQueryRewriter   → LLMQueryRewriter      (injected OpenAIGenerator)
    NoOpReranker            → CohereReranker
    NoOpEvaluator           → RagasEvaluator

The `RetrievalEvaluator` stays `LocalRetrievalEvaluator` (the default — it is
free and dependency-light), so the evaluation report carries both the RAGAS
generation metrics and the rank-aware retrieval metrics. This is the stack the
#012b benchmark compares against the simple stack (`compose.py`).

Reuses `compose.SimpleStack` and the stack-agnostic CLI handlers, exactly like
`compose_postgres.py` — only the wiring differs.

Requires `OPENAI_API_KEY`, `COHERE_API_KEY`, and `POSTGRES_URL`. Building the
stack constructs (but does not import) RAGAS; the heavy `ragas` import happens
lazily only when `evaluate` actually scores generation.

CLI:

    python compose_advanced.py set-prompt <domain> <prompt> [--author <name>]
    python compose_advanced.py ingest <path>
    python compose_advanced.py query <conversation_id> <query> [--domain <name>]
    python compose_advanced.py evaluate <test_set.json> [--domain <name>]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

import compose  # for the shared SimpleStack dataclass and CLI handlers
from rag.adapters.cohere.reranker import CohereReranker
from rag.adapters.generic.context_enricher import LLMContextEnricher
from rag.adapters.generic.query_rewriter import LLMQueryRewriter
from rag.adapters.local.chunker_structure_aware import StructureAwareChunker
from rag.adapters.local.document_loader import LocalFileSystemLoader
from rag.adapters.local.retrieval_evaluator import LocalRetrievalEvaluator
from rag.adapters.openai.embedder import OpenAIEmbedder
from rag.adapters.openai.generator import OpenAIGenerator
from rag.adapters.pgvector.vector_store import PgVectorStore
from rag.adapters.postgres.conversation_store import PostgresConversationStore
from rag.adapters.postgres.prompt_store import PostgresPromptStore
from rag.adapters.ragas.evaluator import RagasEvaluator
from rag.errors import ConfigurationError, RAGError
from rag.settings import Settings


@asynccontextmanager
async def build_advanced_stack(
    settings: Settings | None = None,
    *,
    table_prefix: str = "rag",
) -> AsyncIterator[compose.SimpleStack]:
    """Build the advanced stack — Postgres-backed, every advanced stage on."""
    settings = settings or Settings()
    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is required for the advanced stack")
    if not settings.postgres_url:
        raise ConfigurationError("POSTGRES_URL is required for the advanced stack")
    if not settings.cohere_api_key:
        raise ConfigurationError("COHERE_API_KEY is required for the advanced stack")

    async with AsyncExitStack() as stack:
        embedder = await stack.enter_async_context(
            OpenAIEmbedder(
                api_key=settings.openai_api_key,
                model=settings.openai_embedding_model,
            )
        )
        vector_store = await stack.enter_async_context(
            PgVectorStore(
                embedder=embedder,
                dsn=settings.postgres_url,
                dim=settings.openai_embedding_dim,
                table_name=f"{table_prefix}_chunks",
            )
        )
        generator = await stack.enter_async_context(
            OpenAIGenerator(
                api_key=settings.openai_api_key,
                model=settings.openai_chat_model,
            )
        )
        reranker = await stack.enter_async_context(
            CohereReranker(
                api_key=settings.cohere_api_key,
                model=settings.cohere_rerank_model,
            )
        )
        evaluator = await stack.enter_async_context(
            RagasEvaluator(
                api_key=settings.openai_api_key,
                llm_model=settings.openai_chat_model,
                embedding_model=settings.openai_embedding_model,
            )
        )
        conversation_store = await stack.enter_async_context(
            PostgresConversationStore(
                dsn=settings.postgres_url,
                table_name=f"{table_prefix}_messages",
            )
        )
        prompt_store = await stack.enter_async_context(
            PostgresPromptStore(
                dsn=settings.postgres_url,
                table_name=f"{table_prefix}_prompts",
            )
        )
        # The two LLM-driven processing stages share the live generator.
        yield compose.SimpleStack(
            loader=LocalFileSystemLoader(),
            chunker=StructureAwareChunker(),
            enricher=LLMContextEnricher(generator=generator),
            embedder=embedder,
            vector_store=vector_store,
            query_rewriter=LLMQueryRewriter(generator=generator),
            reranker=reranker,
            generator=generator,
            conversation_store=conversation_store,
            prompt_store=prompt_store,
            evaluator=evaluator,
            retrieval_evaluator=LocalRetrievalEvaluator(),
        )


# --- CLI -------------------------------------------------------------------
# Same subcommands as compose.py, wired against build_advanced_stack by
# re-pointing the module-level `build_simple_stack` that compose's CLI handlers
# close over (the pattern compose_postgres.py uses).


async def _run_set_prompt(domain: str, prompt: str, author: str) -> None:
    settings = Settings()
    if not settings.postgres_url:
        raise ConfigurationError("POSTGRES_URL is required for the advanced stack")
    async with PostgresPromptStore(
        dsn=settings.postgres_url, table_name="rag_prompts"
    ) as store:
        await store.save_prompt(domain, prompt, author)
    print(f"saved prompt for domain '{domain}' (author: {author})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compose_advanced.py",
        description="Advanced-stack RAG composition root (Postgres + all advanced stages).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Load + chunk + enrich + embed + index a directory")
    p_ingest.add_argument("source")

    p_query = sub.add_parser("query", help="Ask a question against indexed documents")
    p_query.add_argument("conversation_id")
    p_query.add_argument("query")
    p_query.add_argument("--domain", default="default")
    p_query.add_argument("--search-top-k", type=int, default=150)
    p_query.add_argument("--final-top-k", type=int, default=5)

    p_prompt = sub.add_parser("set-prompt", help="Save a new versioned system prompt for a domain")
    p_prompt.add_argument("domain")
    p_prompt.add_argument("prompt")
    p_prompt.add_argument("--author", default="cli")

    p_eval = sub.add_parser("evaluate", help="Run the offline evaluation pipeline against a JSON test set")
    p_eval.add_argument("cases_path")
    p_eval.add_argument("--domain", default="default")
    p_eval.add_argument("--search-top-k", type=int, default=150)
    p_eval.add_argument("--final-top-k", type=int, default=5)
    p_eval.add_argument("--retrieval-k", type=int, default=10)
    p_eval.add_argument("--per-case", action="store_true")
    # RagasEvaluator is already wired into this stack, so --ragas is redundant
    # here; kept for CLI parity with the other composition roots.
    p_eval.add_argument("--ragas", action="store_true", help=argparse.SUPPRESS)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Re-route compose's CLI helpers to the advanced builder.
    compose.build_simple_stack = build_advanced_stack  # type: ignore[assignment]
    try:
        if args.cmd == "ingest":
            asyncio.run(compose._run_ingest(args.source))
        elif args.cmd == "query":
            asyncio.run(
                compose._run_query(
                    args.conversation_id,
                    args.query,
                    domain=args.domain,
                    search_top_k=args.search_top_k,
                    final_top_k=args.final_top_k,
                )
            )
        elif args.cmd == "set-prompt":
            asyncio.run(_run_set_prompt(args.domain, args.prompt, args.author))
        elif args.cmd == "evaluate":
            asyncio.run(
                compose._run_evaluate(
                    args.cases_path,
                    domain=args.domain,
                    search_top_k=args.search_top_k,
                    final_top_k=args.final_top_k,
                    per_case=args.per_case,
                    use_ragas=args.ragas,
                    retrieval_k=args.retrieval_k,
                )
            )
    except RAGError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
