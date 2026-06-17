"""Postgres production-stack composition root.

Drop-in replacement for `compose.py`: identical pipeline shape, same NoOp/
Identity passthroughs for the advanced stages, same `SimpleStack` dataclass
— only the three storage adapters change:

    ChromaVectorStore         → PgVectorStore
    SQLiteConversationStore   → PostgresConversationStore
    SQLitePromptStore         → PostgresPromptStore

Reads `POSTGRES_URL` and `OPENAI_*` from settings/.env. The simple-stack
CLI handlers (`_run_ingest`, `_run_query`, `_run_evaluate`) are
stack-agnostic — they accept a builder callable — so this file only owns
wiring and the `set-prompt` shortcut.

CLI:

    python compose_postgres.py set-prompt <domain> <prompt> [--author <name>]
    python compose_postgres.py ingest <path>
    python compose_postgres.py query <conversation_id> <query> [--domain <name>]
    python compose_postgres.py evaluate <test_set.json> [--domain <name>] [--ragas]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

import compose  # for the shared SimpleStack dataclass and CLI handlers
from rag.adapters.local.chunker import FixedSizeChunker
from rag.adapters.local.context_enricher import NoOpContextEnricher
from rag.adapters.local.document_loader import LocalFileSystemLoader
from rag.adapters.local.evaluator import NoOpEvaluator
from rag.adapters.local.query_rewriter import IdentityQueryRewriter
from rag.adapters.local.reranker import NoOpReranker
from rag.adapters.openai.embedder import OpenAIEmbedder
from rag.adapters.openai.generator import OpenAIGenerator
from rag.adapters.pgvector.vector_store import PgVectorStore
from rag.adapters.postgres.conversation_store import PostgresConversationStore
from rag.adapters.postgres.prompt_store import PostgresPromptStore
from rag.errors import ConfigurationError, RAGError
from rag.settings import Settings


@asynccontextmanager
async def build_postgres_stack(
    settings: Settings | None = None,
    *,
    table_prefix: str = "rag",
) -> AsyncIterator[compose.SimpleStack]:
    """Build the production stack — same `SimpleStack` shape, Postgres-backed."""
    settings = settings or Settings()
    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is required for the production stack")
    if not settings.postgres_url:
        raise ConfigurationError("POSTGRES_URL is required for the production stack")

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
        yield compose.SimpleStack(
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


# --- CLI -------------------------------------------------------------------
# Same subcommands as compose.py, but wired against build_postgres_stack.
# We monkey-patch the module-level `build_simple_stack` reference that
# compose's CLI handlers close over — keeps both files structurally
# identical without introducing a dependency-injection abstraction.


async def _run_set_prompt(domain: str, prompt: str, author: str) -> None:
    settings = Settings()
    if not settings.postgres_url:
        raise ConfigurationError("POSTGRES_URL is required for the production stack")
    async with PostgresPromptStore(
        dsn=settings.postgres_url, table_name="rag_prompts"
    ) as store:
        await store.save_prompt(domain, prompt, author)
    print(f"saved prompt for domain '{domain}' (author: {author})")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compose_postgres.py",
        description="Postgres production-stack RAG composition root.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Load + chunk + embed + index a directory of documents")
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
    p_eval.add_argument("--per-case", action="store_true")
    p_eval.add_argument(
        "--ragas",
        action="store_true",
        help="Score with RagasEvaluator instead of the NoOp (requires the 'ragas' group)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # Re-route compose's CLI helpers to the Postgres builder.
    compose.build_simple_stack = build_postgres_stack  # type: ignore[assignment]
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
                )
            )
    except RAGError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
