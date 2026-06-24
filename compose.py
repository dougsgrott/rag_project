"""Simple stack composition root.

Wires the OpenAI + ChromaDB + SQLite + local-filesystem stack with
passthrough advanced stages (NoOp/Identity). This file is the *only* place
adapters are constructed — the pipeline and UI never import adapters
directly (see ADR-0002).

CLI:

    python compose.py set-prompt <domain> <prompt> [--author <name>]
    python compose.py ingest <path>
    python compose.py query <conversation_id> <query> [--domain <name>]
    python compose.py evaluate <test_set.json> [--domain <name>] [--ragas]
    python compose.py evaluate-multihop [<dataset_dir>] [--limit N] [--ragas]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from rag.adapters.chroma.vector_store import ChromaVectorStore
from rag.adapters.local.chunker import FixedSizeChunker
from rag.adapters.local.context_enricher import NoOpContextEnricher
from rag.adapters.local.document_loader import LocalFileSystemLoader
from rag.adapters.local.evaluator import NoOpEvaluator
from rag.adapters.local.multihop_rag import (
    MultiHopRagDocumentLoader,
    load_multihop_cases,
)
from rag.adapters.local.query_rewriter import IdentityQueryRewriter
from rag.adapters.local.reranker import NoOpReranker
from rag.adapters.local.retrieval_evaluator import LocalRetrievalEvaluator
from rag.adapters.openai.embedder import OpenAIEmbedder
from rag.adapters.openai.generator import OpenAIGenerator
from rag.adapters.ragas.evaluator import RagasEvaluator
from rag.adapters.sqlite.conversation_store import SQLiteConversationStore
from rag.adapters.sqlite.prompt_store import SQLitePromptStore
from rag.errors import ConfigurationError, RAGError
from rag.pipeline.evaluate import (
    EvalCase,
    EvalRecord,
    EvaluationReport,
    aggregate,
    evaluate_test_set,
)
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
from rag.stages.retrieval_evaluator import RetrievalEvaluator
from rag.stages.vector_store import VectorStore
from rag.types import RetrievalResult


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
    retrieval_evaluator: RetrievalEvaluator


@asynccontextmanager
async def build_simple_stack(
    settings: Settings | None = None,
    *,
    collection_name: str = "rag_documents",
    persist_dir: str | None = None,
    use_hybrid: bool = False,
) -> AsyncIterator[SimpleStack]:
    """Build and yield the fully-wired simple stack.

    Resource-owning adapters are entered through one `AsyncExitStack` so a
    single `async with` releases everything in reverse order on exit
    (ADR-0007). NoOp/Identity adapters are stateless and constructed inline.

    `persist_dir` overrides the Chroma store location for this stack; when
    None the configured `settings.chroma_persist_dir` is used. This lets a
    caller isolate a dataset's index in its own file rather than sharing the
    default store.

    `use_hybrid` wraps the dense Chroma store and a BM25 store in a
    `HybridVectorStore` (technique A1) — dense + lexical retrieval fused with
    RRF. It is off by default so the plain Chroma path needs no `rank-bm25`; the
    BM25 dependency is imported lazily only when this is set.
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
        dense_store = ChromaVectorStore(
            embedder=embedder,
            collection_name=collection_name,
            persist_dir=persist_dir or settings.chroma_persist_dir,
        )
        vector_store: VectorStore
        if use_hybrid:
            # Lazy imports: the BM25 store pulls in `rank-bm25` (the `bm25`
            # dependency group), which the default dense-only path does not need.
            from rag.adapters.hybrid.vector_store import HybridVectorStore
            from rag.adapters.local.bm25_vector_store import BM25VectorStore

            vector_store = await stack.enter_async_context(
                HybridVectorStore(dense=dense_store, sparse=BM25VectorStore())
            )
        else:
            vector_store = await stack.enter_async_context(dense_store)
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
            # LocalRetrievalEvaluator is pure-Python and free, so unlike the
            # LLM-judge Evaluator (gated behind --ragas) it is the default; it
            # only scores cases that carry qrels.
            retrieval_evaluator=LocalRetrievalEvaluator(),
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


def _load_eval_cases(path: str) -> list[EvalCase]:
    """Load a test set from JSON.

    Format: a list of objects with `query` (required) and these optionals:
    `reference` (string or null — the gold answer, unlocks Context Recall and
    Answer Correctness), `relevant_sources` (list of document_source strings)
    and `relevant_chunks` (list of "<document_source>::<position>" chunk-ID
    strings). The latter two are the gold relevance labels (qrels) that unlock
    the retrieval metrics. Example:

        [
          {"query": "What is X?", "reference": "X is ...",
           "relevant_sources": ["x.md"]},
          {"query": "Tell me about Y",
           "relevant_chunks": ["y.md::3", "y.md::4"]}
        ]
    """
    import json
    from pathlib import Path

    text = Path(path).expanduser().read_text(encoding="utf-8")
    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ConfigurationError(
            f"{path}: expected a JSON list of cases, got {type(raw).__name__}"
        )

    def _str_set(item: dict[str, object], key: str, i: int) -> set[str] | None:
        value = item.get(key)
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigurationError(
                f"{path}[{i}]: `{key}` must be a list of strings"
            )
        return set(value)

    cases: list[EvalCase] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "query" not in item:
            raise ConfigurationError(
                f"{path}[{i}]: each case must be an object with a `query` field"
            )
        cases.append(
            EvalCase(
                query=item["query"],
                reference=item.get("reference"),
                relevant_sources=_str_set(item, "relevant_sources", i),
                relevant_chunks=_str_set(item, "relevant_chunks", i),
            )
        )
    return cases


async def _build_evaluator(
    extra: AsyncExitStack, *, default: Evaluator, use_ragas: bool
) -> Evaluator:
    """Return the `default` evaluator, or a RagasEvaluator when `--ragas` is set.

    The simple stack ships the NoOpEvaluator (zeros); `--ragas` swaps in the
    RAGAS-backed evaluator so the report carries real quality metrics without
    changing the rest of the stack (see issue #012a). The Ragas adapter is
    entered through `extra` so it is released with the rest of the stack.
    """
    if not use_ragas:
        return default
    settings = Settings()
    if not settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is required for --ragas")
    return await extra.enter_async_context(
        RagasEvaluator(
            api_key=settings.openai_api_key,
            llm_model=settings.openai_chat_model,
            embedding_model=settings.openai_embedding_model,
        )
    )


def _print_per_case(records: Sequence[EvalRecord]) -> None:
    for i, record in enumerate(records):
        print(f"--- case {i + 1}: {record.case.query!r}")
        print(f"  answer:  {record.answer.content[:200]}")

        def _fmt(v: float | None) -> str:
            return f"{v:.3f}" if v is not None else "—"

        print(
            f"  gen:     "
            f"F={record.result.faithfulness:.3f} "
            f"AR={record.result.answer_relevancy:.3f} "
            f"CP={record.result.context_precision:.3f} "
            f"CR={_fmt(record.result.context_recall)} "
            f"AC={_fmt(record.result.answer_correctness)}"
        )

        def _ret(r: RetrievalResult | None) -> str:
            if r is None:
                return "— (no qrels)"
            return (
                f"R@{r.k}={r.recall_at_k:.3f} P@{r.k}={r.precision_at_k:.3f} "
                f"MRR={r.mrr:.3f} nDCG={r.ndcg:.3f} Hit={r.hit_rate:.3f}"
            )

        print(f"  search:  {_ret(record.retrieval)}")
        print(f"  rerank:  {_ret(record.rerank)}")
        print()


def _print_by_question_type(
    report: EvaluationReport, question_types: dict[str, str]
) -> None:
    """Re-aggregate the report per `question_type` and print each slice.

    The categories (inference / comparison / temporal / null) fail in different
    ways — null_query cases carry no qrels, so their retrieval lines read
    "no qrels" — and one averaged number would hide that. Grouping by the
    sidecar keeps the segments visible.
    """
    by_type: dict[str, list[EvalRecord]] = {}
    for record in report.records:
        question_type = question_types.get(record.case.query, "(unknown)")
        by_type.setdefault(question_type, []).append(record)

    print("\n=== by question_type ===")
    for question_type in sorted(by_type):
        records = by_type[question_type]
        print(f"\n[{question_type}] {len(records)} cases")
        print(aggregate(records))


async def _run_evaluate(
    cases_path: str,
    *,
    domain: str,
    search_top_k: int,
    final_top_k: int,
    per_case: bool,
    use_ragas: bool = False,
    retrieval_k: int = 10,
    output_file: str | None = None,
) -> None:
    cases = _load_eval_cases(cases_path)
    settings = Settings()
    async with build_simple_stack() as s, AsyncExitStack() as extra:
        evaluator = await _build_evaluator(
            extra, default=s.evaluator, use_ragas=use_ragas
        )
        report = await evaluate_test_set(
            prompt_store=s.prompt_store,
            conversation_store=s.conversation_store,
            query_rewriter=s.query_rewriter,
            vector_store=s.vector_store,
            reranker=s.reranker,
            generator=s.generator,
            evaluator=evaluator,
            retrieval_evaluator=s.retrieval_evaluator,
            domain=domain,
            cases=cases,
            search_top_k=search_top_k,
            final_top_k=final_top_k,
            retrieval_k=retrieval_k,
        )
    if per_case:
        _print_per_case(report.records)
    print(report)

    # Optional report tracking hook for regular test sets
    if output_file:
        from rag.pipeline.report import save_markdown_report
        meta_config = {
            "domain": domain,
            "chat_model": settings.openai_chat_model,
            "embedding_model": settings.openai_embedding_model,
            "search_top_k": search_top_k,
            "final_top_k": final_top_k
        }
        save_markdown_report(report, output_file, meta_config=meta_config)
        print(f"Detailed evaluation metrics written safely to {output_file}")


async def _run_evaluate_multihop(
    dataset_dir: str,
    *,
    domain: str,
    search_top_k: int,
    final_top_k: int,
    per_case: bool,
    use_ragas: bool,
    retrieval_k: int,
    skip_index: bool,
    limit: int | None,
    persist_dir: str | None,
    use_hybrid: bool = False,
    output_file: str | None = None,
) -> None:
    """Index the MultiHop-RAG corpus, run the eval set, slice by question_type.

    Indexes into its own Chroma file (default ``<dataset_dir>/.chroma``, a
    `multihop_rag` collection) so the news corpus stays isolated from the
    default store. Indexing is idempotent-unfriendly (re-running re-embeds, and
    upsert overwrites rather than appends), so pass `--skip-index` on repeat
    runs once the corpus is already indexed.

    `use_hybrid` selects the A1 hybrid (dense + BM25) retriever. The BM25 index
    is in-memory and rebuilt per run, so `--skip-index` (which exists to avoid
    re-embedding the Chroma store) cannot apply: it is ignored under `--hybrid`,
    and a full ingest runs so both stores are populated.
    """
    from pathlib import Path
    from datetime import datetime

    settings = Settings()
    loader = MultiHopRagDocumentLoader()
    cases, question_types = load_multihop_cases(dataset_dir)
    if limit is not None:
        cases = cases[:limit]
    store_dir = persist_dir or str(Path(dataset_dir) / ".chroma")

    if use_hybrid and skip_index:
        print(
            "note: --skip-index is ignored under --hybrid "
            "(the in-memory BM25 index must be rebuilt); indexing the corpus"
        )
        skip_index = False

    async with (
        build_simple_stack(
            collection_name="multihop_rag",
            persist_dir=store_dir,
            use_hybrid=use_hybrid,
        ) as s,
        AsyncExitStack() as extra,
    ):
        if not skip_index:
            count = await ingest_documents(
                loader=loader,
                chunker=s.chunker,
                enricher=s.enricher,
                store=s.vector_store,
                source=dataset_dir,
            )
            print(f"indexed {count} chunks from {dataset_dir} into {store_dir}")

        evaluator = await _build_evaluator(
            extra, default=s.evaluator, use_ragas=use_ragas
        )
        report = await evaluate_test_set(
            prompt_store=s.prompt_store,
            conversation_store=s.conversation_store,
            query_rewriter=s.query_rewriter,
            vector_store=s.vector_store,
            reranker=s.reranker,
            generator=s.generator,
            evaluator=evaluator,
            retrieval_evaluator=s.retrieval_evaluator,
            domain=domain,
            cases=cases,
            search_top_k=search_top_k,
            final_top_k=final_top_k,
            retrieval_k=retrieval_k,
        )
    if per_case:
        _print_per_case(report.records)
    print(report)
    _print_by_question_type(report, question_types)

    # Markdown report logging logic
    if output_file:
        from rag.pipeline.report import save_markdown_report
        
        # If the user leaves it as default or passes a placeholder, resolve the timestamp
        if "{timestamp}" in output_file or output_file == "evaluation_report.md":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # We redirect to a dedicated 'reports' directory to avoid cluttering root
            output_path = Path("reports") / f"evaluation_report_{timestamp}.md"
        else:
            output_path = Path(output_file)

        meta_config = {
            "domain": domain,
            "chat_model": settings.openai_chat_model,
            "embedding_model": settings.openai_embedding_model,
            "search_top_k": search_top_k,
            "final_top_k": final_top_k
        }
        
        save_markdown_report(
            report, 
            output_path, 
            meta_config=meta_config, 
            question_types=question_types
        )
        print(f"Detailed Markdown evaluation metrics written to: {output_path}")


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

    p_eval = sub.add_parser("evaluate", help="Run the offline evaluation pipeline against a JSON test set")
    p_eval.add_argument("cases_path", help="Path to a JSON file with a list of {query, reference?} objects")
    p_eval.add_argument("--domain", default="default")
    p_eval.add_argument("--search-top-k", type=int, default=150)
    p_eval.add_argument("--final-top-k", type=int, default=5)
    p_eval.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="Cutoff k for retrieval metrics on the wide candidate set (qrels only)",
    )
    p_eval.add_argument("--per-case", action="store_true", help="Also print each case's result")
    p_eval.add_argument(
        "--ragas",
        action="store_true",
        help="Score with RagasEvaluator instead of the NoOp (requires the 'ragas' group)",
    )
    p_eval.add_argument("--output", default=None, help="Path to write out the Markdown log file")

    p_mh = sub.add_parser(
        "evaluate-multihop",
        help="Index the MultiHop-RAG corpus and evaluate, sliced by question_type",
    )
    p_mh.add_argument(
        "dataset_dir",
        nargs="?",
        default="datasets/multihop-rag",
        help="MultiHop-RAG dataset directory (with corpus.json + MultiHopRAG.json)",
    )
    p_mh.add_argument("--domain", default="default")
    p_mh.add_argument("--search-top-k", type=int, default=150)
    p_mh.add_argument("--final-top-k", type=int, default=5)
    p_mh.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="Cutoff k for retrieval metrics on the wide candidate set (qrels only)",
    )
    p_mh.add_argument("--per-case", action="store_true", help="Also print each case's result")
    p_mh.add_argument(
        "--ragas",
        action="store_true",
        help="Score with RagasEvaluator instead of the NoOp (requires the 'ragas' group)",
    )
    p_mh.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip corpus indexing (use when the collection is already populated)",
    )
    p_mh.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid dense+BM25 retrieval fused with RRF (technique A1). "
        "Forces a full re-index since the BM25 index is in-memory.",
    )
    p_mh.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N queries (cheap smoke run; full set is 2,556)",
    )
    p_mh.add_argument(
        "--persist-dir",
        default=None,
        help="Chroma store location (default: <dataset_dir>/.chroma, isolated "
        "from the default store)",
    )
    p_mh.add_argument(
        "--output", 
        default="evaluation_report.md", 
        help="Path to write the Markdown file. If default, it auto-resolves to 'reports/evaluation_report_<timestamp>.md'"
    )

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
        elif args.cmd == "evaluate":
            asyncio.run(
                _run_evaluate(
                    args.cases_path,
                    domain=args.domain,
                    search_top_k=args.search_top_k,
                    final_top_k=args.final_top_k,
                    per_case=args.per_case,
                    use_ragas=args.ragas,
                    retrieval_k=args.retrieval_k,
                    output_file=args.output
                )
            )
        elif args.cmd == "evaluate-multihop":
            asyncio.run(
                _run_evaluate_multihop(
                    args.dataset_dir,
                    domain=args.domain,
                    search_top_k=args.search_top_k,
                    final_top_k=args.final_top_k,
                    per_case=args.per_case,
                    use_ragas=args.ragas,
                    retrieval_k=args.retrieval_k,
                    skip_index=args.skip_index,
                    limit=args.limit,
                    persist_dir=args.persist_dir,
                    use_hybrid=args.hybrid,
                    output_file=args.output
                )
            )
    except RAGError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
