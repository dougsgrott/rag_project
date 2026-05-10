"""Pipeline stage interfaces.

Eleven abstract base classes — one per stage in the ingest, query, and
evaluation pipelines. Backend Adapters in `rag/adapters/<backend>/` implement
exactly one of these interfaces. Method signatures match the data-flow
contracts in `CONTEXT.md` and must remain in lockstep with `rag/types.py`.

All methods are async (see ADR-0004).
"""

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

__all__ = [
    "Chunker",
    "ContextEnricher",
    "ConversationStore",
    "DocumentLoader",
    "Embedder",
    "Evaluator",
    "Generator",
    "PromptStore",
    "QueryRewriter",
    "Reranker",
    "VectorStore",
]
