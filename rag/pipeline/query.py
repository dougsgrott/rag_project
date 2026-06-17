"""Query pipeline.

Composes the stages defined in `CONTEXT.md`:

    get_prompt → get_history → rewrite → search → rerank → generate → append

The Orchestration Layer is backend-agnostic — adapters are injected.

Two entry points:
- `execute_query` — runs the full pipeline and returns `(answer, context)`
  without persisting the turn. Used by `evaluate.py` so a test set never
  pollutes the live conversation history.
- `answer_query` — calls `execute_query` and appends the `(user, assistant)`
  pair to the conversation store. Used by the UI and the live `query` CLI.
"""

from rag.stages.conversation_store import ConversationStore
from rag.stages.generator import Generator
from rag.stages.prompt_store import PromptStore
from rag.stages.query_rewriter import QueryRewriter
from rag.stages.reranker import Reranker
from rag.stages.vector_store import VectorStore
from rag.types import Message, SearchResult

__all__ = ["execute_query", "answer_query"]


async def execute_query(
    *,
    prompt_store: PromptStore,
    conversation_store: ConversationStore,
    query_rewriter: QueryRewriter,
    vector_store: VectorStore,
    reranker: Reranker,
    generator: Generator,
    conversation_id: str,
    domain: str,
    query: str,
    search_top_k: int = 150,
    final_top_k: int = 5,
) -> tuple[Message, list[SearchResult], list[SearchResult]]:
    """Run the query pipeline; return the answer, the wide candidate set, and
    the reranked context used.

    The pre-rerank `candidates` are returned alongside the final `context` so
    the evaluation path can score the retriever and the reranker as separate IR
    stages; the live path ignores them.

    Does *not* persist the turn — callers decide whether to append to the
    conversation store (the production path) or discard it (the evaluation
    path).
    """
    system_prompt = await prompt_store.get_prompt(domain)
    history = await conversation_store.get_history(conversation_id)
    rewritten = await query_rewriter.rewrite(query, history)
    candidates = await vector_store.search(rewritten, top_k=search_top_k)
    context = await reranker.rerank(rewritten, candidates, top_k=final_top_k)
    answer = await generator.generate(query, context, system_prompt, history)
    return answer, candidates, context


async def answer_query(
    *,
    prompt_store: PromptStore,
    conversation_store: ConversationStore,
    query_rewriter: QueryRewriter,
    vector_store: VectorStore,
    reranker: Reranker,
    generator: Generator,
    conversation_id: str,
    domain: str,
    query: str,
    search_top_k: int = 150,
    final_top_k: int = 5,
) -> Message:
    answer, _candidates, _context = await execute_query(
        prompt_store=prompt_store,
        conversation_store=conversation_store,
        query_rewriter=query_rewriter,
        vector_store=vector_store,
        reranker=reranker,
        generator=generator,
        conversation_id=conversation_id,
        domain=domain,
        query=query,
        search_top_k=search_top_k,
        final_top_k=final_top_k,
    )
    await conversation_store.append_message(
        conversation_id, Message(role="user", content=query)
    )
    await conversation_store.append_message(conversation_id, answer)
    return answer
