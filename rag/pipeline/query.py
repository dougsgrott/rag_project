"""Query pipeline.

Composes the stages defined in `CONTEXT.md`:

    get_prompt → get_history → rewrite → search → rerank → generate → append

The Orchestration Layer is backend-agnostic — adapters are injected.
"""

from rag.stages.conversation_store import ConversationStore
from rag.stages.generator import Generator
from rag.stages.prompt_store import PromptStore
from rag.stages.query_rewriter import QueryRewriter
from rag.stages.reranker import Reranker
from rag.stages.vector_store import VectorStore
from rag.types import Message

__all__ = ["answer_query"]


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
    system_prompt = await prompt_store.get_prompt(domain)
    history = await conversation_store.get_history(conversation_id)
    rewritten = await query_rewriter.rewrite(query, history)
    candidates = await vector_store.search(rewritten, top_k=search_top_k)
    context = await reranker.rerank(rewritten, candidates, top_k=final_top_k)
    answer = await generator.generate(query, context, system_prompt, history)
    await conversation_store.append_message(
        conversation_id, Message(role="user", content=query)
    )
    await conversation_store.append_message(conversation_id, answer)
    return answer
