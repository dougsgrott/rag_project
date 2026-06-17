"""LLM-driven query rewriter.

Resolves references in follow-up questions ("what about the other one?",
"e quanto a esse?", "tell me more about that") by asking an injected
`Generator` to rewrite the raw query into a self-contained search query,
using the conversation history as context.

First-turn queries (empty history) skip the LLM call entirely — the
identity case is free and deterministic.

Stack-agnostic: the `Generator` is injected, so the simple stack supplies
`OpenAIGenerator`, the Cortex stack supplies `CortexGenerator`. No
backend-specific imports here — the Cortex stack uses this without needing
an OpenAI key (see ADR-0006).
"""

from __future__ import annotations

from rag.stages.generator import Generator
from rag.stages.query_rewriter import QueryRewriter
from rag.types import Message

__all__ = ["LLMQueryRewriter"]


_DEFAULT_SYSTEM_PROMPT = (
    "You rewrite conversational follow-up questions into self-contained search "
    "queries by resolving references (\"the other one\", \"that\", \"isso\", "
    "etc.) using prior conversation turns. Output only the rewritten query — "
    "no preamble, no explanation, no surrounding quotes."
)


_USER_TEMPLATE = """Conversation so far:
{history}

Current question: {query}

Rewrite the current question into a self-contained search query that does \
not require the conversation history to interpret. Preserve the original \
language. If the question is already self-contained, return it unchanged. \
Reply with the rewritten query only."""


class LLMQueryRewriter(QueryRewriter):
    """Reformulates follow-up queries via an injected `Generator`.

    Behaviour:

    - Empty history → returns the query verbatim. No LLM call is made.
    - Non-empty history → renders history as `<role>content</role>` lines
      into the user-turn template; sends it to the Generator with an empty
      `history=[]` and `context=[]` so the Generator treats it as a
      one-shot transform rather than a conversational continuation.
    - Whitespace-only Generator response → falls back to the original
      query (defensive — don't pass `""` to `VectorStore.search()`).
    """

    def __init__(
        self,
        *,
        generator: Generator,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._generator = generator
        self._system_prompt = system_prompt

    async def rewrite(self, query: str, history: list[Message]) -> str:
        if not history:
            return query
        answer = await self._generator.generate(
            query=_USER_TEMPLATE.format(
                history=self._format_history(history),
                query=query,
            ),
            context=[],
            system_prompt=self._system_prompt,
            history=[],
        )
        rewritten = answer.content.strip()
        return rewritten or query

    @staticmethod
    def _format_history(history: list[Message]) -> str:
        return "\n".join(f"<{m.role}>{m.content}</{m.role}>" for m in history)
