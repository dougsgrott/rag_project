from rag.stages.query_rewriter import QueryRewriter
from rag.types import Message

__all__ = ["IdentityQueryRewriter"]


class IdentityQueryRewriter(QueryRewriter):
    """Passthrough: returns the input query unchanged."""

    async def rewrite(self, query: str, history: list[Message]) -> str:
        return query
