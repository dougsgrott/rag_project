from abc import ABC, abstractmethod

from rag.types import Message

__all__ = ["QueryRewriter"]


class QueryRewriter(ABC):
    """Reformulates a raw user query into a self-contained search query.

    Sees prior conversation turns so it can resolve references like "what
    about the other one?".
    """

    @abstractmethod
    async def rewrite(self, query: str, history: list[Message]) -> str:
        ...
