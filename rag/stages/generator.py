from abc import ABC, abstractmethod

from rag.types import Message, SearchResult

__all__ = ["Generator"]


class Generator(ABC):
    """Generates an answer message from query, retrieved context, and history.

    Stateless: conversation history is passed in by the Orchestration Layer
    after being fetched from the `ConversationStore`.
    """

    @abstractmethod
    async def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str,
        history: list[Message],
    ) -> Message:
        ...
