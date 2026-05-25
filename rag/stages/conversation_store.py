from abc import ABC, abstractmethod

from rag.types import Message

__all__ = ["ConversationStore"]


class ConversationStore(ABC):
    """Persists and retrieves multi-turn `Message` history per conversation."""

    @abstractmethod
    async def get_history(self, conversation_id: str) -> list[Message]:
        ...

    @abstractmethod
    async def append_message(self, conversation_id: str, message: Message) -> None:
        ...

    @abstractmethod
    async def list_conversations(self) -> list[str]:
        """Return all known conversation IDs, ordered by most-recent activity first."""
        ...
