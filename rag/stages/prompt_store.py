from abc import ABC, abstractmethod

__all__ = ["PromptStore"]


class PromptStore(ABC):
    """Stores versioned system prompts for a domain.

    `save_prompt` records `author` so the prompt history is auditable.
    `get_prompt` returns the latest prompt for the given domain.
    """

    @abstractmethod
    async def get_prompt(self, domain: str) -> str:
        ...

    @abstractmethod
    async def save_prompt(self, domain: str, prompt: str, author: str) -> None:
        ...
