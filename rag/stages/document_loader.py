from abc import ABC, abstractmethod

from rag.types import Document

__all__ = ["DocumentLoader"]


class DocumentLoader(ABC):
    """Loads source content into `Document` objects.

    `source` is a backend-specific locator: a filesystem path, a URI, or a
    Snowflake stage identifier. Adapters interpret it as needed.
    """

    @abstractmethod
    async def load(self, source: str) -> list[Document]:
        ...
