import asyncio
from pathlib import Path

from rag.errors import ConfigurationError
from rag.stages.document_loader import DocumentLoader
from rag.types import Document

__all__ = ["LocalFileSystemLoader"]

_DEFAULT_EXTENSIONS: tuple[str, ...] = (".txt", ".md")


class LocalFileSystemLoader(DocumentLoader):
    """Loads UTF-8 text documents from a local file or directory.

    `source` may point at a single file or a directory. Directories are
    traversed recursively; files matching `extensions` are loaded.
    """

    def __init__(self, *, extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS) -> None:
        self._extensions = tuple(e.lower() for e in extensions)

    async def load(self, source: str) -> list[Document]:
        return await asyncio.to_thread(self._load_sync, source)

    def _load_sync(self, source: str) -> list[Document]:
        path = Path(source).expanduser()
        if not path.exists():
            raise ConfigurationError(f"source path does not exist: {source}")

        files: list[Path]
        if path.is_file():
            files = [path]
        else:
            files = sorted(
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in self._extensions
            )

        documents: list[Document] = []
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    content=content,
                    source=str(file_path.resolve()),
                    metadata={"filename": file_path.name},
                )
            )
        return documents
