from pathlib import Path

import pytest

from rag.adapters.local.document_loader import LocalFileSystemLoader
from rag.errors import ConfigurationError

from tests.stages.document_loader_conformance import DocumentLoaderConformance


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("alpha content", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.txt").write_text("gamma", encoding="utf-8")
    return tmp_path


class TestLocalFileSystemLoaderConformance(DocumentLoaderConformance):
    @pytest.fixture
    def loader(self) -> LocalFileSystemLoader:
        return LocalFileSystemLoader()

    @pytest.fixture
    def sample_source(self, docs_dir: Path) -> str:
        return str(docs_dir)


class TestLocalFileSystemLoaderUnit:
    async def test_directory_load_walks_recursively(self, docs_dir: Path) -> None:
        loader = LocalFileSystemLoader()
        docs = await loader.load(str(docs_dir))
        names = sorted(d.metadata["filename"] for d in docs)
        assert names == ["a.txt", "b.md", "c.txt"]

    async def test_directory_load_skips_unrecognised_extensions(
        self, docs_dir: Path
    ) -> None:
        loader = LocalFileSystemLoader()
        docs = await loader.load(str(docs_dir))
        assert all(not d.source.endswith(".bin") for d in docs)

    async def test_single_file_load(self, docs_dir: Path) -> None:
        loader = LocalFileSystemLoader()
        docs = await loader.load(str(docs_dir / "a.txt"))
        assert len(docs) == 1
        assert docs[0].content == "alpha content"
        assert docs[0].metadata["filename"] == "a.txt"

    async def test_missing_path_raises_configuration_error(self, tmp_path: Path) -> None:
        loader = LocalFileSystemLoader()
        with pytest.raises(ConfigurationError):
            await loader.load(str(tmp_path / "does-not-exist"))

    async def test_custom_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "page.html").write_text("<p>hi</p>", encoding="utf-8")
        loader = LocalFileSystemLoader(extensions=(".html",))
        docs = await loader.load(str(tmp_path))
        assert len(docs) == 1
        assert docs[0].metadata["filename"] == "page.html"
