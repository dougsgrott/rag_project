import json
from pathlib import Path

import pytest

from rag.adapters.local.multihop_rag import (
    MultiHopRagDocumentLoader,
    load_multihop_cases,
)
from rag.errors import ConfigurationError

from tests.stages.document_loader_conformance import DocumentLoaderConformance

_CORPUS = [
    {
        "title": "Article One",
        "author": "Reporter A",
        "source": "The Verge",
        "published_at": "2023-09-28T12:00:00+00:00",
        "category": "technology",
        "url": "https://example.com/one",
        "body": "Body of article one.",
    },
    {
        "title": "Article Two",
        "author": "Reporter B",
        "source": "TechCrunch",
        "published_at": "2023-10-01T09:00:00+00:00",
        "category": "technology",
        "url": "https://example.com/two",
        "body": "Body of article two.",
    },
]

_QUERIES = [
    {
        "query": "Who links one and two?",
        "answer": "Someone",
        "question_type": "inference_query",
        "evidence_list": [
            {"url": "https://example.com/one", "fact": "f1"},
            {"url": "https://example.com/two", "fact": "f2"},
        ],
    },
    {
        "query": "An unanswerable question?",
        "answer": "Insufficient information.",
        "question_type": "null_query",
        "evidence_list": [],
    },
]


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    (tmp_path / "corpus.json").write_text(json.dumps(_CORPUS), encoding="utf-8")
    (tmp_path / "MultiHopRAG.json").write_text(json.dumps(_QUERIES), encoding="utf-8")
    return tmp_path


class TestMultiHopRagDocumentLoaderConformance(DocumentLoaderConformance):
    @pytest.fixture
    def loader(self) -> MultiHopRagDocumentLoader:
        return MultiHopRagDocumentLoader()

    @pytest.fixture
    def sample_source(self, dataset_dir: Path) -> str:
        return str(dataset_dir)


class TestMultiHopRagDocumentLoaderUnit:
    async def test_loads_corpus_from_directory(self, dataset_dir: Path) -> None:
        docs = await MultiHopRagDocumentLoader().load(str(dataset_dir))
        assert len(docs) == 2
        assert docs[0].content == "Body of article one."
        # The article URL is the canonical source id (joins with qrels).
        assert docs[0].source == "https://example.com/one"

    async def test_loads_corpus_from_direct_file(self, dataset_dir: Path) -> None:
        docs = await MultiHopRagDocumentLoader().load(str(dataset_dir / "corpus.json"))
        assert len(docs) == 2

    async def test_metadata_preserves_fields_and_renames_publisher(
        self, dataset_dir: Path
    ) -> None:
        docs = await MultiHopRagDocumentLoader().load(str(dataset_dir))
        meta = docs[0].metadata
        assert meta["title"] == "Article One"
        assert meta["author"] == "Reporter A"
        # The article's news outlet is stored as `publisher`, not `source`,
        # so it never collides with `Document.source`.
        assert meta["publisher"] == "The Verge"
        assert meta["category"] == "technology"
        assert meta["url"] == "https://example.com/one"

    async def test_missing_source_raises_configuration_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ConfigurationError):
            await MultiHopRagDocumentLoader().load(str(tmp_path / "nope"))

    async def test_record_missing_required_field_raises(self, tmp_path: Path) -> None:
        (tmp_path / "corpus.json").write_text(
            json.dumps([{"url": "u", "title": "no body"}]), encoding="utf-8"
        )
        with pytest.raises(ConfigurationError):
            await MultiHopRagDocumentLoader().load(str(tmp_path / "corpus.json"))


class TestLoadMultihopCases:
    def test_builds_cases_with_qrels_from_evidence(self, dataset_dir: Path) -> None:
        cases, _ = load_multihop_cases(str(dataset_dir))
        assert len(cases) == 2
        first = cases[0]
        assert first.query == "Who links one and two?"
        assert first.reference == "Someone"
        assert first.relevant_sources == {
            "https://example.com/one",
            "https://example.com/two",
        }
        assert first.has_qrels

    def test_null_query_has_no_qrels(self, dataset_dir: Path) -> None:
        cases, _ = load_multihop_cases(str(dataset_dir))
        null_case = cases[1]
        # No evidence -> no qrels -> exercises abstention, not retrieval.
        assert null_case.relevant_sources is None
        assert not null_case.has_qrels

    def test_question_type_sidecar(self, dataset_dir: Path) -> None:
        _, question_types = load_multihop_cases(str(dataset_dir))
        assert question_types["Who links one and two?"] == "inference_query"
        assert question_types["An unanswerable question?"] == "null_query"

    def test_loads_from_direct_file(self, dataset_dir: Path) -> None:
        cases, _ = load_multihop_cases(str(dataset_dir / "MultiHopRAG.json"))
        assert len(cases) == 2

    def test_missing_source_raises_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            load_multihop_cases(str(tmp_path / "nope"))
