"""Local readers for the MultiHop-RAG dataset.

MultiHop-RAG (Tang & Yang, COLM 2024) ships two JSON files and no train/test
split — it is an evaluation-only benchmark over a single shared corpus:

- ``corpus.json``      — a list of 609 news articles.
- ``MultiHopRAG.json`` — a list of 2,556 queries, each with an ``answer``,
  a ``question_type`` and an ``evidence_list`` of supporting articles.

Two readers live here, both pure local-filesystem (no external dependency, so
this belongs in ``local/`` per the adapter layering rule):

- ``MultiHopRagDocumentLoader`` — a ``DocumentLoader`` over ``corpus.json`` for
  the ingest pipeline.
- ``load_multihop_cases`` — builds the offline-eval test set (``EvalCase`` list)
  from ``MultiHopRAG.json``, plus a ``question_type`` sidecar for slicing the
  report by query category.

Both readers key documents by their **article URL**. The URL is unique across
the corpus and every evidence reference resolves to one, so a case's
``relevant_sources`` (the qrels) line up exactly with the ``Document.source``
values the loader emits — which is what the retrieval metrics join on.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from rag.errors import ConfigurationError
from rag.pipeline.evaluate import EvalCase
from rag.stages.document_loader import DocumentLoader
from rag.types import Document

__all__ = ["MultiHopRagDocumentLoader", "load_multihop_cases"]

_CORPUS_FILENAME = "corpus.json"
_QUERIES_FILENAME = "MultiHopRAG.json"


def _resolve(source: str, default_filename: str) -> Path:
    """Resolve `source` to a file: use it directly if it is a file, else look
    for `default_filename` inside it if it is a directory.
    """
    path = Path(source).expanduser()
    if path.is_dir():
        path = path / default_filename
    if not path.is_file():
        raise ConfigurationError(f"source file does not exist: {path}")
    return path


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ConfigurationError(
            f"{path}: expected a JSON list, got {type(raw).__name__}"
        )
    return raw


def _require(record: dict[str, Any], key: str, path: Path, i: int) -> Any:
    if key not in record:
        raise ConfigurationError(f"{path}[{i}]: missing required field `{key}`")
    return record[key]


class MultiHopRagDocumentLoader(DocumentLoader):
    """Loads the MultiHop-RAG news corpus into `Document` objects.

    `source` may point at the dataset directory (in which case ``corpus.json``
    is read from it) or directly at the corpus JSON file. Each article becomes
    one `Document` whose `source` is the article URL; the remaining article
    fields are preserved in `metadata` (the article's news outlet is stored as
    `publisher` to avoid colliding with `Document.source`).
    """

    async def load(self, source: str) -> list[Document]:
        return await asyncio.to_thread(self._load_sync, source)

    def _load_sync(self, source: str) -> list[Document]:
        path = _resolve(source, _CORPUS_FILENAME)
        records = _load_json_list(path)
        documents: list[Document] = []
        for i, record in enumerate(records):
            url = _require(record, "url", path, i)
            body = _require(record, "body", path, i)
            documents.append(
                Document(
                    content=body,
                    source=url,
                    metadata={
                        "title": record.get("title"),
                        "author": record.get("author"),
                        "publisher": record.get("source"),
                        "published_at": record.get("published_at"),
                        "category": record.get("category"),
                        "url": url,
                    },
                )
            )
        return documents


def load_multihop_cases(source: str) -> tuple[list[EvalCase], dict[str, str]]:
    """Build the MultiHop-RAG offline-eval test set.

    `source` may point at the dataset directory (``MultiHopRAG.json`` is read
    from it) or directly at the queries JSON file.

    Returns ``(cases, question_type_by_query)``:

    - ``cases`` — one `EvalCase` per query. `reference` is the gold answer and
      `relevant_sources` is the set of supporting-article URLs (the qrels).
      ``null_query`` cases carry no evidence, so their `relevant_sources` is
      ``None`` (no qrels) — they exercise abstention, not retrieval.
    - ``question_type_by_query`` — maps each query string to its
      ``question_type`` (``inference_query`` / ``comparison_query`` /
      ``temporal_query`` / ``null_query``) so the evaluation report can be
      sliced per category. `EvalCase` carries no metadata field, so this rides
      alongside rather than inside it.
    """
    path = _resolve(source, _QUERIES_FILENAME)
    records = _load_json_list(path)

    cases: list[EvalCase] = []
    question_types: dict[str, str] = {}
    for i, record in enumerate(records):
        query = _require(record, "query", path, i)
        evidence = record.get("evidence_list") or []
        sources = {e["url"] for e in evidence if isinstance(e, dict) and "url" in e}
        cases.append(
            EvalCase(
                query=query,
                reference=record.get("answer"),
                relevant_sources=sources or None,
            )
        )
        question_type = record.get("question_type")
        if question_type is not None:
            question_types[query] = question_type
    return cases, question_types
