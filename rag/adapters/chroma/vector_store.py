from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any

import chromadb

from rag.stages.embedder import Embedder
from rag.stages.vector_store import VectorStore
from rag.types import Chunk, SearchResult

__all__ = ["ChromaVectorStore"]


class ChromaVectorStore(VectorStore):
    """Chroma-backed VectorStore.

    Embedding is delegated to an injected `Embedder` (ADR-0005). When
    `persist_dir` is set, Chroma writes to disk; otherwise it runs entirely
    in-memory. Collection metadata pins distance to cosine.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        collection_name: str = "rag_documents",
        persist_dir: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._client: chromadb.api.ClientAPI | None = None
        self._collection: Any = None

    async def __aenter__(self) -> "ChromaVectorStore":
        self._client, self._collection = await asyncio.to_thread(self._init_client)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Chroma persists synchronously on write; ephemeral clients have no
        # state to flush. Drop references so a re-entered context starts clean.
        self._collection = None
        self._client = None

    async def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embedded = await self._embedder.embed(chunks)
        ids = [self._chunk_id(ec.chunk) for ec in embedded]
        documents = [ec.chunk.content for ec in embedded]
        embeddings = [ec.vector for ec in embedded]
        metadatas = [
            {"document_source": ec.chunk.document_source, "position": ec.chunk.position}
            for ec in embedded
        ]
        collection = self._require_collection()
        await asyncio.to_thread(
            collection.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query_vector = await self._embedder.embed_query(query)
        collection = self._require_collection()
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_vector],
            n_results=top_k,
        )
        return self._to_search_results(result)

    def _init_client(self) -> tuple[chromadb.api.ClientAPI, Any]:
        if self._persist_dir:
            client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return client, collection

    def _require_collection(self) -> Any:
        if self._collection is None:
            raise RuntimeError("ChromaVectorStore used outside its async context manager")
        return self._collection

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        return f"{chunk.document_source}::{chunk.position}"

    @staticmethod
    def _to_search_results(result: dict[str, Any]) -> list[SearchResult]:
        ids_batches = result.get("ids") or []
        if not ids_batches or not ids_batches[0]:
            return []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for content, meta, distance in zip(documents, metadatas, distances, strict=True):
            meta = meta or {}
            chunk = Chunk(
                content=content,
                document_source=str(meta.get("document_source", "")),
                position=int(meta.get("position", 0)),
                metadata={k: v for k, v in meta.items() if k not in {"document_source", "position"}},
            )
            # Chroma returns cosine *distance* in [0, 2]; convert to similarity.
            score = 1.0 - float(distance)
            results.append(SearchResult(chunk=chunk, score=score))
        return results
