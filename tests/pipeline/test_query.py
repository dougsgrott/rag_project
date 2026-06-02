"""End-to-end test for the query orchestration.

Uses stub stages so no external service is touched. Verifies call order,
that the rewriter receives history, that context flows from search through
rerank into generate, and that both turns are persisted.
"""

from dataclasses import dataclass, field

import pytest

from rag.errors import ConfigurationError
from rag.pipeline.query import answer_query
from rag.stages.conversation_store import ConversationStore
from rag.stages.generator import Generator
from rag.stages.prompt_store import PromptStore
from rag.stages.query_rewriter import QueryRewriter
from rag.stages.reranker import Reranker
from rag.stages.vector_store import VectorStore
from rag.types import Message, SearchResult

from tests.stages._helpers import make_search_results


@dataclass
class _Recorder:
    calls: list[tuple] = field(default_factory=list)


class _StubPromptStore(PromptStore):
    def __init__(self, recorder: _Recorder, prompt: str = "ground answers in context") -> None:
        self._r = recorder
        self._prompt = prompt

    async def get_prompt(self, domain: str) -> str:
        self._r.calls.append(("get_prompt", domain))
        return self._prompt

    async def save_prompt(self, domain: str, prompt: str, author: str) -> None:
        raise NotImplementedError


class _StubConversationStore(ConversationStore):
    def __init__(self, recorder: _Recorder, history: list[Message] | None = None) -> None:
        self._r = recorder
        self._history: dict[str, list[Message]] = {"conv": list(history or [])}

    async def get_history(self, conversation_id: str) -> list[Message]:
        self._r.calls.append(("get_history", conversation_id))
        return list(self._history.get(conversation_id, []))

    async def append_message(self, conversation_id: str, message: Message) -> None:
        self._r.calls.append(("append_message", conversation_id, message))
        self._history.setdefault(conversation_id, []).append(message)

    async def list_conversations(self) -> list[str]:
        self._r.calls.append(("list_conversations",))
        return list(self._history.keys())


class _StubRewriter(QueryRewriter):
    def __init__(self, recorder: _Recorder, prefix: str = "rewritten:") -> None:
        self._r = recorder
        self._prefix = prefix

    async def rewrite(self, query: str, history: list[Message]) -> str:
        self._r.calls.append(("rewrite", query, tuple(history)))
        return f"{self._prefix}{query}"


class _StubVectorStore(VectorStore):
    def __init__(self, recorder: _Recorder, results: list[SearchResult]) -> None:
        self._r = recorder
        self._results = results

    async def index(self, chunks):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        self._r.calls.append(("search", query, top_k))
        return list(self._results[:top_k])


class _StubReranker(Reranker):
    def __init__(self, recorder: _Recorder) -> None:
        self._r = recorder

    async def rerank(self, query, candidates, top_k):  # type: ignore[no-untyped-def]
        self._r.calls.append(("rerank", query, len(candidates), top_k))
        return list(candidates[:top_k])


class _StubGenerator(Generator):
    def __init__(self, recorder: _Recorder, answer: str = "stub-answer") -> None:
        self._r = recorder
        self._answer = answer

    async def generate(self, query, context, system_prompt, history):  # type: ignore[no-untyped-def]
        self._r.calls.append(("generate", query, len(context), system_prompt, tuple(history)))
        return Message(role="assistant", content=self._answer)


async def _run(history: list[Message] | None = None, search_top_k: int = 4, final_top_k: int = 2):
    rec = _Recorder()
    candidates = make_search_results(5)
    prompt = _StubPromptStore(rec)
    conv = _StubConversationStore(rec, history=history)
    rewriter = _StubRewriter(rec)
    vstore = _StubVectorStore(rec, candidates)
    reranker = _StubReranker(rec)
    generator = _StubGenerator(rec)
    answer = await answer_query(
        prompt_store=prompt,
        conversation_store=conv,
        query_rewriter=rewriter,
        vector_store=vstore,
        reranker=reranker,
        generator=generator,
        conversation_id="conv",
        domain="default",
        query="what about alpha?",
        search_top_k=search_top_k,
        final_top_k=final_top_k,
    )
    return rec, conv, answer


async def test_returns_generator_message() -> None:
    rec, _, answer = await _run()
    assert answer == Message(role="assistant", content="stub-answer")


async def test_call_order_matches_data_flow_contract() -> None:
    rec, _, _ = await _run()
    names = [c[0] for c in rec.calls]
    assert names == [
        "get_prompt",
        "get_history",
        "rewrite",
        "search",
        "rerank",
        "generate",
        "append_message",
        "append_message",
    ]


async def test_rewriter_receives_history() -> None:
    prior = [Message("user", "earlier"), Message("assistant", "earlier reply")]
    rec, _, _ = await _run(history=prior)
    rewrite_call = next(c for c in rec.calls if c[0] == "rewrite")
    assert rewrite_call[2] == tuple(prior)


async def test_rewritten_query_is_what_search_and_rerank_see() -> None:
    rec, _, _ = await _run()
    search_call = next(c for c in rec.calls if c[0] == "search")
    rerank_call = next(c for c in rec.calls if c[0] == "rerank")
    assert search_call[1] == "rewritten:what about alpha?"
    assert rerank_call[1] == "rewritten:what about alpha?"


async def test_reranked_context_is_what_generator_receives() -> None:
    rec, _, _ = await _run(search_top_k=5, final_top_k=2)
    rerank_call = next(c for c in rec.calls if c[0] == "rerank")
    generate_call = next(c for c in rec.calls if c[0] == "generate")
    assert rerank_call[3] == 2  # top_k for reranker
    assert generate_call[2] == 2  # context length seen by generator


async def test_both_turns_appended_to_history() -> None:
    rec, conv, _ = await _run()
    appended = [c for c in rec.calls if c[0] == "append_message"]
    assert len(appended) == 2
    assert appended[0][2] == Message("user", "what about alpha?")
    assert appended[1][2].role == "assistant"
    assert conv._history["conv"][-2:] == [
        Message("user", "what about alpha?"),
        Message("assistant", "stub-answer"),
    ]


async def test_missing_prompt_bubbles_up_as_configuration_error() -> None:
    rec = _Recorder()
    class _FailingPrompt(PromptStore):
        async def get_prompt(self, domain: str) -> str:
            raise ConfigurationError("no prompt for domain")
        async def save_prompt(self, *a, **kw): raise NotImplementedError

    with pytest.raises(ConfigurationError):
        await answer_query(
            prompt_store=_FailingPrompt(),
            conversation_store=_StubConversationStore(rec),
            query_rewriter=_StubRewriter(rec),
            vector_store=_StubVectorStore(rec, []),
            reranker=_StubReranker(rec),
            generator=_StubGenerator(rec),
            conversation_id="conv",
            domain="missing",
            query="q",
        )
