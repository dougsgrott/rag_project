"""Tests for the LLM-driven `LLMQueryRewriter`."""

from __future__ import annotations

import pytest

from rag.adapters.generic.query_rewriter import LLMQueryRewriter
from rag.stages.generator import Generator
from rag.types import Message, SearchResult

from tests.stages._helpers import make_message
from tests.stages.query_rewriter_conformance import QueryRewriterConformance


# --- Test doubles ----------------------------------------------------------


class _StaticGenerator(Generator):
    """Returns the same fixed response for every call."""

    def __init__(self, content: str = "rewritten query") -> None:
        self._content = content
        self.call_count = 0

    async def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str,
        history: list[Message],
    ) -> Message:
        self.call_count += 1
        return Message("assistant", self._content)


class _CapturingGenerator(Generator):
    """Records each call's arguments."""

    def __init__(self, content: str = "captured") -> None:
        self._content = content
        self.calls: list[dict] = []

    async def generate(
        self,
        query: str,
        context: list[SearchResult],
        system_prompt: str,
        history: list[Message],
    ) -> Message:
        self.calls.append(
            {
                "query": query,
                "context": context,
                "system_prompt": system_prompt,
                "history": history,
            }
        )
        return Message("assistant", self._content)


class _EmptyGenerator(Generator):
    async def generate(self, query, context, system_prompt, history):  # type: ignore[no-untyped-def]
        return Message("assistant", "   \n  ")  # whitespace-only


# --- Conformance -----------------------------------------------------------


class TestLLMQueryRewriterConformance(QueryRewriterConformance):
    @pytest.fixture
    def rewriter(self) -> LLMQueryRewriter:
        return LLMQueryRewriter(generator=_StaticGenerator("rewritten"))


# --- Unit ------------------------------------------------------------------


class TestLLMQueryRewriterUnit:
    async def test_empty_history_returns_query_unchanged_no_llm_call(self) -> None:
        gen = _StaticGenerator("should-not-be-returned")
        rewriter = LLMQueryRewriter(generator=gen)
        result = await rewriter.rewrite("what is alpha?", [])
        assert result == "what is alpha?"
        assert gen.call_count == 0

    async def test_with_history_returns_generator_content(self) -> None:
        rewriter = LLMQueryRewriter(generator=_StaticGenerator("the other pavement type"))
        history = [
            make_message("user", "tell me about asphalt pavement"),
            make_message("assistant", "Asphalt is the standard surface course."),
        ]
        result = await rewriter.rewrite("what about the other one?", history)
        assert result == "the other pavement type"

    async def test_strips_whitespace_from_generator_response(self) -> None:
        rewriter = LLMQueryRewriter(generator=_StaticGenerator("  trimmed query  \n"))
        result = await rewriter.rewrite(
            "follow up", [make_message("user", "prior")]
        )
        assert result == "trimmed query"

    async def test_whitespace_only_response_falls_back_to_original(self) -> None:
        rewriter = LLMQueryRewriter(generator=_EmptyGenerator())
        history = [make_message("user", "prior")]
        result = await rewriter.rewrite("follow up", history)
        assert result == "follow up"

    async def test_prompt_includes_query_and_history(self) -> None:
        cap = _CapturingGenerator(content="x")
        rewriter = LLMQueryRewriter(generator=cap)
        history = [
            make_message("user", "tell me about alpha"),
            make_message("assistant", "alpha is the first letter"),
        ]
        await rewriter.rewrite("what about the next one?", history)
        assert len(cap.calls) == 1
        prompt = cap.calls[0]["query"]
        assert "what about the next one?" in prompt
        assert "tell me about alpha" in prompt
        assert "alpha is the first letter" in prompt
        # History encoded with role tags so the LLM can parse turn boundaries.
        assert "<user>tell me about alpha</user>" in prompt
        assert "<assistant>alpha is the first letter</assistant>" in prompt

    async def test_passes_empty_context_and_history_to_generator(self) -> None:
        cap = _CapturingGenerator(content="x")
        rewriter = LLMQueryRewriter(generator=cap)
        await rewriter.rewrite(
            "follow up",
            [make_message("user", "prior")],
        )
        call = cap.calls[0]
        assert call["context"] == []
        # History is encoded inside the user-turn prompt, not as the
        # Generator's `history` parameter — keeps the rewrite a one-shot
        # transform rather than a conversational continuation.
        assert call["history"] == []

    async def test_system_prompt_is_configurable(self) -> None:
        cap = _CapturingGenerator(content="x")
        custom_prompt = "Custom rewrite instruction."
        rewriter = LLMQueryRewriter(generator=cap, system_prompt=custom_prompt)
        await rewriter.rewrite("q", [make_message("user", "p")])
        assert cap.calls[0]["system_prompt"] == custom_prompt

    async def test_works_with_any_generator_implementation(self) -> None:
        """No backend-specific isinstance checks — any Generator must work."""

        class _AnyGenerator(Generator):
            async def generate(self, query, context, system_prompt, history):  # type: ignore[no-untyped-def]
                return Message("assistant", "from any generator")

        rewriter = LLMQueryRewriter(generator=_AnyGenerator())
        result = await rewriter.rewrite(
            "follow up", [make_message("user", "prior")]
        )
        assert result == "from any generator"
