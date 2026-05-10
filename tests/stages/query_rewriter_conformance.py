"""Conformance tests every `QueryRewriter` adapter must pass."""

import pytest

from rag.stages.query_rewriter import QueryRewriter

from tests.stages._helpers import make_message


class QueryRewriterConformance:
    @pytest.fixture
    def rewriter(self) -> QueryRewriter:
        raise NotImplementedError("subclass must provide a `rewriter` fixture")

    async def test_rewrite_returns_non_empty_string(self, rewriter: QueryRewriter) -> None:
        result = await rewriter.rewrite("what about the other one?", [])
        assert isinstance(result, str)
        assert result.strip()

    async def test_rewrite_with_history(self, rewriter: QueryRewriter) -> None:
        history = [
            make_message("user", "tell me about alpha"),
            make_message("assistant", "alpha is the first letter"),
        ]
        result = await rewriter.rewrite("and the next one?", history)
        assert isinstance(result, str)
        assert result.strip()
