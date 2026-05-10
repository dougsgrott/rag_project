import pytest

from rag.adapters.local.query_rewriter import IdentityQueryRewriter

from tests.stages._helpers import make_message
from tests.stages.query_rewriter_conformance import QueryRewriterConformance


class TestIdentityQueryRewriter(QueryRewriterConformance):
    @pytest.fixture
    def rewriter(self) -> IdentityQueryRewriter:
        return IdentityQueryRewriter()

    async def test_returns_query_unchanged_with_empty_history(
        self, rewriter: IdentityQueryRewriter
    ) -> None:
        assert await rewriter.rewrite("what is alpha?", []) == "what is alpha?"

    async def test_returns_query_unchanged_with_history(
        self, rewriter: IdentityQueryRewriter
    ) -> None:
        history = [make_message("user", "previous turn")]
        assert await rewriter.rewrite("follow-up", history) == "follow-up"
