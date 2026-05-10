"""Conformance tests every `Generator` adapter must pass."""

import pytest

from rag.stages.generator import Generator
from rag.types import Message

from tests.stages._helpers import make_search_results


class GeneratorConformance:
    @pytest.fixture
    def generator(self) -> Generator:
        raise NotImplementedError("subclass must provide a `generator` fixture")

    async def test_generate_returns_assistant_message(self, generator: Generator) -> None:
        result = await generator.generate(
            query="what is alpha?",
            context=make_search_results(3),
            system_prompt="You are a helpful assistant.",
            history=[],
        )
        assert isinstance(result, Message)
        assert result.role == "assistant"
        assert isinstance(result.content, str) and result.content
