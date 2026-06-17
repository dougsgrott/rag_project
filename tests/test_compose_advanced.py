"""Smoke tests for the advanced composition root.

The advanced stack needs live OpenAI + Cohere + Postgres backends, so it can't
be fully built here. These tests cover what's reachable without backends: the
CLI surface and the fail-fast credential checks (which run before any adapter
is constructed).
"""

import pytest

import compose_advanced as ca
from rag.errors import ConfigurationError
from rag.settings import Settings


def test_parser_exposes_expected_subcommands_and_flags() -> None:
    parser = ca._build_parser()
    args = parser.parse_args(
        ["evaluate", "cases.json", "--retrieval-k", "7", "--final-top-k", "3", "--per-case"]
    )
    assert args.cmd == "evaluate"
    assert args.retrieval_k == 7
    assert args.final_top_k == 3
    assert args.per_case is True

    for cmd in ("ingest", "query", "set-prompt"):
        # Each subcommand parses without error given its required positionals.
        parser.parse_args({"ingest": ["ingest", "p"], "query": ["query", "c", "q"],
                           "set-prompt": ["set-prompt", "d", "txt"]}[cmd])


async def test_build_requires_openai_key() -> None:
    settings = Settings(openai_api_key=None, postgres_url="postgresql://x", cohere_api_key="c")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        async with ca.build_advanced_stack(settings):
            pass


async def test_build_requires_postgres_url() -> None:
    settings = Settings(openai_api_key="sk-x", postgres_url=None, cohere_api_key="c")
    with pytest.raises(ConfigurationError, match="POSTGRES_URL"):
        async with ca.build_advanced_stack(settings):
            pass


async def test_build_requires_cohere_key() -> None:
    settings = Settings(openai_api_key="sk-x", postgres_url="postgresql://x", cohere_api_key=None)
    with pytest.raises(ConfigurationError, match="COHERE_API_KEY"):
        async with ca.build_advanced_stack(settings):
            pass
