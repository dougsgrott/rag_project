"""Shared fixtures for Postgres-backed adapter tests.

Tests in this tree are gated by ``INTEGRATION=true`` and a reachable
PostgreSQL instance with the ``pgvector`` extension available. Without
both, the suites skip.

Run a compatible Postgres locally with::

    docker run -d --name rag-pg -e POSTGRES_PASSWORD=postgres \\
      -p 5432:5432 pgvector/pgvector:pg17

Then::

    INTEGRATION=true TEST_POSTGRES_DSN=postgresql://postgres:postgres@localhost:5432/postgres \\
      uv run --group dev --group pgvector pytest tests/adapters/postgres tests/adapters/pgvector
"""

from __future__ import annotations

import os
import uuid

import pytest

INTEGRATION = os.environ.get("INTEGRATION", "").lower() in {"1", "true", "yes"}
TEST_POSTGRES_DSN = os.environ.get(
    "TEST_POSTGRES_DSN",
    "postgresql://postgres:postgres@localhost:5432/postgres",
)

requires_integration = pytest.mark.skipif(
    not INTEGRATION,
    reason="Set INTEGRATION=true (and TEST_POSTGRES_DSN) to run Postgres adapter tests",
)


def unique_table(prefix: str) -> str:
    """Return a unique-per-test table name to keep concurrent runs isolated."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
