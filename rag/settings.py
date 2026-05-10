"""Credentials and service URLs sourced from environment variables.

A single `Settings` instance is read at the edge — typically inside `compose.py`
when wiring adapters — and the relevant fields are passed into adapter
constructors. Stages and the Orchestration Layer never read environment
variables directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"

    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-english-v3.0"

    snowflake_account: str | None = None
    snowflake_user: str | None = None
    snowflake_password: str | None = None
    snowflake_role: str | None = None
    snowflake_warehouse: str | None = None
    snowflake_database: str | None = None
    snowflake_schema: str | None = None
    snowflake_cortex_search_service: str | None = None
    snowflake_cortex_chat_model: str = "mistral-large2"
    snowflake_cortex_embedding_model: str = "e5-base-v2"

    postgres_url: str | None = None

    chroma_persist_dir: str = "./.chroma"

    sqlite_path: str = "./rag.db"
