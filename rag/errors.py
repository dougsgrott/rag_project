"""Error hierarchy for the RAG pipeline.

All adapter exceptions inherit from `RAGError` so the Orchestration Layer can
handle errors uniformly regardless of backend. Retry policy lives in the
Orchestration Layer — adapters raise, the pipeline decides.
"""

__all__ = [
    "RAGError",
    "BackendCommunicationError",
    "RateLimitError",
    "ConfigurationError",
    "RetrievalError",
    "GenerationError",
]


class RAGError(Exception):
    """Base class for every error raised by a RAG pipeline stage."""


class BackendCommunicationError(RAGError):
    """Timeout or network failure talking to a backend. Retryable with backoff."""


class RateLimitError(RAGError):
    """Backend returned HTTP 429 or equivalent. Retryable with backoff."""


class ConfigurationError(RAGError):
    """Missing credential, malformed setting, or unreachable required resource. Fail fast."""


class RetrievalError(RAGError):
    """Vector search returned no usable results."""


class GenerationError(RAGError):
    """LLM failed to produce a response."""
