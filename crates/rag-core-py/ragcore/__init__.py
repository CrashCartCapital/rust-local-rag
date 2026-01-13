"""
ragcore - Python bindings for rag-core RAG engine.

Provides semantic search capabilities with pluggable embedding backends.

Example:
    >>> from ragcore import RagEngine
    >>> engine = RagEngine.create_mock("/tmp/index")
    >>> results = engine.search("query", top_k=5)
"""

from ragcore._native import (
    # Core engine
    RagEngine,
    # Types
    QuerySpec,
    SearchResult,
    ScoreBreakdown,
    EngineStats,
    # Exceptions
    RagError,
    EmbeddingError,
    RerankError,
    IndexError,
    ConfigError,
    ValidationError,
    # Metadata
    __version__,
)

__all__ = [
    # Core
    "RagEngine",
    # Types
    "QuerySpec",
    "SearchResult",
    "ScoreBreakdown",
    "EngineStats",
    # Exceptions
    "RagError",
    "EmbeddingError",
    "RerankError",
    "IndexError",
    "ConfigError",
    "ValidationError",
    # Metadata
    "__version__",
]
