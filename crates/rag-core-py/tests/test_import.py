"""TDD: Basic import tests for ragcore module."""

import pytest


def test_module_imports():
    """FR-1: Python can import ragcore module."""
    import ragcore

    assert hasattr(ragcore, "RagEngine")
    assert hasattr(ragcore, "QuerySpec")
    assert hasattr(ragcore, "SearchResult")
    assert hasattr(ragcore, "ScoreBreakdown")
    assert hasattr(ragcore, "EngineStats")


def test_exception_hierarchy():
    """Verify exception classes are importable."""
    from ragcore import (
        RagError,
        EmbeddingError,
        RerankError,
        IndexError,
        ConfigError,
        ValidationError,
    )

    # Verify hierarchy
    assert issubclass(EmbeddingError, RagError)
    assert issubclass(RerankError, RagError)
    assert issubclass(IndexError, RagError)
    assert issubclass(ConfigError, RagError)
    assert issubclass(ValidationError, RagError)


def test_version_available():
    """Verify __version__ is exposed."""
    from ragcore import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_mock_engine_creates(mock_engine):
    """Verify mock engine can be instantiated (TDD placeholder)."""
    assert mock_engine is not None


def test_mock_engine_stats(mock_engine):
    """Verify stats() returns EngineStats."""
    from ragcore import EngineStats

    stats = mock_engine.stats()

    assert isinstance(stats, EngineStats)
    assert stats.chunk_count == 0
    assert stats.document_count == 0
    # Default mock dimension is 768
    assert stats.embedding_dimension == 768
    assert stats.embedding_model == "mock-768"


def test_mock_engine_custom_dimension(tmp_path):
    """Verify mock engine respects custom dimension."""
    from ragcore import RagEngine

    engine = RagEngine.create_mock(str(tmp_path), dimension=384)
    stats = engine.stats()

    assert stats.embedding_dimension == 384
    assert stats.embedding_model == "mock-384"


def test_query_spec_creation():
    """Verify QuerySpec can be instantiated with kwargs."""
    from ragcore import QuerySpec

    spec = QuerySpec(top_k=5, embedding_weight=0.8, diversity_factor=0.2)

    assert spec.top_k == 5
    assert spec.embedding_weight == pytest.approx(0.8, rel=1e-5)
    assert spec.diversity_factor == pytest.approx(0.2, rel=1e-5)


def test_query_spec_defaults():
    """Verify QuerySpec has sensible defaults."""
    from ragcore import QuerySpec

    spec = QuerySpec()

    assert spec.top_k == 10
    assert spec.embedding_weight == pytest.approx(0.7, rel=1e-5)
    assert spec.lexical_weight == pytest.approx(0.3, rel=1e-5)
    assert spec.diversity_factor == pytest.approx(0.0, abs=1e-5)
