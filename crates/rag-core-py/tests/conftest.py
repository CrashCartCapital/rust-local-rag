"""Pytest configuration and fixtures for rag-core-py tests."""

import pytest


@pytest.fixture
def mock_engine(tmp_path):
    """Create a mock RagEngine for testing.

    Uses the built-in mock backend that returns deterministic embeddings.
    """
    from ragcore import RagEngine

    return RagEngine.create_mock(str(tmp_path))
