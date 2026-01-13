"""Phase 5: Tests for error handling hardening."""

import pytest


class TestExceptionHierarchy:
    """Test the exception hierarchy is correctly defined."""

    def test_exception_base_class(self):
        """Test RagError is the base for all ragcore exceptions."""
        from ragcore import (
            RagError,
            EmbeddingError,
            RerankError,
            IndexError,
            ConfigError,
            ValidationError,
        )

        # All exceptions inherit from RagError
        assert issubclass(EmbeddingError, RagError)
        assert issubclass(RerankError, RagError)
        assert issubclass(IndexError, RagError)
        assert issubclass(ConfigError, RagError)
        assert issubclass(ValidationError, RagError)

        # RagError inherits from Exception
        assert issubclass(RagError, Exception)

    def test_catch_all_ragcore_errors(self):
        """Test that catching RagError catches all ragcore exceptions."""
        from ragcore import RagError, ValidationError

        # Create a ValidationError
        error = ValidationError("test error")

        # Can be caught as RagError
        try:
            raise error
        except RagError as e:
            assert "test error" in str(e)


class TestEmbeddingErrors:
    """Test embedding error handling."""

    def test_backend_embed_error_becomes_embedding_error(self, tmp_path):
        """Test that embed() errors become EmbeddingError."""
        from ragcore import RagEngine, EmbeddingError

        class FailingEmbedder:
            def model_id(self) -> str:
                return "failing-embedder"

            def dimension(self) -> int:
                return 32

            def embed(self, text: str) -> list[float]:
                raise RuntimeError("Embedding failed!")

        engine = RagEngine.create(str(tmp_path), backend=FailingEmbedder())

        with pytest.raises(EmbeddingError, match="Embedding failed"):
            engine.upsert_document("test.txt", "Content that will fail.")


class TestValidationErrors:
    """Test input validation errors."""

    def test_empty_query_raises_value_error(self, tmp_path):
        """Test that empty query raises ValueError."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path))

        with pytest.raises(ValueError, match="empty"):
            engine.search("")

    def test_async_empty_query_raises_value_error(self, tmp_path):
        """Test that async empty query raises ValueError."""
        import asyncio
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path))

        with pytest.raises(ValueError, match="empty"):
            asyncio.run(engine.asearch(""))


class TestBackendValidation:
    """Test backend validation errors."""

    def test_missing_model_id_raises(self, tmp_path):
        """Test error when backend missing model_id."""
        from ragcore import RagEngine

        class NoModelId:
            def dimension(self) -> int:
                return 32

            def embed(self, text: str) -> list[float]:
                return [0.0] * 32

        with pytest.raises(AttributeError, match="model_id"):
            RagEngine.create(str(tmp_path), backend=NoModelId())

    def test_missing_dimension_raises(self, tmp_path):
        """Test error when backend missing dimension."""
        from ragcore import RagEngine

        class NoDimension:
            def model_id(self) -> str:
                return "test"

            def embed(self, text: str) -> list[float]:
                return [0.0] * 32

        with pytest.raises(AttributeError, match="dimension"):
            RagEngine.create(str(tmp_path), backend=NoDimension())

    def test_missing_embed_raises(self, tmp_path):
        """Test error when backend missing embed."""
        from ragcore import RagEngine

        class NoEmbed:
            def model_id(self) -> str:
                return "test"

            def dimension(self) -> int:
                return 32

        with pytest.raises(AttributeError, match="embed"):
            RagEngine.create(str(tmp_path), backend=NoEmbed())


class TestErrorMessages:
    """Test that error messages are actionable."""

    def test_embedding_error_contains_context(self, tmp_path):
        """Test EmbeddingError contains useful context."""
        from ragcore import RagEngine, EmbeddingError

        class ContextualFailer:
            def model_id(self) -> str:
                return "contextual-failer"

            def dimension(self) -> int:
                return 32

            def embed(self, text: str) -> list[float]:
                raise ValueError(f"Failed to embed text of length {len(text)}")

        engine = RagEngine.create(str(tmp_path), backend=ContextualFailer())

        try:
            engine.upsert_document("test.txt", "Short text")
            assert False, "Should have raised"
        except EmbeddingError as e:
            error_str = str(e)
            # Error should contain the original message context
            assert "Failed to embed" in error_str or "10" in error_str

    def test_attribute_error_names_missing_method(self, tmp_path):
        """Test AttributeError identifies the missing method."""
        from ragcore import RagEngine

        class IncompleteBackend:
            def model_id(self) -> str:
                return "incomplete"
            # Missing dimension and embed

        try:
            RagEngine.create(str(tmp_path), backend=IncompleteBackend())
            assert False, "Should have raised"
        except AttributeError as e:
            # Error should mention what's missing
            error_str = str(e)
            assert "dimension" in error_str or "embed" in error_str


class TestPanicSafety:
    """Test that Rust panics don't escape to Python.

    Note: These tests verify the catch_panic! macro works correctly.
    Actual panic injection would require specific Rust test code.
    """

    def test_mock_engine_handles_zero_dimension(self, tmp_path):
        """Test that zero dimension is handled gracefully."""
        from ragcore import RagEngine

        # This could potentially panic in Rust, but should be caught
        try:
            engine = RagEngine.create_mock(str(tmp_path), dimension=0)
            # If we get here, it either accepted 0 or raised a Python exception
            # Either is acceptable - the key is no crash
        except (RuntimeError, ValueError):
            pass  # Acceptable - error was caught and converted

    def test_repeated_operations_dont_corrupt_state(self, tmp_path):
        """Test that errors don't corrupt engine state."""
        from ragcore import RagEngine, EmbeddingError

        class SometimesFails:
            def __init__(self):
                self.call_count = 0

            def model_id(self) -> str:
                return "sometimes-fails"

            def dimension(self) -> int:
                return 32

            def embed(self, text: str) -> list[float]:
                self.call_count += 1
                if self.call_count % 2 == 0:
                    raise RuntimeError("Intermittent failure")
                return [0.5] * 32

        engine = RagEngine.create(str(tmp_path), backend=SometimesFails())

        # First call should succeed
        engine.upsert_document("doc1.txt", "First document")

        # Second call should fail
        with pytest.raises(EmbeddingError):
            engine.upsert_document("doc2.txt", "Second document")

        # Engine should still be usable
        assert engine.stats().document_count == 1
        assert "doc1.txt" in engine.list_documents()
