"""Phase 5: Tests for pickle support."""

import pickle
import pytest


# Module-level classes for pickle tests (local classes can't be pickled)
class ModuleLevelEmbedder:
    """Simple picklable embedder at module level."""

    def __init__(self, dim=32):
        self.dim = dim

    def model_id(self) -> str:
        return "pickle-test-embedder"

    def dimension(self) -> int:
        return self.dim

    def embed(self, text: str) -> list[float]:
        return [float(len(text))] + [0.5] * (self.dim - 1)


class ModuleLevelReranker:
    """Simple picklable reranker at module level."""

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        return [{"chunk_id": c["chunk_id"], "relevance": 0.9} for c in candidates]


class TestPickleMockEngine:
    """Test pickle support for mock engines."""

    def test_pickle_mock_engine_roundtrip(self, tmp_path):
        """Test mock engine can be pickled and unpickled."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=128)
        engine.upsert_document("test.txt", "Hello pickle world!")

        # Pickle and unpickle
        pickled = pickle.dumps(engine)
        restored = pickle.loads(pickled)

        # Restored engine should work
        assert restored.stats().embedding_dimension == 128
        assert "test.txt" not in restored.list_documents()  # State not preserved (just construction)

    def test_pickle_mock_engine_preserves_dimension(self, tmp_path):
        """Test pickle preserves mock dimension."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path / "orig"), dimension=512)

        pickled = pickle.dumps(engine)
        restored = pickle.loads(pickled)

        assert restored.stats().embedding_dimension == 512

    def test_pickle_mock_engine_with_loaded_data(self, tmp_path):
        """Test pickle + save/load preserves data."""
        from ragcore import RagEngine

        # Create, add data, save
        engine = RagEngine.create_mock(str(tmp_path), dimension=64)
        engine.upsert_document("doc.txt", "Content for pickle test.")
        engine.save()

        # Pickle and restore
        pickled = pickle.dumps(engine)
        restored = pickle.loads(pickled)

        # Load persisted data
        restored.load()

        # Should have the document now
        assert restored.stats().document_count == 1
        assert "doc.txt" in restored.list_documents()


class TestPicklePythonBackend:
    """Test pickle support for Python backend engines."""

    def test_pickle_python_backend_simple(self, tmp_path):
        """Test pickling engine with simple Python embedder."""
        from ragcore import RagEngine

        # Use module-level class (local classes can't be pickled)
        embedder = ModuleLevelEmbedder(dim=64)
        engine = RagEngine.create(str(tmp_path), backend=embedder)

        # Pickle and unpickle
        pickled = pickle.dumps(engine)
        restored = pickle.loads(pickled)

        # Restored engine should work
        assert restored.stats().embedding_dimension == 64
        chunk_count = restored.upsert_document("test.txt", "Testing pickle.")
        assert chunk_count > 0

    def test_pickle_python_backend_with_reranker(self, tmp_path):
        """Test pickling engine with Python embedder and reranker."""
        from ragcore import RagEngine

        # Use module-level classes
        engine = RagEngine.create(
            str(tmp_path),
            backend=ModuleLevelEmbedder(dim=32),
            reranker=ModuleLevelReranker(),
        )

        # Pickle and unpickle
        pickled = pickle.dumps(engine)
        restored = pickle.loads(pickled)

        # Restored engine should work
        assert restored.stats().embedding_dimension == 32

    def test_pickle_local_class_not_picklable(self, tmp_path):
        """Test that local classes (not at module level) can't be pickled.

        This is standard Python behavior, not a ragcore limitation.
        """
        from ragcore import RagEngine

        # Local class - can't be pickled by design
        class LocalEmbedder:
            def model_id(self) -> str:
                return "local"

            def dimension(self) -> int:
                return 32

            def embed(self, text: str) -> list[float]:
                return [0.5] * 32

        engine = RagEngine.create(str(tmp_path), backend=LocalEmbedder())

        # Should raise pickle error (can't pickle local class)
        with pytest.raises(AttributeError, match="Can't get local object"):
            pickle.dumps(engine)

    def test_pickle_non_picklable_backend_raises(self, tmp_path):
        """Test that non-picklable backend raises pickle error."""
        from ragcore import RagEngine

        class NonPicklableEmbedder:
            def __init__(self):
                # Lambda is not picklable
                self._func = lambda x: x

            def model_id(self) -> str:
                return "non-picklable"

            def dimension(self) -> int:
                return 32

            def embed(self, text: str) -> list[float]:
                return [0.5] * 32

        engine = RagEngine.create(str(tmp_path), backend=NonPicklableEmbedder())

        # Should raise pickle error (can't pickle local class OR lambda)
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            pickle.dumps(engine)


class TestMultiprocessing:
    """Test engine works with multiprocessing."""

    def test_mock_engine_pickle_for_multiprocessing(self, tmp_path):
        """Test mock engine pickle is suitable for multiprocessing.

        Actual multiprocessing spawning is platform-specific and requires
        module-level functions. This test verifies the pickle format is
        correct for multiprocessing use.
        """
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=256)

        # Verify pickle/unpickle works (prerequisite for multiprocessing)
        pickled = pickle.dumps(engine)
        restored = pickle.loads(pickled)

        assert restored.stats().embedding_dimension == 256

        # Verify __reduce__ returns the expected format
        reduce_result = engine.__reduce__()
        assert len(reduce_result) == 2
        callable_, args = reduce_result
        assert callable_.__name__ == "create_mock"
        assert args[1] == 256  # dimension
