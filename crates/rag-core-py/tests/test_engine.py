"""Phase 2: Engine integration tests for ragcore module."""

import pytest


class TestEngineBasics:
    """Test basic engine operations from Phase 2."""

    def test_engine_index_dir_getter(self, mock_engine, tmp_path):
        """Verify index_dir property returns the correct path."""
        assert mock_engine.index_dir == str(tmp_path)

    def test_engine_repr(self, mock_engine, tmp_path):
        """Verify __repr__ includes index_dir."""
        repr_str = repr(mock_engine)
        assert "RagEngine" in repr_str
        assert str(tmp_path) in repr_str

    def test_list_documents_initially_empty(self, mock_engine):
        """Verify list_documents returns empty list on new engine."""
        docs = mock_engine.list_documents()
        assert isinstance(docs, list)
        assert len(docs) == 0

    def test_needs_reindex_initially_false(self, mock_engine):
        """Verify needs_reindex is False on new engine."""
        assert mock_engine.needs_reindex() is False


class TestEnginePersistence:
    """Test engine save/load operations."""

    def test_save_empty_engine(self, mock_engine):
        """Verify save() works on empty engine."""
        # Should not raise
        mock_engine.save()

    def test_save_creates_index_file(self, mock_engine, tmp_path):
        """Verify save() creates an index file."""
        mock_engine.save()

        # Check that some index file was created
        # The exact filename depends on the model name
        files = list(tmp_path.iterdir())
        # With mock-768 backend, should create chunks_mock-768.json
        assert any("chunks" in f.name for f in files if f.is_file())

    def test_load_empty_engine(self, mock_engine, tmp_path):
        """Verify load() works (no-op on fresh engine)."""
        # Save first to create the file
        mock_engine.save()
        # Then load (should not raise)
        mock_engine.load()

    def test_roundtrip_empty_engine(self, tmp_path):
        """Verify save/load roundtrip preserves state."""
        from ragcore import RagEngine

        # Create and save
        engine1 = RagEngine.create_mock(str(tmp_path))
        engine1.save()

        # Create new engine and load
        engine2 = RagEngine.create_mock(str(tmp_path))
        engine2.load()

        # Both should have same stats
        assert engine1.stats().chunk_count == engine2.stats().chunk_count
        assert engine1.stats().document_count == engine2.stats().document_count


class TestEngineStats:
    """Test engine statistics from rag-core integration."""

    def test_stats_uses_rag_core_health(self, mock_engine):
        """Verify stats() uses rag-core's health() method."""
        from ragcore import EngineStats

        stats = mock_engine.stats()
        assert isinstance(stats, EngineStats)

        # Should report is_healthy state
        # (We can't directly check is_healthy, but these should be consistent)
        assert stats.chunk_count >= 0
        assert stats.document_count >= 0
        assert len(stats.embedding_model) > 0
        assert stats.embedding_dimension > 0

    def test_stats_repr(self, mock_engine):
        """Verify EngineStats has useful repr."""
        stats = mock_engine.stats()
        repr_str = repr(stats)
        assert "EngineStats" in repr_str
        assert "documents" in repr_str
        assert "chunks" in repr_str


class TestErrorHandling:
    """Test error handling at FFI boundary."""

    def test_create_mock_with_invalid_dimension_zero(self, tmp_path):
        """Test behavior with edge case dimension=0."""
        from ragcore import RagEngine

        # Dimension 0 should work (edge case) - the engine creates but embeddings are empty
        engine = RagEngine.create_mock(str(tmp_path), dimension=0)
        assert engine.stats().embedding_dimension == 0

    def test_create_mock_with_large_dimension(self, tmp_path):
        """Test behavior with large dimension."""
        from ragcore import RagEngine

        # Large dimension should work
        engine = RagEngine.create_mock(str(tmp_path), dimension=4096)
        assert engine.stats().embedding_dimension == 4096
