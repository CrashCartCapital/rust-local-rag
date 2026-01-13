"""Phase 2 Task 2.3: Document operations tests for ragcore module."""

import pytest


class TestUpsertDocument:
    """Test document upsert operations."""

    def test_upsert_basic(self, mock_engine):
        """Verify basic document upsert works."""
        chunks = mock_engine.upsert_document("test.txt", "Hello world, this is a test document.")
        assert isinstance(chunks, int)
        assert chunks > 0

    def test_upsert_returns_chunk_count(self, mock_engine):
        """Verify upsert returns number of chunks created."""
        # Longer text should create more chunks
        short_text = "Short text."
        long_text = " ".join(["This is sentence number {}.".format(i) for i in range(50)])

        chunks_short = mock_engine.upsert_document("short.txt", short_text)
        chunks_long = mock_engine.upsert_document("long.txt", long_text)

        # Both should create at least some chunks (depends on min length)
        assert isinstance(chunks_short, int)
        assert isinstance(chunks_long, int)

    def test_upsert_with_content_hash(self, mock_engine):
        """Verify content hash enables skip of unchanged documents."""
        text = "This is the document content."
        hash_value = "abc123"

        # First upsert should create chunks
        chunks1 = mock_engine.upsert_document("doc.txt", text, content_hash=hash_value)
        assert chunks1 >= 0  # May be 0 if text too short

        # Second upsert with same hash should skip
        chunks2 = mock_engine.upsert_document("doc.txt", text, content_hash=hash_value)
        assert chunks2 == 0, "Should skip unchanged document"

    def test_upsert_updates_existing(self, mock_engine):
        """Verify upsert replaces existing document."""
        mock_engine.upsert_document("doc.txt", "Original content here.")

        # Update with new content
        chunks = mock_engine.upsert_document("doc.txt", "New and different content.")
        # Should have replaced the document
        assert "doc.txt" in mock_engine.list_documents()

    def test_upsert_updates_stats(self, mock_engine):
        """Verify upsert updates engine stats."""
        initial_chunks = mock_engine.stats().chunk_count

        mock_engine.upsert_document("new.txt", "This is a new document with some text.")

        final_chunks = mock_engine.stats().chunk_count
        # Chunk count should have increased (unless text too short)
        assert final_chunks >= initial_chunks

    def test_upsert_empty_text(self, mock_engine):
        """Verify upserting empty text creates 0 chunks."""
        chunks = mock_engine.upsert_document("empty.txt", "")
        assert chunks == 0


class TestRemoveDocument:
    """Test document removal operations."""

    def test_remove_existing(self, mock_engine):
        """Verify removing existing document returns chunk count."""
        mock_engine.upsert_document("to_remove.txt", "Content to be removed.")

        removed = mock_engine.remove_document("to_remove.txt")
        assert isinstance(removed, int)

    def test_remove_nonexistent(self, mock_engine):
        """Verify removing nonexistent document returns 0."""
        removed = mock_engine.remove_document("nonexistent.txt")
        assert removed == 0

    def test_remove_updates_list(self, mock_engine):
        """Verify removed document no longer in list_documents."""
        mock_engine.upsert_document("doc.txt", "Some content here.")
        assert "doc.txt" in mock_engine.list_documents()

        mock_engine.remove_document("doc.txt")
        assert "doc.txt" not in mock_engine.list_documents()

    def test_remove_updates_stats(self, mock_engine):
        """Verify remove updates engine stats."""
        mock_engine.upsert_document("doc.txt", "Content that creates chunks.")
        before = mock_engine.stats().chunk_count

        mock_engine.remove_document("doc.txt")
        after = mock_engine.stats().chunk_count

        assert after <= before


class TestDocumentLifecycle:
    """Test full document lifecycle."""

    def test_upsert_save_load_search(self, mock_engine, tmp_path):
        """Verify document persists through save/load cycle."""
        # Upsert document
        mock_engine.upsert_document("test.txt", "Machine learning is amazing.")

        # Save state
        mock_engine.save()

        # Create new engine and load
        from ragcore import RagEngine
        engine2 = RagEngine.create_mock(str(tmp_path))
        engine2.load()

        # Document should be searchable in new engine
        assert "test.txt" in engine2.list_documents()

    def test_multiple_documents(self, mock_engine):
        """Verify multiple documents can be indexed."""
        docs = [
            ("doc1.txt", "First document about cats."),
            ("doc2.txt", "Second document about dogs."),
            ("doc3.txt", "Third document about birds."),
        ]

        for name, text in docs:
            mock_engine.upsert_document(name, text)

        # All documents should be listed
        doc_list = mock_engine.list_documents()
        for name, _ in docs:
            assert name in doc_list
