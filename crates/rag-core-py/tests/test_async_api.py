"""Phase 4: Tests for async Python API (asearch, aupsert_document)."""

import asyncio
import pytest


class TestAsyncSearch:
    """Test asearch() async search method."""

    @pytest.mark.asyncio
    async def test_asearch_basic(self, tmp_path):
        """Test basic async search returns results."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=64)
        engine.upsert_document("doc.txt", "The quick brown fox jumps over the lazy dog.")

        results = await engine.asearch("fox", top_k=5)

        assert len(results) > 0
        assert results[0].document == "doc.txt"

    @pytest.mark.asyncio
    async def test_asearch_empty_query_raises(self, tmp_path):
        """Test async search raises on empty query."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path))

        with pytest.raises(ValueError, match="empty"):
            await engine.asearch("")

    @pytest.mark.asyncio
    async def test_asearch_empty_engine(self, tmp_path):
        """Test async search on empty engine returns empty list."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path))
        results = await engine.asearch("anything", top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_asearch_with_top_k(self, tmp_path):
        """Test async search respects top_k parameter."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=32)

        # Add multiple documents
        for i in range(5):
            engine.upsert_document(f"doc{i}.txt", f"Document {i} with some content about testing.")

        results = await engine.asearch("testing", top_k=2)

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_asearch_with_spec(self, tmp_path):
        """Test async search with QuerySpec."""
        from ragcore import RagEngine, QuerySpec

        engine = RagEngine.create_mock(str(tmp_path), dimension=64)
        engine.upsert_document("doc.txt", "Test document for search.")

        spec = QuerySpec(diversity_factor=0.5)
        results = await engine.asearch("test", top_k=5, spec=spec)

        assert isinstance(results, list)


class TestAsyncUpsertDocument:
    """Test aupsert_document() async upsert method."""

    @pytest.mark.asyncio
    async def test_aupsert_document_basic(self, tmp_path):
        """Test basic async document upsert."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=64)

        chunk_count = await engine.aupsert_document("async_doc.txt", "Hello async world!")

        assert chunk_count > 0
        assert engine.stats().document_count == 1
        assert "async_doc.txt" in engine.list_documents()

    @pytest.mark.asyncio
    async def test_aupsert_document_with_hash(self, tmp_path):
        """Test async upsert with content hash."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=64)

        # First upsert
        await engine.aupsert_document("doc.txt", "Content", content_hash="hash123")

        # Second upsert with same hash should skip
        chunk_count = await engine.aupsert_document("doc.txt", "Content", content_hash="hash123")
        assert chunk_count == 0

    @pytest.mark.asyncio
    async def test_aupsert_document_empty_text(self, tmp_path):
        """Test async upsert with empty text."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path))

        chunk_count = await engine.aupsert_document("empty.txt", "")
        assert chunk_count == 0


class TestAsyncParity:
    """Test that async methods produce same results as sync methods."""

    @pytest.mark.asyncio
    async def test_search_parity(self, tmp_path):
        """Test asearch produces same results as search."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=64)
        engine.upsert_document("test.txt", "Document about machine learning and AI.")

        # Sync search
        sync_results = engine.search("machine learning", top_k=5)

        # Async search
        async_results = await engine.asearch("machine learning", top_k=5)

        # Results should be identical
        assert len(sync_results) == len(async_results)
        for sync_r, async_r in zip(sync_results, async_results):
            assert sync_r.document == async_r.document
            assert sync_r.text == async_r.text
            assert abs(sync_r.score - async_r.score) < 1e-6

    @pytest.mark.asyncio
    async def test_upsert_parity(self, tmp_path):
        """Test aupsert_document produces same chunk count as upsert_document."""
        from ragcore import RagEngine

        # Sync engine
        sync_engine = RagEngine.create_mock(str(tmp_path / "sync"), dimension=64)
        sync_count = sync_engine.upsert_document("doc.txt", "Test content for parity check.")

        # Async engine
        async_engine = RagEngine.create_mock(str(tmp_path / "async"), dimension=64)
        async_count = await async_engine.aupsert_document("doc.txt", "Test content for parity check.")

        assert sync_count == async_count


class TestAsyncConcurrency:
    """Test async methods work correctly with concurrent calls."""

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, tmp_path):
        """Test multiple concurrent async searches."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=32)
        engine.upsert_document("doc.txt", "Test document for concurrent searches.")

        # Launch 5 concurrent searches
        tasks = [engine.asearch(f"query{i}", top_k=3) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should complete without error
        assert len(results) == 5
        for r in results:
            assert isinstance(r, list)

    @pytest.mark.asyncio
    async def test_concurrent_upserts(self, tmp_path):
        """Test multiple concurrent async upserts."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=32)

        # Launch 5 concurrent upserts
        tasks = [
            engine.aupsert_document(f"doc{i}.txt", f"Content for document {i}")
            for i in range(5)
        ]
        counts = await asyncio.gather(*tasks)

        # All should complete without error
        assert len(counts) == 5
        assert all(c > 0 for c in counts)
        assert engine.stats().document_count == 5

    @pytest.mark.asyncio
    async def test_mixed_sync_async(self, tmp_path):
        """Test mixing sync and async operations."""
        from ragcore import RagEngine

        engine = RagEngine.create_mock(str(tmp_path), dimension=32)

        # Sync upsert
        engine.upsert_document("sync_doc.txt", "Sync document content.")

        # Async upsert
        await engine.aupsert_document("async_doc.txt", "Async document content.")

        # Should have both documents
        assert engine.stats().document_count == 2
        docs = engine.list_documents()
        assert "sync_doc.txt" in docs
        assert "async_doc.txt" in docs

        # Sync search
        sync_results = engine.search("document", top_k=10)

        # Async search
        async_results = await engine.asearch("document", top_k=10)

        # Both should return results from both documents
        assert len(sync_results) == len(async_results)
