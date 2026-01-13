"""Phase 2 Task 2.2: Search functionality tests for ragcore module."""

import pytest


class TestSearchBasics:
    """Test basic search operations."""

    def test_search_empty_engine(self, mock_engine):
        """Verify searching an empty engine returns empty list."""
        results = mock_engine.search("test query")
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_with_top_k(self, mock_engine):
        """Verify top_k parameter is respected."""
        # Search with different top_k values
        results_5 = mock_engine.search("query", top_k=5)
        results_10 = mock_engine.search("query", top_k=10)

        # Both should be empty lists (no documents)
        assert len(results_5) == 0
        assert len(results_10) == 0

    def test_search_empty_query_raises(self, mock_engine):
        """Verify empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            mock_engine.search("")


class TestSearchWithQuerySpec:
    """Test search with QuerySpec configuration."""

    def test_search_with_spec(self, mock_engine):
        """Verify search accepts QuerySpec."""
        from ragcore import QuerySpec

        spec = QuerySpec(
            top_k=5,
            embedding_weight=0.8,
            diversity_factor=0.2
        )

        # Should not raise
        results = mock_engine.search("test", spec=spec)
        assert isinstance(results, list)

    def test_search_with_diversity(self, mock_engine):
        """Verify diversity search works."""
        from ragcore import QuerySpec

        spec = QuerySpec(diversity_factor=0.5)
        results = mock_engine.search("test", spec=spec)
        assert isinstance(results, list)


class TestSearchResults:
    """Test search result types and fields."""

    def test_search_returns_search_result_type(self, mock_engine):
        """Verify search returns SearchResult objects."""
        # Empty engine returns empty list, but type checking should work
        results = mock_engine.search("test")
        assert isinstance(results, list)

        # If we had results, they would be SearchResult instances
        # This is tested more thoroughly in test_search_result_fields below

    def test_search_result_fields(self, mock_engine):
        """Task 2.4 TDD: Verify all SearchResult fields are accessible from Python."""
        from ragcore import SearchResult, ScoreBreakdown

        # Add a document so we get search results
        mock_engine.upsert_document("test.txt", "This is test content for search verification.")
        results = mock_engine.search("test")

        # Should have at least one result
        assert len(results) > 0

        r = results[0]
        # Verify result is SearchResult type
        assert isinstance(r, SearchResult)

        # Verify all expected fields are accessible and correct types
        assert isinstance(r.id, str)
        assert isinstance(r.text, str)
        assert isinstance(r.document, str)
        assert isinstance(r.chunk_index, int)
        # page_number may be None or int
        assert r.page_number is None or isinstance(r.page_number, int)
        # section may be None or str
        assert r.section is None or isinstance(r.section, str)
        assert isinstance(r.score, float)

        # Verify ScoreBreakdown
        assert isinstance(r.scores, ScoreBreakdown)
        # Score breakdown fields may be None or float
        assert r.scores.embedding is None or isinstance(r.scores.embedding, float)
        assert r.scores.lexical is None or isinstance(r.scores.lexical, float)
        assert r.scores.initial is None or isinstance(r.scores.initial, float)
        assert r.scores.reranker is None or isinstance(r.scores.reranker, float)
        assert isinstance(r.scores.total, float)

        # Total score should match the result score
        assert r.scores.total == r.score

    def test_query_spec_fields(self, mock_engine):
        """Task 2.4 TDD: Verify QuerySpec fields are accessible and work correctly."""
        from ragcore import QuerySpec

        # Create with custom values
        spec = QuerySpec(
            top_k=15,
            embedding_weight=0.6,
            lexical_weight=0.4,
            reranker_weight=0.8,
            initial_weight=0.2,
            diversity_factor=0.3
        )

        # Verify all getters work
        assert spec.top_k == 15
        assert abs(spec.embedding_weight - 0.6) < 0.001
        assert abs(spec.lexical_weight - 0.4) < 0.001
        assert abs(spec.reranker_weight - 0.8) < 0.001
        assert abs(spec.initial_weight - 0.2) < 0.001
        assert abs(spec.diversity_factor - 0.3) < 0.001

        # Use in search (should not raise)
        results = mock_engine.search("test", spec=spec)
        assert isinstance(results, list)


class TestSearchThreadSafety:
    """Test thread safety of search operations."""

    def test_search_concurrent_calls(self, mock_engine):
        """Verify multiple search calls don't deadlock."""
        import threading
        import time

        results = []
        errors = []

        def do_search():
            try:
                r = mock_engine.search("test", top_k=5)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_search) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # All threads should complete without errors
        assert len(errors) == 0
        assert len(results) == 5
