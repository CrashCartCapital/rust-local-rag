"""Phase 4: Tests for RagEngine with Python embedding/reranking backends."""

import pytest


class SimplePyEmbedder:
    """Simple Python embedding backend for testing.

    Returns deterministic embeddings based on text length.
    """

    def __init__(self, dimension: int = 768):
        self._dimension = dimension

    def model_id(self) -> str:
        return "simple-py-embedder"

    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Return embedding where first component is text length, rest are 0.5."""
        result = [float(len(text))]
        result.extend([0.5] * (self._dimension - 1))
        return result


class BatchPyEmbedder(SimplePyEmbedder):
    """Python embedding backend with batch support."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for multiple texts."""
        return [self.embed(t) for t in texts]


class SimplePyReranker:
    """Simple Python reranker for testing.

    Returns relevance based on text length.
    """

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Return reranked results with relevance = text_length / 100."""
        results = []
        for c in candidates:
            text_len = len(c.get("text", ""))
            results.append({
                "chunk_id": c["chunk_id"],
                "relevance": min(text_len / 100.0, 1.0),
            })
        return results


class LogprobPyReranker:
    """Python reranker that returns logprobs."""

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Return reranked results with logprobs."""
        results = []
        for c in candidates:
            results.append({
                "chunk_id": c["chunk_id"],
                "relevance": 0.8,
                "yes_logprob": -0.5,
                "no_logprob": -2.0,
            })
        return results


class TestCreateWithPythonBackend:
    """Test RagEngine.create() with Python backends."""

    def test_create_with_simple_embedder(self, tmp_path):
        """Test creating engine with a simple Python embedder."""
        from ragcore import RagEngine

        embedder = SimplePyEmbedder(dimension=128)
        engine = RagEngine.create(str(tmp_path), backend=embedder)

        stats = engine.stats()
        assert stats.embedding_dimension == 128
        assert stats.embedding_model == "simple-py-embedder"

    def test_create_with_batch_embedder(self, tmp_path):
        """Test creating engine with batch-capable Python embedder."""
        from ragcore import RagEngine

        embedder = BatchPyEmbedder(dimension=256)
        engine = RagEngine.create(str(tmp_path), backend=embedder)

        stats = engine.stats()
        assert stats.embedding_dimension == 256

    def test_create_with_reranker(self, tmp_path):
        """Test creating engine with both embedder and reranker."""
        from ragcore import RagEngine

        embedder = SimplePyEmbedder(dimension=64)
        reranker = SimplePyReranker()
        engine = RagEngine.create(str(tmp_path), backend=embedder, reranker=reranker)

        stats = engine.stats()
        assert stats.embedding_dimension == 64


class TestPythonBackendOperations:
    """Test engine operations using Python backends."""

    def test_upsert_with_python_embedder(self, tmp_path):
        """Test document upsert calls Python embed method."""
        from ragcore import RagEngine

        embedder = SimplePyEmbedder(dimension=64)
        engine = RagEngine.create(str(tmp_path), backend=embedder)

        # Upsert a document
        chunk_count = engine.upsert_document("test.txt", "Hello world! This is a test document.")

        # Should have created chunks
        assert chunk_count > 0
        assert engine.stats().document_count == 1
        assert engine.stats().chunk_count == chunk_count

    def test_search_with_python_embedder(self, tmp_path):
        """Test search calls Python embed method."""
        from ragcore import RagEngine

        embedder = SimplePyEmbedder(dimension=64)
        engine = RagEngine.create(str(tmp_path), backend=embedder)

        # Upsert a document
        engine.upsert_document("doc1.txt", "The quick brown fox jumps over the lazy dog.")

        # Search
        results = engine.search("fox", top_k=5)

        # Should return results
        assert len(results) > 0
        assert results[0].document == "doc1.txt"

    def test_multiple_documents_with_python_embedder(self, tmp_path):
        """Test multiple documents with Python embedder."""
        from ragcore import RagEngine

        embedder = BatchPyEmbedder(dimension=32)
        engine = RagEngine.create(str(tmp_path), backend=embedder)

        # Upsert multiple documents
        engine.upsert_document("doc1.txt", "First document about cats.")
        engine.upsert_document("doc2.txt", "Second document about dogs.")
        engine.upsert_document("doc3.txt", "Third document about birds.")

        assert engine.stats().document_count == 3

        # Search should work
        results = engine.search("document", top_k=10)
        assert len(results) >= 3

    def test_save_load_with_python_embedder(self, tmp_path):
        """Test persistence with Python embedder."""
        from ragcore import RagEngine

        embedder = SimplePyEmbedder(dimension=64)

        # Create, upsert, save
        engine1 = RagEngine.create(str(tmp_path), backend=embedder)
        engine1.upsert_document("test.txt", "Test content for persistence.")
        engine1.save()

        # Create new engine and load
        engine2 = RagEngine.create(str(tmp_path), backend=embedder)
        engine2.load()

        # Should have same state
        assert engine2.stats().document_count == 1
        assert "test.txt" in engine2.list_documents()


class TestPythonBackendErrors:
    """Test error handling for Python backends."""

    def test_missing_model_id_raises(self, tmp_path):
        """Test error when backend missing model_id method."""
        from ragcore import RagEngine

        class NoModelId:
            def dimension(self) -> int:
                return 768
            def embed(self, text: str) -> list[float]:
                return [0.0] * 768

        with pytest.raises(AttributeError, match="model_id"):
            RagEngine.create(str(tmp_path), backend=NoModelId())

    def test_missing_dimension_raises(self, tmp_path):
        """Test error when backend missing dimension method."""
        from ragcore import RagEngine

        class NoDimension:
            def model_id(self) -> str:
                return "test"
            def embed(self, text: str) -> list[float]:
                return [0.0] * 768

        with pytest.raises(AttributeError, match="dimension"):
            RagEngine.create(str(tmp_path), backend=NoDimension())

    def test_missing_embed_raises(self, tmp_path):
        """Test error when backend missing embed method."""
        from ragcore import RagEngine

        class NoEmbed:
            def model_id(self) -> str:
                return "test"
            def dimension(self) -> int:
                return 768

        with pytest.raises(AttributeError, match="embed"):
            RagEngine.create(str(tmp_path), backend=NoEmbed())

    def test_missing_rerank_raises(self, tmp_path):
        """Test error when reranker missing rerank method."""
        from ragcore import RagEngine

        class NoRerank:
            pass

        embedder = SimplePyEmbedder()

        with pytest.raises(AttributeError, match="rerank"):
            RagEngine.create(str(tmp_path), backend=embedder, reranker=NoRerank())


class TestAsyncPythonBackend:
    """Test async Python backends (coroutine methods).

    Note: Async Python backends require a Python asyncio event loop to be
    running when called from Rust. This is complex to set up when the Rust
    side uses block_on(). These tests are skipped for now.

    Technical debt: TD-11 tracks implementing proper async Python backend support.
    """

    @pytest.mark.skip(reason="Async Python backends require asyncio event loop - see TD-11")
    def test_async_embedder(self, tmp_path):
        """Test async Python embedder."""
        from ragcore import RagEngine

        class AsyncEmbedder:
            def model_id(self) -> str:
                return "async-embedder"

            def dimension(self) -> int:
                return 64

            async def embed(self, text: str) -> list[float]:
                """Async embed method."""
                # In real use, this might await an HTTP call
                return [float(len(text))] + [0.5] * 63

        engine = RagEngine.create(str(tmp_path), backend=AsyncEmbedder())

        # Upsert should work with async embedder
        chunk_count = engine.upsert_document("async_test.txt", "Testing async embedder.")
        assert chunk_count > 0

        # Search should work
        results = engine.search("testing", top_k=5)
        assert len(results) > 0

    @pytest.mark.skip(reason="Async Python backends require asyncio event loop - see TD-11")
    def test_async_reranker(self, tmp_path):
        """Test async Python reranker."""
        from ragcore import RagEngine

        class AsyncReranker:
            async def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
                """Async rerank method."""
                return [
                    {"chunk_id": c["chunk_id"], "relevance": 0.9}
                    for c in candidates
                ]

        embedder = SimplePyEmbedder(dimension=32)
        engine = RagEngine.create(str(tmp_path), backend=embedder, reranker=AsyncReranker())

        # Upsert and search should work with async reranker
        engine.upsert_document("test.txt", "Test document content.")
        results = engine.search("test", top_k=5)
        assert len(results) > 0
