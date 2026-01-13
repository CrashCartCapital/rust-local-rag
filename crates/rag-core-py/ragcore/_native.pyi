"""Type stubs for ragcore._native (Rust extension module)."""

from typing import Any, Awaitable, Optional

__version__: str

# ============================================================================
# Exceptions
# ============================================================================

class RagError(Exception):
    """Base exception for all ragcore errors."""
    ...

class EmbeddingError(RagError):
    """Raised when embedding generation fails."""
    ...

class RerankError(RagError):
    """Raised when reranking fails."""
    ...

class IndexError(RagError):
    """Raised when index operations fail (save/load/sync)."""
    ...

class ConfigError(RagError):
    """Raised when configuration is invalid."""
    ...

class ValidationError(RagError):
    """Raised when input validation fails."""
    ...

# ============================================================================
# Types
# ============================================================================

class QuerySpec:
    """Search configuration parameters.

    All parameters are keyword-only with sensible defaults.

    Attributes:
        top_k: Maximum number of results to return.
        embedding_weight: Weight for embedding similarity (0.0-1.0).
        lexical_weight: Weight for lexical/BM25 scoring (0.0-1.0).
        reranker_weight: Weight for reranker scores (0.0-1.0).
        initial_weight: Weight for initial scores (0.0-1.0).
        diversity_factor: MMR diversity factor (0.0-1.0, 0.0 = no diversity).
    """

    top_k: int
    embedding_weight: float
    lexical_weight: float
    reranker_weight: float
    initial_weight: float
    diversity_factor: float

    def __init__(
        self,
        *,
        top_k: int = 10,
        embedding_weight: float = 0.7,
        lexical_weight: float = 0.3,
        reranker_weight: float = 0.7,
        initial_weight: float = 0.3,
        diversity_factor: float = 0.0,
    ) -> None: ...

    def __repr__(self) -> str: ...

class ScoreBreakdown:
    """Detailed scoring components for a search result.

    Attributes:
        embedding: Embedding similarity score (if available).
        lexical: Lexical/BM25 score (if available).
        initial: Initial combined score before reranking.
        reranker: Reranker relevance score (if available).
        total: Final combined score.
    """

    embedding: Optional[float]
    lexical: Optional[float]
    initial: Optional[float]
    reranker: Optional[float]
    total: float

    def __repr__(self) -> str: ...

class SearchResult:
    """A single search result with metadata and scores.

    Attributes:
        id: Unique chunk identifier.
        text: The matched text content.
        document: Source document name.
        chunk_index: Index of this chunk within the document.
        page_number: Page number (if available from source).
        section: Section/heading (if available from source).
        score: Final relevance score (shortcut for scores.total).
        scores: Detailed score breakdown.
    """

    id: str
    text: str
    document: str
    chunk_index: int
    page_number: Optional[int]
    section: Optional[str]
    score: float
    scores: ScoreBreakdown

    def __repr__(self) -> str: ...

class EngineStats:
    """Engine health and statistics.

    Attributes:
        document_count: Number of indexed documents.
        chunk_count: Total number of text chunks.
        embedding_model: Name of the embedding model.
        embedding_dimension: Dimension of embedding vectors.
    """

    document_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimension: int

    def __repr__(self) -> str: ...

# ============================================================================
# Core Engine
# ============================================================================

class RagEngine:
    """RAG engine for semantic document search.

    The main entry point for Python applications. Provides semantic search
    over documents with pluggable embedding and reranking backends.

    Properties:
        index_dir: Path to the index directory.

    Examples:
        Create a mock engine for testing:

        >>> engine = RagEngine.create_mock("/tmp/index")
        >>> engine.upsert_document("doc.txt", "Hello world!")
        >>> results = engine.search("hello", top_k=5)

        Create an engine with a custom embedding backend:

        >>> class MyEmbedder:
        ...     def model_id(self) -> str: return "my-model"
        ...     def dimension(self) -> int: return 768
        ...     def embed(self, text: str) -> list[float]: ...
        >>> engine = RagEngine.create("/tmp/index", backend=MyEmbedder())
    """

    @property
    def index_dir(self) -> str:
        """Path to the index directory."""
        ...

    @staticmethod
    def create(
        index_dir: str,
        backend: Any,
        reranker: Optional[Any] = None,
    ) -> "RagEngine":
        """Create a new engine with a Python embedding backend.

        Args:
            index_dir: Directory for storing the index.
            backend: Python object implementing embedding backend protocol:
                - model_id() -> str
                - dimension() -> int
                - embed(text: str) -> list[float] (sync or async)
                - embed_batch(texts: list[str]) -> list[list[float]] (optional)
            reranker: Optional Python object implementing reranker protocol:
                - rerank(query: str, candidates: list[dict]) -> list[dict]

        Returns:
            A new RagEngine instance.

        Raises:
            AttributeError: If backend/reranker missing required methods.
        """
        ...

    @staticmethod
    def create_mock(
        index_dir: str,
        dimension: int = 768,
    ) -> "RagEngine":
        """Create a new engine with a mock backend (for testing).

        Args:
            index_dir: Directory for storing the index.
            dimension: Embedding dimension (default: 768).

        Returns:
            A new RagEngine instance with deterministic mock embeddings.
        """
        ...

    def stats(self) -> EngineStats:
        """Get engine statistics.

        Returns:
            EngineStats with current engine state.
        """
        ...

    def save(self) -> None:
        """Save the engine state to disk.

        Persists all indexed documents and embeddings to the index directory.

        Raises:
            IndexError: If persistence fails.
        """
        ...

    def load(self) -> None:
        """Load engine state from disk.

        Restores previously indexed documents and embeddings.

        Raises:
            IndexError: If loading fails.
        """
        ...

    def list_documents(self) -> list[str]:
        """List all indexed documents.

        Returns:
            List of document names currently in the index.
        """
        ...

    def needs_reindex(self) -> bool:
        """Check if the engine needs reindexing.

        Returns:
            True if the embedding model has changed since the last index.
        """
        ...

    def upsert_document(
        self,
        name: str,
        text: str,
        content_hash: Optional[str] = None,
    ) -> int:
        """Add or update a document in the index.

        The document text is automatically chunked and embedded.
        If a document with the same name exists, it is replaced.

        Args:
            name: Document name/identifier (e.g., "paper.pdf").
            text: Full text content of the document.
            content_hash: Optional content hash for change detection.
                If provided and matches the stored hash, the document is skipped.

        Returns:
            Number of chunks created (0 if skipped or empty).

        Raises:
            EmbeddingError: If embedding fails.
        """
        ...

    def remove_document(self, name: str) -> int:
        """Remove a document from the index.

        Args:
            name: Document name/identifier to remove.

        Returns:
            Number of chunks removed (0 if document not found).

        Raises:
            IndexError: If removal fails.
        """
        ...

    def search(
        self,
        query: str,
        top_k: int = 10,
        spec: Optional[QuerySpec] = None,
    ) -> list[SearchResult]:
        """Search for documents matching the query.

        Performs semantic search using the engine's embedding backend.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return (default: 10).
            spec: Optional QuerySpec for advanced configuration.

        Returns:
            List of SearchResult objects, sorted by score descending.

        Raises:
            ValueError: If query is empty.
            EmbeddingError: If embedding fails.
        """
        ...

    def asearch(
        self,
        query: str,
        top_k: int = 10,
        spec: Optional[QuerySpec] = None,
    ) -> Awaitable[list[SearchResult]]:
        """Async version of search.

        Returns a coroutine that can be awaited from async Python code.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return (default: 10).
            spec: Optional QuerySpec for advanced configuration.

        Returns:
            A coroutine that resolves to List[SearchResult].

        Raises:
            ValueError: If query is empty.
            EmbeddingError: If embedding fails.

        Examples:
            >>> async def search_async(engine):
            ...     results = await engine.asearch("query", top_k=5)
            ...     return results
        """
        ...

    def aupsert_document(
        self,
        name: str,
        text: str,
        content_hash: Optional[str] = None,
    ) -> Awaitable[int]:
        """Async version of upsert_document.

        Returns a coroutine that can be awaited from async Python code.

        Args:
            name: Document name/identifier.
            text: Full text content of the document.
            content_hash: Optional content hash for change detection.

        Returns:
            A coroutine that resolves to the number of chunks created.

        Raises:
            EmbeddingError: If embedding fails.

        Examples:
            >>> async def add_doc_async(engine):
            ...     chunks = await engine.aupsert_document("doc.txt", "Hello!")
            ...     return chunks
        """
        ...

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """Pickle support via __reduce__ protocol.

        Returns (callable, args) tuple for reconstruction.

        For mock engines: (RagEngine.create_mock, (index_dir, dimension))
        For Python backends: (RagEngine.create, (index_dir, backend, reranker))

        Note: Python backends must themselves be picklable (module-level classes)
        for the engine to be picklable.
        """
        ...

    def __repr__(self) -> str: ...
