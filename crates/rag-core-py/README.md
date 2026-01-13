# ragcore

Python bindings for [rag-core](../rag-core/), a batteries-included RAG (Retrieval-Augmented Generation) engine written in Rust.

## Features

- **Fast**: Rust-powered semantic search with Python convenience
- **Pluggable**: Bring your own embedding and reranking backends
- **Async-native**: Full async/await support for non-blocking operations
- **Persistence**: Save/load index to disk with automatic model change detection
- **Pickle support**: Serialize engines for multiprocessing workflows
- **Type-safe**: Complete type stubs for IDE support

## Installation

```bash
# From source (development)
cd crates/rag-core-py
pip install maturin
maturin develop

# Run tests
pip install pytest pytest-asyncio
pytest tests/
```

## Quick Start

### Mock Engine (Testing)

```python
from ragcore import RagEngine

# Create a mock engine for testing (no external dependencies)
engine = RagEngine.create_mock("/tmp/test-index", dimension=768)

# Index documents
engine.upsert_document("doc1.txt", "Python is a programming language.")
engine.upsert_document("doc2.txt", "Rust provides memory safety.")

# Search
results = engine.search("programming", top_k=5)
for r in results:
    print(f"{r.document}:{r.chunk_index} (score: {r.score:.3f})")
    print(f"  {r.text[:100]}...")
```

### Custom Embedding Backend

```python
from ragcore import RagEngine

class OllamaBackend:
    """Example backend using Ollama for embeddings."""

    def model_id(self) -> str:
        return "nomic-embed-text"

    def dimension(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        # Call your embedding service here
        import requests
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text}
        )
        return resp.json()["embedding"]

# Create engine with custom backend
engine = RagEngine.create("./my-index", backend=OllamaBackend())
engine.upsert_document("paper.pdf", extracted_text)
engine.save()
```

### Adding a Reranker

```python
from ragcore import RagEngine

class LLMReranker:
    """Example reranker using an LLM for relevance scoring."""

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        # candidates: [{"chunk_id": str, "text": str, "document": str, ...}, ...]
        scored = []
        for c in candidates:
            # Score each candidate (0.0-1.0)
            relevance = self._score_with_llm(query, c["text"])
            scored.append({"chunk_id": c["chunk_id"], "relevance": relevance})
        return scored

    def _score_with_llm(self, query: str, text: str) -> float:
        # Your LLM scoring logic here
        return 0.8

# Engine with both embedding backend and reranker
engine = RagEngine.create(
    "./index",
    backend=OllamaBackend(),
    reranker=LLMReranker()
)
```

### Async API

```python
import asyncio
from ragcore import RagEngine

async def main():
    engine = RagEngine.create_mock("/tmp/async-test")

    # Async document indexing
    chunks = await engine.aupsert_document("doc.txt", "Content here...")
    print(f"Indexed {chunks} chunks")

    # Async search
    results = await engine.asearch("query", top_k=5)
    for r in results:
        print(f"{r.document}: {r.score:.3f}")

asyncio.run(main())
```

### Advanced Search with QuerySpec

```python
from ragcore import RagEngine, QuerySpec

engine = RagEngine.create_mock("/tmp/test")
engine.upsert_document("doc.txt", "Some content...")

# Fine-tune search behavior
spec = QuerySpec(
    top_k=10,
    embedding_weight=0.7,    # Weight for embedding similarity
    lexical_weight=0.3,      # Weight for BM25/lexical scoring
    reranker_weight=0.7,     # Weight for reranker scores
    initial_weight=0.3,      # Weight for initial scores
    diversity_factor=0.3,    # MMR diversity (0.0 = no diversity)
)

results = engine.search("query", spec=spec)
```

### Persistence

```python
from ragcore import RagEngine

# Create and populate
engine = RagEngine.create_mock("/tmp/persistent")
engine.upsert_document("doc.txt", "Content...")
engine.save()

# Later: reload
engine2 = RagEngine.create_mock("/tmp/persistent")
engine2.load()
print(engine2.stats())  # Shows document_count, chunk_count, etc.

# Check if reindexing needed (model changed)
if engine2.needs_reindex():
    print("Embedding model changed, reindex required")
```

## API Reference

### RagEngine

| Method | Description |
|--------|-------------|
| `create(index_dir, backend, reranker=None)` | Create engine with Python backend |
| `create_mock(index_dir, dimension=768)` | Create engine with mock backend |
| `upsert_document(name, text, content_hash=None)` | Add/update document, returns chunk count |
| `aupsert_document(name, text, content_hash=None)` | Async version of upsert_document |
| `remove_document(name)` | Remove document, returns chunks removed |
| `search(query, top_k=10, spec=None)` | Search documents, returns list of SearchResult |
| `asearch(query, top_k=10, spec=None)` | Async version of search |
| `list_documents()` | List all indexed document names |
| `stats()` | Get EngineStats (counts, model info) |
| `save()` | Persist index to disk |
| `load()` | Load index from disk |
| `needs_reindex()` | Check if model changed since last index |
| `index_dir` | Property: path to index directory |

### SearchResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | str | Unique chunk identifier |
| `text` | str | Matched text content |
| `document` | str | Source document name |
| `chunk_index` | int | Index within document |
| `page_number` | int \| None | Page number if available |
| `section` | str \| None | Section heading if available |
| `score` | float | Final relevance score |
| `scores` | ScoreBreakdown | Detailed score components |

### ScoreBreakdown

| Attribute | Type | Description |
|-----------|------|-------------|
| `embedding` | float \| None | Embedding similarity score |
| `lexical` | float \| None | Lexical/BM25 score |
| `initial` | float \| None | Combined initial score |
| `reranker` | float \| None | Reranker relevance score |
| `total` | float | Final combined score |

### EngineStats

| Attribute | Type | Description |
|-----------|------|-------------|
| `document_count` | int | Number of indexed documents |
| `chunk_count` | int | Total number of chunks |
| `embedding_model` | str | Name of embedding model |
| `embedding_dimension` | int | Dimension of embeddings |

## Backend Protocol

### Embedding Backend (Required)

Your embedding backend must implement these methods:

```python
class EmbeddingBackend(Protocol):
    def model_id(self) -> str:
        """Return unique identifier for this model."""
        ...

    def dimension(self) -> int:
        """Return embedding vector dimension."""
        ...

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text. Can be sync or async."""
        ...

    # Optional: batch embedding for efficiency
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...
```

### Reranker Backend (Optional)

```python
class RerankerBackend(Protocol):
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """
        Rerank candidates by relevance to query.

        Args:
            query: Search query string
            candidates: List of dicts with keys:
                - chunk_id: str
                - text: str
                - document: str
                - chunk_index: int
                - page_number: int | None
                - section: str | None

        Returns:
            List of dicts with keys:
                - chunk_id: str (must match input)
                - relevance: float (0.0-1.0)
        """
        ...
```

## Exception Hierarchy

All ragcore exceptions inherit from `RagError`:

```python
from ragcore import (
    RagError,          # Base exception
    EmbeddingError,    # Embedding generation failed
    RerankError,       # Reranking failed
    IndexError,        # Index operations failed (save/load)
    ConfigError,       # Invalid configuration
    ValidationError,   # Input validation failed
)

try:
    engine.search("")
except ValueError as e:
    print("Empty query not allowed")

try:
    engine.upsert_document("doc.txt", "content")
except EmbeddingError as e:
    print(f"Embedding failed: {e}")

# Catch all ragcore errors
try:
    ...
except RagError as e:
    print(f"RAG error: {e}")
```

## Pickle Support

Engines can be pickled for use with multiprocessing:

```python
import pickle
from ragcore import RagEngine

# Mock engines pickle easily
engine = RagEngine.create_mock("/tmp/index", dimension=512)
pickled = pickle.dumps(engine)
restored = pickle.loads(pickled)

# Python backends: class must be at module level (not local)
class ModuleLevelBackend:
    def model_id(self) -> str: return "my-model"
    def dimension(self) -> int: return 768
    def embed(self, text: str) -> list[float]: return [0.0] * 768

engine = RagEngine.create("/tmp/index", backend=ModuleLevelBackend())
pickled = pickle.dumps(engine)  # Works!

# Note: Lambda-containing backends cannot be pickled (Python limitation)
```

## Type Hints

Full type stubs are provided in `ragcore/_native.pyi`. Your IDE should automatically pick these up for autocompletion and type checking.

```python
from ragcore import RagEngine, QuerySpec, SearchResult

def search_docs(engine: RagEngine, query: str) -> list[SearchResult]:
    spec = QuerySpec(top_k=5, diversity_factor=0.2)
    return engine.search(query, spec=spec)
```

## License

MIT
