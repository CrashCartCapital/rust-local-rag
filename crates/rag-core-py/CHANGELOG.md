# Changelog

All notable changes to ragcore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-13

### Added

- Initial release of ragcore Python bindings for rag-core
- `RagEngine` class with full RAG functionality
  - `create()` - Create engine with custom Python embedding backend
  - `create_mock()` - Create engine with mock backend for testing
  - `upsert_document()` / `aupsert_document()` - Sync/async document indexing
  - `search()` / `asearch()` - Sync/async semantic search
  - `remove_document()` - Remove documents from index
  - `save()` / `load()` - Persist and restore index state
  - `list_documents()` - List indexed document names
  - `stats()` - Get engine statistics
  - `needs_reindex()` - Check if model changed
- `QuerySpec` class for advanced search configuration
  - `embedding_weight` / `lexical_weight` - First-stage scoring weights
  - `reranker_weight` / `initial_weight` - Second-stage scoring weights
  - `diversity_factor` - MMR diversity factor
- `SearchResult` class with detailed score breakdown
- Exception hierarchy: `RagError`, `EmbeddingError`, `RerankError`, `IndexError`, `ConfigError`, `ValidationError`
- Full async/await support via `pyo3-async-runtimes`
- Pickle support for multiprocessing workflows
- Complete type stubs (`.pyi`) for IDE support
- PEP 561 compliance (`py.typed` marker)
- Signal handling (GIL release during long operations)
- Pluggable embedding backend protocol
- Optional reranker backend protocol

### Backend Protocol

Embedding backends must implement:
- `model_id() -> str`
- `dimension() -> int`
- `embed(text: str) -> list[float]`
- `embed_batch(texts: list[str]) -> list[list[float]]` (optional)

Reranker backends must implement:
- `rerank(query: str, candidates: list[dict]) -> list[dict]`

### Known Limitations

- Async Python backends (backends with async embed/rerank methods) are not yet supported
- Python backends must be picklable (module-level classes) for pickle support
