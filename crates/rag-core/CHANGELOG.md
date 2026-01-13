# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-12

### Added

- Initial release of rag-core as a standalone library
- **Core Engine**
  - `RagEngine<B, R>` - Main entry point with pluggable embedding and reranking backends
  - Sentence-aware text chunking with configurable overlap
  - Hybrid search combining embedding similarity and lexical (BM25-style) scoring
  - MMR (Maximal Marginal Relevance) diversification for search results

- **Traits**
  - `EmbeddingBackend` - Pluggable trait for embedding providers (Ollama, OpenAI, etc.)
  - `Rerank` - Optional trait for second-stage LLM-based reranking
  - `PersistenceBackend` - Pluggable storage with `JsonFileBackend` default

- **Error Handling**
  - Structured `EngineError` with operation context
  - `ValidationKind` for input validation failures (NaN, Inf, dimension mismatch)
  - `PersistenceError` with path and operation context

- **Input Validation**
  - NaN/Inf detection in embeddings
  - Dimension mismatch enforcement
  - Empty text rejection
  - Atomic batch operations with validate-before-commit pattern

- **Persistence** (feature-gated: `persistence`)
  - `JsonFileBackend` with atomic writes (temp file + rename)
  - `EngineState` for save/load operations
  - Legacy format migration support
  - Model-partitioned storage (`chunks_{model}.json`)

- **Observability** (feature-gated: `tracing`)
  - Tracing spans for major operations

### Documentation

- Comprehensive README with architecture overview
- Rustdoc for all public types and traits
- Examples: `basic_usage.rs`, `custom_backend.rs`, `persistence.rs`

### Migration Guide

This is the initial release. For users migrating from the `rust-local-rag` main crate:

1. Add `rag-core` as a dependency instead of copying internal modules
2. Implement `EmbeddingBackend` for your embedding service
3. Use `RagEngine::new(embedder)` or `RagEngine::with_config(embedder, config)`
4. Enable `persistence` feature for save/load functionality

[Unreleased]: https://github.com/ryanpappal/rust-local-rag/compare/rag-core-v0.1.0...HEAD
[0.1.0]: https://github.com/ryanpappal/rust-local-rag/releases/tag/rag-core-v0.1.0
