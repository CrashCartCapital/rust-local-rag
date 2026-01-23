# Development Guide

This guide covers building, running, and contributing to `rust-local-rag`.

## Prerequisites

*   Rust toolchain (stable)
*   Ollama (running)
*   Poppler (optional, for fallback PDF extraction)

## Common Commands

We use `make` for common tasks.

```bash
# Setup & Maintenance
make setup                  # Complete development environment setup
make update                 # Update dependencies within constraints
make upgrade                # Upgrade to latest versions (including breaking changes)
make install-tools          # Install required development tools
make fix                    # Auto-fix clippy issues

# Development
make run                    # Run application (DEV=true)
make dev-start              # Start Ollama + Run application
make watch                  # Watch for changes and run checks
make logs                   # View application logs
make kill                   # Kill all rust-local-rag processes

# Build
make build                  # Debug build
make release                # Optimized release build
make clean                  # Clean build artifacts
make clean-build            # Clean Rust build cache only
make clean-all              # Comprehensive cleanup (build + models)

# Testing
make test                   # Run tests
cargo test                  # Alternative

# Code Quality
make check                  # Cargo check
make lint                   # Clippy
make clippy                 # Clippy with warnings as errors
make fmt                    # Format code

# Full CI Pipeline
make ci                     # check + lint + test + build
```

## Installation

```bash
make install                # Install release binary to ~/.cargo/bin
make install-release        # Install optimized release binary (explicit profile)
make install-production     # Build release + install
make uninstall              # Remove from system
make which-installed        # Check if installed and location
```

## Ollama Management

Helpers for managing the Ollama dependency.

**Note:** These commands primarily manage the embedding model (`nomic-embed-text`) which is required. The reranker model is optional and must be pulled manually if desired (see [Setup](setup.md)).

```bash
make setup-ollama           # Start Ollama + pull embedding model
make ollama-start           # Start Ollama server
make ollama-stop            # Stop Ollama server
make ollama-status          # Check status and list models
make ollama-models          # Pull required embedding model
make clean-ollama           # Clean Ollama models (WARNING: Re-download needed)
```

## Repository Layout

### Source Code
*   `src/main.rs`: Application entrypoint; env + logging; initializes engine + job system.
*   `src/lib.rs`: Library definitions and module exports.
*   `src/config.rs`: Centralized configuration loading.
*   `src/mcp_server.rs`: MCP server entrypoint (re-exports `src/mcp`).
*   `src/mcp/`: MCP implementation (HTTP, tools, models).
*   `src/rag_engine.rs`: Server wrapper: PDF extraction + env/config + calls into `rag-core`.
*   `src/embeddings.rs`: Ollama embeddings client.
*   `src/reranker.rs`: Ollama-based reranker.
*   `src/job_manager.rs`: SQLite job persistence.
*   `src/job_payload.rs`: Job data structures.
*   `src/index_store.rs`: SQLite index management (Chunks/Embeddings).
*   `src/worker.rs`: Background worker for indexing.
*   `src/progress_logger.rs`: Progress logging utilities.
*   `crates/rag-core/`: Reusable core library: chunking, retrieval, scoring, persistence.
*   `src/bin/rag_tui/`: TUI client application.

### Data & Configuration
*   `documents/`: Directory for PDF documents to be indexed.
*   `data/`: Local data storage (SQLite DB, embeddings).
*   `logs/`: Application logs.
*   `prompts/`: System prompts (e.g. for reranker).
*   `active-prd/`: Active Product Requirements Documents.
*   `.archive/`: Archived documents and code.

### Documentation & Analysis
*   `docs/`: User and developer documentation.
*   `analysis/`: Technical debt analysis and older PRDs.
*   `eval/`: Python evaluation harness and configs.
*   `frontend/`: Frontend application code.
*   `tests/`: Integration tests.

## Testing & Evaluation

### Rust Tests
```bash
cargo test
cargo test --bin rag-tui
```

### Evaluation Harness
Located in `eval/`. Requires Python >= 3.11.

```bash
# Start server first
make run

# Run baseline evaluation
python -m eval.run evaluate --config baseline -v
```

See [Evaluation Framework Spec](RAG_EVALUATION_FRAMEWORK_SPEC.md) for details.
