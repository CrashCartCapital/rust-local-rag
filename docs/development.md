# Development Guide

This guide covers building, running, and contributing to `rust-local-rag`.

## Prerequisites

*   Rust toolchain (stable)
*   Ollama (running)
*   Poppler (optional, for fallback PDF extraction)
*   Python >= 3.11 (for evaluation)

## Common Commands

We use `make` for common tasks.

```bash
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

# Testing
make test                   # Run tests
cargo test                  # Alternative

# Code Quality
make check                  # Cargo check
make lint                   # Clippy
make clippy                 # Clippy with warnings as errors
make fmt                    # Format code

# Maintenance
make update                 # Update dependencies
make fix                    # Auto-fix clippy issues

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

```bash
make setup-ollama           # Start Ollama + pull nomic-embed-text
make ollama-start           # Start Ollama server
make ollama-stop            # Stop Ollama server
make ollama-status          # Check status and list models
make ollama-models          # Pull required embedding model (nomic-embed-text)
```

> **Note:** The reranker model (`dengcao/Qwen3-Reranker-4B:Q5_K_M`) is optional but recommended. `make ollama-models` does not pull it to save bandwidth. Pull it manually with `ollama pull dengcao/Qwen3-Reranker-4B:Q5_K_M` if you want to use reranking.

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
*   `src/index_store.rs`: SQLite index management.
*   `src/job_manager.rs`: SQLite job persistence.
*   `src/progress_logger.rs`: Structured logging for progress events.
*   `src/worker.rs`: Background worker for indexing.
*   `crates/rag-core/`: Reusable core library: chunking, retrieval, scoring, persistence.
*   `crates/rag-core-py/`: Python bindings for the core library (used in eval).
*   `src/bin/rag_tui/`: TUI client application.

### Data & Configuration
*   `documents/`: Directory for PDF documents to be indexed.
*   `data/`: Local data storage (SQLite DB, embeddings).
*   `logs/`: Application logs.
*   `prompts/`: System prompts (e.g. for reranker).

### Documentation & Analysis
*   `docs/`: User and developer documentation.
*   `analysis/`: Product Requirements Documents (PRDs) and technical debt analysis.
*   `eval/`: Python evaluation harness and configs.

## Testing & Evaluation

### Rust Tests
```bash
cargo test
cargo test --bin rag-tui
```

### Evaluation Harness
Located in `eval/`. Requires Python >= 3.11.

```bash
# Install dependencies
uv pip install -e eval/  # or: pip install -e eval/

# Start server first
make run

# Run baseline evaluation
python -m eval.run evaluate --config baseline -v
```

See [Evaluation Framework Spec](RAG_EVALUATION_FRAMEWORK_SPEC.md) for details.
