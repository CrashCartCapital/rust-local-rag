# Development Guide

This guide covers building, running, and contributing to `rust-local-rag`.

## Prerequisites

*   Rust toolchain (stable)
*   Ollama (running)
*   Poppler (optional, for fallback PDF extraction)

## Common Commands

We use `make` for common tasks.

```bash
# Development run (console logging)
make run

# Build only
make build                  # Debug build
make release                # Optimized release build

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
make install                # Install debug binary to ~/.cargo/bin
make install-release        # Install optimized binary
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
make ollama-models          # Pull required models
```

## Repository Layout

*   `src/main.rs`: Entrypoint; env + logging; initializes engine + job system.
*   `src/lib.rs`: Library exports.
*   `src/mcp/`: MCP implementation (tools, HTTP server, formatting).
*   `src/mcp_server.rs`: Re-exports MCP modules.
*   `crates/rag-core/src/*`: Reusable core library: chunking, retrieval, scoring, persistence.
*   `src/rag_engine.rs`: Server wrapper: PDF extraction + env/config + calls into `rag-core`.
*   `src/embeddings.rs`: Ollama embeddings client.
*   `src/reranker.rs`: Ollama-based reranker.
*   `src/job_manager.rs`: SQLite job persistence.
*   `src/worker.rs`: Background worker for indexing.
*   `src/config.rs`: Configuration loading.
*   `src/bin/rag_tui/`: TUI client.
*   `docs/*`: Documentation.
*   `eval/*`: Python evaluation harness.

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
