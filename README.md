# Rust Local RAG

A local RAG (Retrieval-Augmented Generation) system built in Rust that integrates with Claude Desktop via the Model Context Protocol (MCP). Search and analyze PDF documents within Claude conversations without sending data to external services.

This is a maintained fork of [ksaritek/rust-local-rag](https://github.com/ksaritek/rust-local-rag) with additional features including configurable embedding models, background job processing, and model-partitioned storage.

## What It Does

- Processes PDF documents locally using pure-Rust lopdf (with poppler fallback)
- Generates embeddings using local Ollama models
- Provides semantic search through document collections
- Integrates with Claude Desktop via MCP protocol
- Keeps all data processing local

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is a standard that allows AI assistants like Claude to interact with external tools and data sources.

## Architecture

```
┌─────────────────┐    MCP Protocol     ┌──────────────────┐
│                 │    (stdin/stdout)   │                  │
│  Claude Desktop │ ◄─────────────────► │   Rust RAG       │
│                 │                     │   MCP Server     │
└─────────────────┘                     └──────────────────┘
                                               │
                                               ▼
                                       ┌──────────────────┐
                                       │  Local RAG Stack │
                                       │                  │
                                       │  • PDF Parser    │
                                       │  • Ollama        │
                                       │  • Vector Store  │
                                       │  • Search Engine │
                                       └──────────────────┘
```

## Features

### Semantic Document Search
- Vector-based similarity search using Ollama embeddings
- Two-stage retrieval: embedding similarity + LLM reranking
- **MMR Diversification**: Maximal Marginal Relevance algorithm balances relevance with result diversity
- Configurable result count (top-k, max 100) and diversity factor (0.0-1.0)

### Configurable Embedding Pipeline
- Select any installed Ollama embedding model via `OLLAMA_EMBEDDING_MODEL`
- Defaults to `nomic-embed-text`
- Validates model availability at startup

### Document Management
- Automatic PDF text extraction (pure-Rust lopdf, poppler fallback)
- Sentence-aware chunking with metadata
- Background job system for non-blocking document processing
- Automatic reindexing when embedding model changes

### Model-Partitioned Storage
- Each embedding model gets its own index file (`chunks_{model}.json`)
- Switch between models without losing previously computed embeddings
- Atomic file writes prevent data corruption

### Privacy
- All processing happens locally
- No external API calls for document content
- Embeddings stored locally

## Quick Start

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Ollama
brew install ollama

# (Optional) Install Poppler for fallback PDF parsing
brew install poppler

# Start Ollama and install embedding model
make setup-ollama
```

### Build and Install

```bash
git clone https://github.com/CrashCartCapital/rust-local-rag.git
cd rust-local-rag
make install
```

### Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
    "mcpServers": {
        "rust-local-rag": {
            "command": "/Users/YOUR_USERNAME/.cargo/bin/rust-local-rag",
            "env": {
                "DATA_DIR": "/Users/YOUR_USERNAME/Documents/data",
                "DOCUMENTS_DIR": "/Users/YOUR_USERNAME/Documents/rag",
                "LOG_DIR": "/tmp/rust-local-rag",
                "LOG_LEVEL": "info",
                "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text"
            }
        }
    }
}
```

Replace `YOUR_USERNAME` with your actual username.

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATA_DIR` | Embeddings and index storage | `./data` |
| `DOCUMENTS_DIR` | PDF source directory | `./documents` |
| `LOG_DIR` | Log output directory | Auto-detected |
| `LOG_LEVEL` | Tracing level (error/warn/info/debug/trace) | `info` |
| `LOG_MAX_MB` | Max log file size before truncation | `5` |
| `OLLAMA_URL` | Ollama API endpoint | `http://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `nomic-embed-text` |
| `OLLAMA_RERANK_MODEL` | LLM model for reranking (optional) | None (falls back to embedding-only) |
| `DEV` or `DEVELOPMENT` | Enable console logging | unset |

### Add Documents

```bash
# Add PDFs to documents directory
cp your-files.pdf ~/Documents/rag/

# Restart Claude Desktop
# Ask Claude: "Search my documents for information about X"
```

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_documents` | Semantic search with query, top_k (max 100), and diversity_factor (0.0-1.0) |
| `list_documents` | List all indexed documents |
| `get_stats` | System statistics (chunk count, embedding/reranker models) |
| `start_reindex` | Trigger background reindexing, returns job ID |
| `get_job_status` | Check job progress by ID |

### Search Parameters

- **query**: Search query string
- **top_k**: Number of results (default: 5, max: 100)
- **diversity_factor**: MMR diversity control (default: 0.3)
  - `0.0` = Pure relevance ranking (no diversity)
  - `0.3` = Balanced (recommended default)
  - `0.7+` = High diversity (reduces similar/duplicate results)

## Health Endpoints

- `/healthz` - Liveness probe (200 if process running)
- `/readyz` - Readiness probe (200 if engine lock acquirable within 100ms)

## Development

```bash
# Run in development mode
make run

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run linter
cargo clippy
```

## Documentation

- [Setup Guide](docs/setup.md) - Installation and configuration details
- [Usage Guide](docs/how-to-use.md) - Claude Desktop integration examples
- [Evaluation Framework](docs/RAG_EVALUATION_FRAMEWORK_SPEC.md) - RAG quality measurement
- [Reranker Guide](docs/RERANKER_DEBUGGING_POSTMORTEM.md) - LLM reranker implementation notes

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) - Protocol specification
- [rmcp](https://crates.io/crates/rmcp) - Rust MCP SDK
- [Ollama](https://ollama.ai/) - Local embedding generation
