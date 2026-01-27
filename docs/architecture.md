# Architecture & Design

`rust-local-rag` is a local Retrieval-Augmented Generation system that integrates with Claude Desktop via the Model Context Protocol (MCP).

## High-Level Architecture

```
┌─────────────────┐   MCP Streamable HTTP   ┌──────────────────┐
│                 │   (JSON-RPC over HTTP)  │                  │
│  Claude Desktop │ ◄─────────────────────► │   Rust RAG       │
│                 │   http://host:port/mcp  │   MCP Server     │
└─────────────────┘                         └──────────────────┘
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

## Core Components

1.  **MCP Server (`src/mcp_server.rs`)**: Handles communication with Claude Desktop. Exposes tools like `search_documents`.
2.  **RAG Core (`crates/rag-core/src/*`)**: Reusable engine: chunking, search (cosine + MMR), and persistence format.
3.  **RAG Server Wrapper (`src/rag_engine.rs`)**: Server-only glue: PDF extraction + env/config + calls into `rag-core`.
4.  **Job System (`src/job_manager.rs`, `src/worker.rs`)**: Handles long-running tasks (like indexing) in the background to avoid blocking the MCP server.
5.  **Embedding Service (`src/embeddings.rs`)**: Interfaces with Ollama to generate vector embeddings.
6.  **Reranker (`src/reranker.rs`)**: Optional second-stage reranking using an LLM to improve search relevance.

## PDF Processing Pipeline

1.  **Discovery**: Scans `DOCUMENTS_DIR` for PDF files.
2.  **Extraction**: Extracts text using `lopdf` (pure Rust). Falls back to `pdftotext` (poppler) if `lopdf` fails.
3.  **Chunking**: Splits text into sentence-aware chunks (~500-1000 chars) with metadata (page number, section).
4.  **Embedding**: Generates embeddings using the configured Ollama model.
5.  **Indexing**: Stores chunks + embeddings in SQLite (`DATA_DIR/jobs.db`) partitioned by `model_id` (`rag_models`, `rag_documents`, `rag_chunks`). Legacy JSON indexes (`chunks_{model}.json`) are auto-migrated once and renamed to `.migrated.bak`.
6.  **Fingerprinting**: Uses SHA-256 hashes to detect unchanged files and skip re-processing.

## Search & Retrieval

1.  **Vector Search**: Finds top-k chunks with highest cosine similarity to the query embedding.
2.  **Reranking (Optional)**: If enabled, a reranker model (e.g., `dengcao/Qwen3-Reranker-4B:Q5_K_M`) evaluates the relevance of the top candidates.
3.  **Diversification**: Uses MMR (Maximal Marginal Relevance) to balance relevance with diversity in results.

## Job-Based Processing

To prevent timeouts in the MCP protocol, long-running operations like reindexing run as background jobs.

*   **Atomic Creation**: Jobs are created atomically in a SQLite database.
*   **Worker**: A background thread processes the job queue.
*   **Status**: Clients poll `get_job_status` to track progress.

## HTTP Surface

The server runs a single HTTP listener (configurable via `MCP_HTTP_BIND`) that serves:

- MCP endpoint: `MCP_HTTP_ENDPOINT` (default: `/mcp`)
- Health: `GET /healthz` (alias: `GET /health`) and `GET /readyz`
- Evaluation endpoints: `POST /search`, `GET /stats`, `GET /documents`, `POST /reindex`, `GET /jobs/active`, `GET /jobs/{job_id}`
