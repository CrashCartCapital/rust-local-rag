# Rust Local RAG

A local RAG (Retrieval-Augmented Generation) system built in Rust that integrates with Claude Desktop via the Model Context Protocol (MCP). Search and analyze PDF documents within Claude conversations without sending data to external services.

## Documentation Map

*   **[Setup](docs/setup.md)**: Installation and prerequisites.
*   **[Configuration](docs/configuration.md)**: Claude Desktop config, Environment Variables, and Model switching.
*   **[Usage](docs/usage.md)**: How to use the MCP tools in Claude.
*   **[Development](docs/development.md)**: Building, testing, and contributing.
*   **[Architecture](docs/architecture.md)**: High-level design and internals.

## Features

*   **Local Processing**: Privacy-first, no external APIs for document content.
*   **PDF Support**: Extracts text from PDFs (pure-Rust with fallback).
*   **Semantic Search**: Vector-based similarity search using Ollama embeddings.
*   **Reranking**: Optional LLM-based reranking for higher relevance.
*   **Reusable Core Library**: Core chunking/search/persistence lives in `crates/rag-core` for reuse in other projects.
*   **MCP Integration**: Seamless integration with Claude Desktop.

## License

MIT License - see [LICENSE](LICENSE) file.
