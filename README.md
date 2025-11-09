# Rust Local RAG

This repository is a maintained fork of [ksaritek/rust-local-rag](https://github.com/ksaritek/rust-local-rag) that adds configurability for choosing which Ollama embedding model to use.
It remains a high-performance, local RAG (Retrieval-Augmented Generation) system built in Rust that integrates with Claude Desktop via the Model Context Protocol (MCP).
Search and analyze your PDF documents directly within Claude conversations without sending data to external services.

## 🎯 Purpose

This project demonstrates how to build a production-ready MCP server using Rust that:

- **Processes PDF documents locally** using poppler for text extraction
- **Generates embeddings** using your preferred local Ollama embedding model (no external API calls)
- **Provides semantic search** through document collections
- **Integrates seamlessly** with Claude Desktop via MCP protocol
- **Maintains privacy** by keeping all data processing local

## 🏗️ What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is a standard that allows AI assistants like Claude to interact with external tools and data sources. Instead of Claude being limited to its training data, MCP enables it to:

- Call external tools and functions
- Access real-time data sources  
- Integrate with local applications
- Maintain context across interactions

## 🦀 How This Project Uses Rust MCP SDK

This implementation leverages the [`rmcp`](https://crates.io/crates/rmcp) crate - the official Rust SDK for MCP - to create a server that exposes RAG capabilities to Claude Desktop.

### MCP Architecture in This Project

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

### Key MCP Components Used

#### 1. **Server Handler Implementation**
```rust
#[tool(tool_box)]
impl ServerHandler for RagMcpServer {
    fn get_info(&self) -> ServerInfo {
        // Provides server metadata to Claude
    }
}
```

#### 2. **Tool Definitions**
Uses `rmcp` macros to expose RAG functionality as MCP tools:

```rust
#[tool(description = "Search through uploaded documents using semantic similarity")]
async fn search_documents(&self, query: String, top_k: Option<usize>) -> Result<CallToolResult, McpError>

#[tool(description = "List all uploaded documents")]  
async fn list_documents(&self) -> Result<CallToolResult, McpError>

#[tool(description = "Get RAG system statistics")]
async fn get_stats(&self) -> Result<CallToolResult, McpError>
```

#### 3. **Transport Layer**
```rust
// Uses stdin/stdout transport for Claude Desktop integration
let service = server.serve(stdio()).await?;
```

## ✨ Features

### 🔍 **Semantic Document Search**
- Vector-based similarity search using Ollama embeddings
- Configurable result count (top-k)
- Relevance scoring for search results

### 🔧 **Customizable Embedding Pipeline**
- Select any installed Ollama embedding model with the `OLLAMA_EMBEDDING_MODEL` environment variable
- Defaults to `nomic-embed-text` for quick setup, but works with any compatible local model
- Validates your selection against the models available in your Ollama installation at startup

### 📁 **Document Management**
- Automatic PDF text extraction via poppler
- Document chunking for optimal embedding generation
- Automatic reindexing when your Ollama embedding model changes
- Real-time document list and statistics

### 🔒 **Privacy-First Design**
- All processing happens locally
- No external API calls for document content
- Embeddings stored locally for fast retrieval
- Background log maintenance with local rotation/truncation only

### ⚡ **High Performance**
- Rust's memory safety and performance
- Async/await for non-blocking operations
- Efficient vector storage and retrieval

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Ollama
brew install ollama

# Install Poppler (for PDF parsing)
brew install poppler

# Start Ollama and install embedding model
make setup-ollama
```

### 2. Build and Install
```bash
git clone <this-repository>
cd rust-local-rag
make install
```

### 3. Configure Claude Desktop
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
    "mcpServers": {
        "rust-local-rag": {
            "command": "/Users/yourusername/.cargo/bin/rust-local-rag",
            "env": {
                "DATA_DIR": "/Users/yourusername/Documents/data",
                "DOCUMENTS_DIR": "/Users/yourusername/Documents/rag",
                "LOG_DIR": "/tmp/rust-local-rag",
                "LOG_LEVEL": "info",
                "LOG_MAX_MB": "10",
                "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text"
            }
        }
    }
}
```

### Choose Your Ollama Embedding Model

Set the `OLLAMA_EMBEDDING_MODEL` environment variable to any embedding model you've pulled into your Ollama installation.
For example:

```bash
ollama pull snowflake-arctic-embed
export OLLAMA_EMBEDDING_MODEL=snowflake-arctic-embed
rust-local-rag
```

If the model is not installed, the server will provide a helpful error listing the available models and how to pull the requested one.

### Environment Configuration

The binary reads configuration from environment variables (and optional `.env` files). Most installs can start with the defaults, but you can customise behaviour with the following values:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATA_DIR` | Where embeddings and index metadata are persisted | `./data` |
| `DOCUMENTS_DIR` | Directory scanned for PDFs to index | `./documents` |
| `LOG_DIR` | Log output directory. Uses `/var/log/rust-local-rag` when writable, otherwise falls back to `./logs`. | Auto-detected |
| `LOG_LEVEL` | Tracing level (`error`, `warn`, `info`, `debug`, `trace`) | `info` |
| `LOG_MAX_MB` | Maximum log file size before automatic truncation | `5` |
| `OLLAMA_URL` | Ollama base URL | `http://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model to request from Ollama | `nomic-embed-text` |
| `DEVELOPMENT` / `DEV` | When set, forces pretty console logging (skips log file) | _unset_ |
| `CONSOLE_LOGS` | Force console logging without enabling other dev toggles | _unset_ |

> ℹ️  Log files are automatically truncated when they exceed `LOG_MAX_MB`. Set `CONSOLE_LOGS=true` if you prefer to view logs directly in the terminal even outside development mode.

### 4. Add Documents and Use
```bash
# Add PDFs to documents directory
cp your-files.pdf ~/Documents/rag/

# Restart Claude Desktop
# Now ask Claude: "Search my documents for information about X"
```

## 🏛️ Architecture

### Technology Stack
- **🦀 Rust**: Core application language for performance and safety
- **📡 rmcp**: Official Rust MCP SDK for Claude integration  
- **🤖 Ollama**: Local embedding generation (nomic-embed-text)
- **📄 Poppler**: PDF text extraction
- **🗃️ Custom Vector Store**: In-memory vector database for fast search

### Data Flow
1. **Document Ingestion**: PDFs → Text extraction → Chunking
2. **Embedding Generation**: Text chunks → Ollama → Vector embeddings  
3. **Indexing**: Embeddings → Local vector store
4. **Search**: Query → Embedding → Similarity search → Results
5. **MCP Integration**: Results → Claude Desktop via MCP protocol

## 🛠️ MCP Integration Details

### Why MCP Over HTTP API?

| Aspect | MCP Approach | HTTP API Approach |
|--------|-------------|------------------|
| **Integration** | Native Claude Desktop support | Requires custom client |
| **Security** | Process isolation, no network | Network exposure required |
| **Performance** | Direct stdin/stdout IPC | Network overhead |
| **User Experience** | Seamless tool integration | Manual API management |

### MCP Tools Exposed

1. **`search_documents`**
   - **Purpose**: Semantic search across document collection
   - **Input**: Query string, optional result count
   - **Output**: Ranked search results with similarity scores

2. **`list_documents`**  
   - **Purpose**: Document inventory management
   - **Input**: None
   - **Output**: List of all indexed documents

3. **`get_stats`**
   - **Purpose**: System monitoring and debugging
   - **Input**: None  
   - **Output**: Embedding counts, memory usage, performance metrics

## 📚 Documentation

- **[Setup Guide](setup.md)**: Complete installation and configuration
- **[Usage Guide](how-to-use.md)**: Claude Desktop integration and usage examples

## 🤝 Contributing

Contributions are welcome! This project demonstrates practical MCP server implementation patterns that can be adapted for other use cases.

### Development
```bash
# Run in development mode
make run

# Check formatting
cargo fmt --check

# Run linter
cargo clippy
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Model Context Protocol](https://modelcontextprotocol.io/)** for the specification
- **[rmcp](https://crates.io/crates/rmcp)** for the excellent Rust MCP SDK
- **[Ollama](https://ollama.ai/)** for local embedding generation
- **Claude Desktop** for MCP integration support

---

**Built with ❤️ in Rust | Powered by MCP | Privacy-focused RAG** 
