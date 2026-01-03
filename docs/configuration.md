# Configuration

## Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

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

**Important**: Replace `yourusername` with your actual username. Use absolute paths for reliable operation.

## Environment Variables

All configuration values can be defined in the `env` section of `claude_desktop_config.json`, or in a `.env` file alongside the binary.

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Embeddings and index storage directory | `./data` |
| `DOCUMENTS_DIR` | PDF documents directory to scan | `./documents` |
| `LOG_DIR` | Log output directory. Uses `/var/log/rust-local-rag` when writable, otherwise `./logs` | Auto-detected |
| `LOG_LEVEL` | Logging level (`error`, `warn`, `info`, `debug`, `trace`) | `info` |
| `LOG_MAX_MB` | Max log file size in MB before truncation | `5` |
| `OLLAMA_URL` | Base URL for the Ollama API | `http://localhost:11434` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model name (must be installed via `ollama pull`) | `nomic-embed-text` |
| `OLLAMA_RERANK_MODEL` | LLM model for reranking search results | `dengcao/Qwen3-Reranker-4B:Q5_K_M` |
| `PROMPTS_DIR` | Directory for prompt template overrides | `./prompts` |
| `MCP_HTTP_BIND` | HTTP health endpoint address | `127.0.0.1:8140` |
| `MCP_HTTP_ENDPOINT` | HTTP MCP endpoint path | `/mcp` |
| `DEVELOPMENT` or `DEV` | Prefer console logging (development friendly) | _unset_ |
| `CONSOLE_LOGS` | Force console logging regardless of environment | _unset_ |

> 💡 **Tip**: Set `OLLAMA_EMBEDDING_MODEL` to any embedding model you've installed. The server verifies your choice at startup.

## Model Switching

You can switch between different Ollama embedding models without losing your indexed data.

1.  **Pull new model**: `ollama pull mxbai-embed-large`
2.  **Update Config**: Change `OLLAMA_EMBEDDING_MODEL` to `mxbai-embed-large`.
3.  **Restart**: Restart Claude Desktop.

Each model gets its own index file (e.g., `chunks_nomic-embed-text.json`, `chunks_mxbai-embed-large.json`). Switching back to a previous model instantly restores its index.
