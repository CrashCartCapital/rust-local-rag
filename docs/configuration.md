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
| `LOG_DIR` | Log output directory. Uses `/var/log/rust-local-rag` if `/var/log` exists, otherwise `./logs` | Auto-detected |
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
| `RAG_EMBEDDING_TIMEOUT_SECS` | Ollama embedding request timeout (seconds) | `1200` |
| `RAG_EMBEDDING_CACHE_SIZE` | LRU cache size for query embeddings | `1000` |
| `RAG_RERANKER_TIMEOUT_SECS` | Ollama reranker request timeout (seconds) | `60` |
| `RAG_RERANKER_CONCURRENCY` | Max concurrent reranker requests | `1` |
| `RAG_DEFAULT_LOGPROB` | Fallback logprob when missing (used for softmax scoring) | `-10.0` |
| `RAG_EMBEDDING_WEIGHT` | First-stage embedding similarity weight (0.0-1.0) | `0.7` |
| `RAG_LEXICAL_WEIGHT` | First-stage lexical/BM25 weight (0.0-1.0) | `0.3` |
| `RAG_RERANKER_WEIGHT` | Second-stage reranker weight (0.0-1.0) | `0.7` |
| `RAG_INITIAL_SCORE_WEIGHT` | Second-stage initial score weight (0.0-1.0) | `0.3` |
| `RAG_EMBEDDING_BATCH_SIZE` | Number of chunks to embed in a single batch | `32` |

> 💡 **Tip**: Set `OLLAMA_EMBEDDING_MODEL` to any embedding model you've installed. The server verifies your choice at startup.

## Model Switching

You can switch between different Ollama embedding models without losing your indexed data.

1.  **Pull new model**: `ollama pull mxbai-embed-large`
2.  **Update Config**: Change `OLLAMA_EMBEDDING_MODEL` to `mxbai-embed-large`.
3.  **Restart**: Restart Claude Desktop.

Each model gets its own partition inside the SQLite index store in `DATA_DIR/jobs.db` (keyed by `model_id`). Switching back to a previous model restores its existing index. If you have legacy JSON index files (`chunks_{model}.json`), they are imported automatically on first run and renamed to `.migrated.bak`.
