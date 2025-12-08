# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential References

- **MCP Tools & AI Ensemble**: See @~/.claude/MCP_TOOLS_REF.md for complete tool inventory, AI ensemble patterns, and workflow methodology
- **MCP Configuration**: See @.mcp.json for mcpjungle connection setup

## 🚨 CRITICAL: mcpjungle Operational Requirements

**NEVER restart mcpjungle from any directory other than `~/03_CODE`**

When restarting mcpjungle server, you MUST:
1. Kill the existing process: `pkill -f "mcpjungle.*start"`
2. Change to the correct directory: `cd ~/03_CODE`
3. Start mcpjungle: `nohup mcpjungle start > /tmp/mcpjungle.log 2>&1 &`

**Why this matters**: mcpjungle uses relative paths for configuration and must be started from ~/03_CODE to correctly resolve MCP server definitions and toolsets.

**DO NOT**:
- ❌ Start mcpjungle from `~/05_RESOURCES/mcpjungle-config`
- ❌ Start mcpjungle from any project directory
- ❌ Change directory in the middle of the restart command

**CORRECT PATTERN**:
```bash
pkill -f "mcpjungle.*start"
sleep 2
cd ~/03_CODE
nohup mcpjungle start > /tmp/mcpjungle.log 2>&1 &
```

### Quick MCP Tools for Rust Development

**AI Consultants (Free Models - Use These!)**:
- `ask-gemini` (model: gemini-3-pro-preview) - Large context, architecture validation, async/concurrency review (PRIMARY)
- `consult_codex` (timeout: 300+) - Code review, performance analysis, implementation guidance
- `ask-qwen` (model: qwen3-coder-plus) - Rust-specialized analysis, quick validation

**Code Analysis**:
- `distill_file` - Extract structure from Rust files
- `grep_files` - Search codebase with regex (fast, exact matches)
- `directory_tree` - View project structure

**Research**:
- `tavily-search` - Web research for Rust patterns, crate documentation
- `githubSearchCode` - Find Rust examples on GitHub

## Project Overview

This is a Rust-based MCP (Model Context Protocol) server that implements a local RAG (Retrieval-Augmented Generation) system. It processes PDF documents locally, generates embeddings using Ollama, and provides semantic search capabilities to Claude Desktop via the MCP protocol.

## MCP Tool Access via ccc-code-mode

**Note**: This project (rust-local-rag) IS an MCP server that provides RAG tools to Claude Desktop. The ccc-code-mode tool mentioned here is SEPARATE - it's a workflow orchestrator that gives Claude Code access to MCP tools during development of this Rust project.

### 🚀 Quick Reference Guide

**For complete workflow tool documentation**, see:
@/Users/ryanpappal/03_CODE/ccc-code-mode/docs/CODE_MODE_QUICK_REFERENCE.md

This guide contains:
- Fast task-to-tool selector
- All 4 workflow tools (search/execute/save workflows, execute code)
- Available MCP tools inside workflows
- Common patterns for Rust development
- Copy-paste examples

### Quick Start (Read This First!)

**Most Common Pattern for Claude Code**:

```typescript
// 1. Need to do something? Search for existing workflows FIRST
search_workflows({ tags: ["rust", "relevant-tag"], limit: 5 })

// 2a. If found: Execute it
execute_workflow({ id: "workflow-id-from-search", input: {...} })

// 2b. If NOT found: Create ad-hoc workflow with execute_code
execute_code({
  code: `
    export default async function doTask(input) {
      // Use tools like: tools["grep_files"], tools["ask-gemini"], etc.
      const result = await tools["tool-name"]({ ...params });
      return { result };
    }
  `,
  input: { your: "data" }
})

// 3. If you'll use this workflow again: Save it
save_workflow({
  name: "Descriptive Name",
  tags: ["rust", "category"],
  code: "... same code as above ..."
})
```

**Key Rules**:
- ✅ ALWAYS use `search_workflows` before creating new workflows
- ✅ Use `execute_code` for ad-hoc tasks (one-time workflows)
- ✅ Use `save_workflow` for reusable workflows
- ✅ Access MCP tools via `tools["tool-name"]({ params })`
- ❌ DON'T use bash/ls to explore ccc-code-mode - use the workflow tools directly!

### What is ccc-code-mode?

**ccc-code-mode** is a TypeScript/Deno workflow execution engine located at `/Users/ryanpappal/03_CODE/ccc-code-mode`. It orchestrates MCP tools (filesystem, web search, AI consultants, etc.) via type-safe, composable workflows.

**Two Ways to Use It**:

1. **Via MCP Tools** (Recommended) - Use the workflow library tools directly from Claude Code
2. **Via CLI** - Execute workflow files manually (less common)

### Workflow Library MCP Tools (USE THESE!)

ccc-code-mode provides **4 MCP tools** for workflow management that Claude Code can call directly:

#### 1. `search_workflows` - Find existing workflows
**When to use**: ALWAYS search BEFORE creating new workflows

```typescript
// Search by tags
search_workflows({ tags: ["rust", "debugging"], limit: 5 })

// Search by description
search_workflows({ query: "cargo error analysis", limit: 10 })

// Search by tools used
search_workflows({ toolsUsed: ["tavily-search", "ask-gemini"] })
```

#### 2. `execute_workflow` - Run a saved workflow
**When to use**: After finding a workflow via search

```typescript
// Execute by workflow ID
execute_workflow({
  id: "workflow-uuid-here",
  input: { error_message: "cargo build error E0308" }
})
```

#### 3. `save_workflow` - Save a new workflow
**When to use**: After creating a new workflow that should be reusable

```typescript
save_workflow({
  name: "Rust Error Analyzer",
  description: "Analyzes Rust compiler errors with multi-model validation",
  tags: ["rust", "debugging", "error-analysis"],
  toolsUsed: ["grep_files", "ask-gemini", "consult_codex"],
  pattern: "debugging",
  code: `
    export default async function analyzeRustError(input: { error: string }) {
      // Workflow implementation
      return { analysis: "..." };
    }
  `
})
```

#### 4. `execute_code` - Run ad-hoc workflow code
**When to use**: One-off workflows that don't need to be saved

```typescript
execute_code({
  code: `
    export default async function quickSearch(input: { pattern: string }) {
      const files = await tools.grep_files({ pattern: input.pattern });
      return { found: files };
    }
  `,
  input: { pattern: "Arc<RwLock" }
})
```

### Workflow Creation Pattern (IMPORTANT!)

**ALWAYS follow this pattern**:

```
1. Search for existing workflows FIRST
   → search_workflows({ tags: ["relevant", "tags"] })

2. If found: Execute it
   → execute_workflow({ id: "...", input: {...} })

3. If NOT found: Create new workflow
   → Write TypeScript code
   → Use execute_code for testing OR save_workflow for reuse
```

### Practical Workflow Examples

#### Example 1: Analyze a Cargo Error (Using execute_code)

```typescript
// One-off analysis - use execute_code
execute_code({
  code: `
    export default async function analyzeCargoError(input: { errorText: string }) {
      // Get AI analysis from multiple models
      const geminiAnalysis = await tools["ask-gemini"]({
        prompt: "Explain this Rust error and how to fix it: " + input.errorText,
        model: "gemini-3-pro-preview"
      });

      const codexAnalysis = await tools["consult_codex"]({
        query: "How to fix this Rust error: " + input.errorText,
        timeout: 300
      });

      return {
        gemini: geminiAnalysis,
        codex: codexAnalysis
      };
    }
  `,
  input: { errorText: "error[E0308]: mismatched types..." }
})
```

#### Example 2: Search for Existing Workflows

```typescript
// ALWAYS search first before creating new workflows!
search_workflows({
  tags: ["rust", "testing"],
  query: "tokio async test patterns",
  limit: 5
})

// If found, execute it:
execute_workflow({
  id: "found-workflow-id",
  input: { test_file: "src/embeddings.rs" }
})
```

#### Example 3: Create and Save a Reusable Workflow

```typescript
// For frequently-used tasks, save the workflow
save_workflow({
  name: "Rust Dependency Analyzer",
  description: "Searches GitHub and web for Rust crate alternatives and best practices",
  tags: ["rust", "dependencies", "research"],
  toolsUsed: ["githubSearchRepositories", "tavily-search"],
  pattern: "research",
  code: `
    export default async function analyzeDependency(input: { crateName: string }) {
      // Search GitHub for alternatives
      const githubResults = await tools["githubSearchRepositories"]({
        queries: [{
          queryTerms: [input.crateName, "rust", "alternative"],
          language: "rust",
          stars: ">100",
          sort: "updated",
          limit: 10
        }]
      });

      // Search web for best practices
      const tavilyResults = await tools["tavily-search"]({
        query: input.crateName + " Rust crate best practices 2025",
        topic: "general",
        max_results: 5
      });

      return {
        alternatives: githubResults,
        bestPractices: tavilyResults
      };
    }
  `
})
```

### Available Development Tools

When working on rust-local-rag, Claude Code has access to these MCP tools:

**Tool Categories**:
- **File Operations**: Advanced file search, grep, directory trees, multi-file operations (mcp-filesystem)
- **Code Analysis**: AI Distiller for structure extraction, Smart Tree for project overviews
- **Web Research**: Tavily search and extraction, deep research capabilities
- **AI Consultants**: Gemini (gemini-3-pro-preview PRIMARY), Codex (OpenAI), Qwen (code-specialized)
- **GitHub Integration**: Code search, file content, repository structure exploration
- **Reasoning**: CRASH for structured, multi-step analysis

### Common Rust Development Workflows

#### 1. Understanding Rust Code Patterns
**When**: Learning how async/await, Arc<RwLock<T>>, or MCP macros work in this codebase

```typescript
// Use execute_code for quick pattern exploration
execute_code({
  code: `
    export default async function explorePattern(input: { pattern: string }) {
      // Search codebase for pattern
      const matches = await tools["grep_files"]({
        pattern: input.pattern,
        output_mode: "content",
        "-A": 5
      });

      // Get AI explanation
      const explanation = await tools["ask-gemini"]({
        prompt: "Explain this Rust pattern and its usage: " + input.pattern + "\\n\\nExamples:\\n" + matches,
        model: "gemini-3-pro-preview"
      });

      return { matches, explanation };
    }
  `,
  input: { pattern: "Arc<RwLock<" }
})
```

#### 2. Rust Documentation Lookup
**When**: Need to understand Rust async patterns, tokio APIs, or rmcp usage

```typescript
// Search for existing workflow first!
search_workflows({ tags: ["rust", "documentation"], limit: 5 })

// If not found, create ad-hoc workflow:
execute_code({
  code: `
    export default async function researchRustTopic(input: { topic: string }) {
      // Search web for documentation
      const docs = await tools["tavily-search"]({
        query: "Rust " + input.topic + " best practices 2025",
        topic: "general",
        max_results: 5
      });

      // Find GitHub examples
      const examples = await tools["githubSearchCode"]({
        queries: [{
          queryTerms: [input.topic],
          language: "rust",
          stars: ">50",
          limit: 10
        }]
      });

      return { documentation: docs, examples };
    }
  `,
  input: { topic: "Arc RwLock async patterns" }
})
```

#### 3. Debugging Cargo Errors
**When**: Compilation errors, dependency conflicts, or type mismatches

```typescript
// This is a common task - check if workflow exists first!
search_workflows({
  tags: ["rust", "debugging", "cargo-errors"],
  query: "analyze cargo compilation errors",
  limit: 5
})

// If not found, use execute_code:
execute_code({
  code: `
    export default async function debugCargoError(input: { errorText: string }) {
      // Get multi-model analysis
      const gemini = await tools["ask-gemini"]({
        prompt: "Explain this Rust cargo error and provide a fix: " + input.errorText,
        model: "gemini-3-pro-preview"
      });

      const codex = await tools["consult_codex"]({
        query: "Fix this Rust error: " + input.errorText,
        timeout: 300
      });

      return { gemini_analysis: gemini, codex_fix: codex };
    }
  `,
  input: { errorText: "error[E0308]: mismatched types..." }
})
```

#### 4. Code Review & Quality Check
**When**: Before commits, reviewing PRs, or refactoring

```bash
# Extract code structure for review
ai-distiller: distill_file(file_path="src/rag_engine.rs")
ai-distiller: distill_directory(directory_path="src/")

# Get multi-model review
crash: Review architecture decisions step-by-step
ask-gemini: "Review this Rust code for safety/performance issues: @src/embeddings.rs"
```

#### 5. Finding Rust Examples
**When**: Implementing new features or learning patterns

```bash
# Search GitHub for similar implementations
githubSearchCode(queries=[{
  queryTerms: ["Ollama", "embeddings", "Rust"],
  language: "rust",
  stars: ">50",
  limit: 10
}])

# Get example code with context
githubGetFileContent(queries=[{
  owner: "owner-name",
  repo: "repo-name",
  filePath: "src/example.rs",
  startLine: 100,
  endLine: 150
}])
```

#### 6. Test-Driven Development
**When**: Writing new tests or fixing failing tests

```bash
# Find existing test patterns
mcp-filesystem: grep_files(pattern="#\[tokio::test\]", output_mode="content", -A=10)

# Search for test examples
search_codebase("async test patterns with tokio", limit=5)

# Analyze test failures with AI
crash: Debug test failure with hypothesis testing
ask-gemini: "Why is this Rust test failing: <test output>"
```

#### 7. Dependency Research
**When**: Evaluating new crates or updating dependencies

```bash
# Research crate options
tavily-search(query="best Rust PDF parsing libraries 2025")
tavily-search(query="rmcp vs alternative MCP Rust libraries")

# Check GitHub activity
githubSearchRepositories(queries=[{
  queryTerms: ["rust", "pdf", "parser"],
  language: "rust",
  stars: ">100",
  sort: "updated"
}])
```

### Integration Examples

#### Example 1: Understanding rmcp Macros
```bash
# Step 1: Find macro usage in codebase
grep_files(pattern="#\[tool", output_mode="content", -A=3)

# Step 2: Search for documentation
tavily-search(query="rmcp Rust MCP SDK tool macro documentation")

# Step 3: Find real-world examples
githubSearchCode(queries=[{
  queryTerms: ["rmcp", "tool", "macro"],
  language: "rust"
}])
```

#### Example 2: Debugging Async Runtime Issues
```bash
# Step 1: Get structured analysis
crash: {
  step: "Analyze tokio runtime behavior",
  thought: "Understanding spawn vs spawn_blocking usage",
  confidence: 0.7
}

# Step 2: Multi-model validation
ask-gemini: "@src/main.rs Analyze async task spawning patterns" model="gemini-2.5-pro"
consult_codex: "Review tokio runtime usage in main.rs" timeout=300

# Step 3: Find similar patterns
githubSearchCode(queries=[{
  queryTerms: ["tokio", "spawn", "RwLock"],
  language: "rust"
}])
```

#### Example 3: Pre-commit Code Quality
```bash
# Extract structure for review
distill_directory(directory_path="src/", include_private=false)

# Get comprehensive review
crash: {
  purpose: "code_review",
  thought: "Systematic review of changes before commit"
}

# Validate with AI consultant
ask-gemini: "Review these changes for Rust best practices: <diff>" model="gemini-2.5-pro"
```

### Quick Reference for Claude Code

**Common Commands**:
```bash
# File operations
grep_files(pattern="TODO", output_mode="files_with_matches")
directory_tree(path="src/", max_depth=2)
search_files(pattern="test_*.rs", recursive=true)

# Code analysis
distill_file(file_path="src/rag_engine.rs")
project_overview(path=".")

# Web research
tavily-search(query="Rust async patterns")
githubSearchCode(queries=[{queryTerms: ["pattern"]}])

# AI consultants (ALWAYS use free models)
ask-gemini(prompt="question", model="gemini-2.5-pro")  # FREE
consult_codex(query="question", timeout=300)  # FREE via CLI
ask-qwen(prompt="question", model="qwen3-coder-plus")  # FREE

# Structured reasoning
crash({step_number: 1, purpose: "analysis", thought: "..."})
```

**Development Workflow Pattern**:
1. **Search** (grep/semantic) → Find relevant code
2. **Analyze** (distill/crash) → Understand structure
3. **Research** (tavily/github) → Find patterns/docs
4. **Validate** (gemini/codex) → Multi-model review
5. **Implement** → Write code
6. **Test** → Verify with cargo test

### Tool Selection Matrix

| Task Type | Primary Tool | Alternative | Use Case |
|-----------|-------------|-------------|----------|
| Find Rust function | `grep_files` | `search_codebase` | Exact identifier search |
| Understand pattern | `search_codebase` | `grep_files` | Conceptual code understanding |
| Lookup docs | `tavily-search` | `githubSearchCode` | External documentation |
| Debug error | `crash` | `ask-gemini` | Structured error analysis |
| Code review | `ask-gemini` | `consult_codex` | Multi-perspective validation |
| Find examples | `githubSearchCode` | `tavily-search` | Real-world code patterns |
| Project structure | `directory_tree` | `distill_directory` | Navigation vs analysis |

## Core Architecture

### Module Structure

- **main.rs**: Entry point, handles logging setup, environment configuration, initializes job system and worker supervisor
- **mcp_server.rs**: MCP protocol implementation using the `rmcp` crate, exposes MCP tools (search_documents, list_documents, get_stats, start_reindex, get_job_status, calibrate_reranker), plus health endpoints (/healthz liveness, /readyz readiness)
- **job_manager.rs**: SQLite-based job persistence with atomic transaction support for concurrent job creation, status tracking, and progress updates
- **worker.rs**: Background worker supervisor that processes reindexing jobs asynchronously, handles job resumption on restart, implements poison pill handling for document failures, and provides lock instrumentation via TimedWriteLockGuard for monitoring lock durations
- **rag_engine.rs**: Core RAG logic - sentence-aware chunking with metadata, embedding storage, similarity search, reranking orchestration, SHA-256 document fingerprinting, persistence, pure-Rust PDF extraction via lopdf with pdftotext fallback
- **embeddings.rs**: Ollama API client for generating embeddings with LRU caching (1000 entries) for query embeddings and batch embedding support
- **reranker.rs**: LLM-based relevance reranking service using Ollama with Phi-4-mini, performs concurrent second-stage scoring of search candidates using JSON-structured prompts with Phi chat template

### Key Design Patterns

1. **Async/Await Throughout**: Uses Tokio runtime for all I/O operations
2. **Arc<RwLock<RagEngine>>**: Shared state pattern for concurrent access to the RAG engine
3. **Job-Based Architecture**: Long-running document processing runs in background jobs, preventing MCP timeout issues
4. **Atomic Transaction Pattern**: SQLite transactions with WAL mode prevent race conditions in concurrent job creation
5. **Worker Supervisor Pattern**: Background task processes job queue and resumes interrupted jobs on restart
6. **Per-Document Locking**: Brief write locks per document (minutes) instead of hours-long locks during reindexing
7. **Poison Pill Handling**: Jobs continue processing remaining documents even if individual documents fail
8. **MCP Macros**: Uses `#[tool]` and `#[tool(tool_box)]` attributes from `rmcp-macros` for defining MCP tools
9. **Graceful Degradation**: Reranker initialization is non-fatal; falls back to embedding scores if unavailable
10. **LRU Caching**: Query embeddings cached with 1000-entry limit to reduce redundant API calls
11. **Batch Processing**: Multiple document chunks embedded in a single Ollama API call for efficiency
12. **Document Fingerprinting**: SHA-256 hashes track document changes to skip re-embedding unchanged files
13. **spawn_blocking for CPU-Bound Work**: All blocking/CPU-intensive operations (PDF extraction, embedding API calls) use `tokio::task::spawn_blocking` to avoid blocking the async runtime
14. **Lock Instrumentation**: `TimedWriteLockGuard<T>` wrapper measures lock hold duration and logs warnings when exceeding threshold (1000ms), with AtomicU64 metrics for testing
15. **Pure-Rust PDF Fallback**: Primary PDF extraction via `lopdf` crate (no external dependencies), with automatic fallback to `pdftotext` command if lopdf fails
16. **Health Probes**: Axum endpoints `/healthz` (liveness - always 200) and `/readyz` (readiness - 200 if engine lock can be acquired within 100ms)

### Data Flow

#### Document Indexing (Job-Based):
1. MCP client calls `start_reindex` tool → JobManager atomically creates job (or returns existing)
2. Job request sent via mpsc channel → WorkerSupervisor spawns background task
3. Worker discovers PDFs → `pdftotext` extraction → sentence-aware chunking with metadata
4. Text chunks → Ollama batch embeddings API → f32 vectors (per-document locking)
5. Embeddings + metadata → in-memory HashMap + disk persistence (model-specific `chunks_{model}.json`)
6. Document SHA-256 hash stored to detect changes on next index
7. Job status/progress updates persisted to SQLite → MCP client polls via `get_job_status`

#### Search Query Flow:
1. Query → LRU-cached embedding → cosine similarity search → top-k candidates
2. Candidates → concurrent LLM reranking (if available) → relevance-scored results
3. Results → MCP response → Claude Desktop

## Common Development Tasks

### Building and Running

```bash
# Development build and run (console logging)
make run                    # Sets DEV=true, logs to console
cargo run                   # Alternative

# Build only
make build                  # Debug build
make release                # Optimized release build

# Check and lint
make check                  # Cargo check
make lint                   # Clippy
make clippy                 # Clippy with warnings as errors
make fmt                    # Format code
make ci                     # Full CI pipeline (check + lint + test + build)
```

### Testing

```bash
make test                   # Run tests
cargo test                  # Alternative
```

### Evaluation Framework

The `eval/` directory contains a Python-based evaluation harness for measuring retrieval quality:

```bash
# Ensure RAG server is running first
make run

# Run baseline evaluation
python -m eval.run evaluate --config baseline -v
```

**Key Files:**
- `eval/configs/baseline.yaml` - Production config (embed-light + phi4-mini)
- `eval/ground_truth/queries.jsonl` - 50 labeled queries
- `eval/reports/BASELINE_EVALUATION_SUMMARY.md` - Latest results

**Latest Results (2025-12-08):**
- Document-Level Hit Rate@5: 77.8% (35/45)
- Latency p95: ~42s (reranking adds significant latency)
- True miss rate: ~2% (most "misses" are valid alternative sources)

See `docs/RAG_EVALUATION_FRAMEWORK_SPEC.md` for full specification.

### Installation

```bash
make install                # Install debug binary to ~/.cargo/bin
make install-release        # Install optimized binary
make install-production     # Build release + install
make uninstall              # Remove from system
make which-installed        # Check if installed and location
```

### Ollama Management

```bash
make setup-ollama           # Start Ollama + pull nomic-embed-text
make ollama-start           # Start Ollama server
make ollama-stop            # Stop Ollama server
make ollama-status          # Check status and list models
make ollama-models          # Pull required models
```

### Running Single Tests

```bash
# Run a specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run tests in a specific module
cargo test embeddings::
```

## Environment Configuration

Configuration is loaded from environment variables and `.env` files:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_DIR` | `./data` | Embeddings and index metadata |
| `DOCUMENTS_DIR` | `./documents` | PDF source directory |
| `LOG_DIR` | `/var/log/rust-local-rag` or `./logs` | Log output location |
| `LOG_LEVEL` | `info` | Tracing level (error/warn/info/debug/trace) |
| `LOG_MAX_MB` | `5` | Max log size before truncation |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model to use |
| `OLLAMA_RERANK_MODEL` | `llama3.1` | LLM model for reranking search results |
| `DEVELOPMENT` or `DEV` | unset | Forces console logging |
| `CONSOLE_LOGS` | unset | Console logging without dev mode |

## Important Implementation Details

### PDF Text Extraction

Uses a two-tier extraction approach:
1. **Primary**: Pure-Rust extraction via `lopdf` crate - no external dependencies required
2. **Fallback**: System `pdftotext` command (from poppler) via `Command::new("pdftotext")` if lopdf fails

This allows the server to work without poppler installed, while still supporting complex PDFs that lopdf may not handle. Both extraction methods run in `spawn_blocking` to avoid blocking the async runtime.

### Sentence-Aware Chunking

The chunking strategy has been enhanced to be sentence-aware with metadata tracking:
- Text is split into sentences using regex patterns
- Chunks are built from complete sentences (no mid-sentence splits)
- Metadata includes: page ranges, sentence ranges, section headings, token counts, overlap info
- Target chunk size is balanced with sentence boundaries for semantic coherence

### Document Change Detection

Uses SHA-256 hashing to detect document changes:
1. On document ingestion, compute SHA-256 hash of PDF bytes
2. Compare against stored hash in `document_hashes` HashMap
3. Skip re-embedding if hash matches (document unchanged)
4. Hashes persisted to disk with chunks for cross-session tracking

### Embedding Service Features

#### LRU Query Cache
- 1000-entry LRU cache for query embeddings via `query_cache: RwLock<LruCache<String, Vec<f32>>>`
- Separate method `get_query_embedding()` checks cache before calling Ollama
- Reduces API calls for repeated queries

#### Batch Embedding
- `embed_texts(&[String])` method supports batch processing
- Single API call embeds multiple chunks simultaneously
- Significantly faster than sequential embedding during document ingestion
- Handles both single and batch responses from Ollama API

### Embedding Model Validation

On startup, `EmbeddingService::new()` verifies:
1. Ollama is running and accessible
2. The specified embedding model exists in Ollama
3. Returns helpful error with available models if not found

Similarly, `RerankerService::new()` verifies the rerank model, but failure is non-fatal.

### Job-Based Document Processing

The server implements a robust job system to handle long-running document indexing operations:

#### Job Architecture:
- **JobManager**: SQLite-based persistence with atomic transaction support
  - `create_reindex_job_if_not_active()`: Atomic check-and-create prevents race conditions
  - WAL mode + 30s busy_timeout for concurrent access
  - `PRAGMA synchronous = NORMAL` for performance
- **WorkerSupervisor**: Background task that processes job queue via mpsc channel
  - Resumes interrupted jobs on server restart (`find_resumable_jobs()`)
  - Spawns async tasks for each job
  - Monitors for supervisor crashes
- **Per-Document Locking**: Worker acquires write lock briefly per document (minutes) instead of holding for hours
- **Poison Pill Handling**: Jobs continue even if individual documents fail, reporting failures in job error field

#### Job Lifecycle:
1. Client calls `start_reindex` → Atomic job creation or "already running" response
2. Job sent to supervisor → Background task spawned
3. Worker processes documents with progress updates
4. Job marked completed (or failed) with detailed status
5. Client polls `get_job_status` to monitor progress

#### Concurrency Behavior:
- Tested with 10 simultaneous job creation requests
- Only 1 job created successfully (atomic transaction works)
- Under extreme synthetic load, some requests may fail with SQLITE_BUSY
- Real-world MCP usage (1-2 concurrent requests) handles perfectly

### Automatic Reindexing

When `RagEngine::load_from_disk()` detects a model change:
1. Clears existing chunks from memory
2. Sets `needs_reindex = true` flag
3. User calls `start_reindex` tool to begin background reindexing
4. Persists new state with updated model name

### Two-Stage Retrieval with Reranking

The search process uses a two-stage approach:

#### Stage 1: Embedding-based Retrieval
- Query embedded using cached embedding service
- Cosine similarity search over all chunks
- Returns top-k initial candidates based on embedding similarity

#### Stage 2: LLM-based Reranking (optional)
- If `RerankerService` is available, candidates are reranked
- Concurrent scoring using `futures::join_all` for parallel processing
- LLM evaluates relevance with full context (query, document name, page, section, chunk text)
- Scores parsed from LLM response, normalized to [0.0, 1.0]
- Falls back to embedding score if reranking fails for any candidate
- Results sorted by final relevance score

**Graceful Degradation**: If reranker model is unavailable at startup, system continues with embedding-only search.

#### Reranker Prompt Architecture (Phi-4-Mini)

The reranker uses a specialized prompt format for Phi-family models. See `prompts/reranker.txt` for the full template.

**Key Implementation Details**:

1. **Phi Chat Template**: Uses `<|user|>...<|end|><|assistant|>` tokens required for Phi models to follow instructions (vs text completion mode)

2. **JSON Output Format**: Forces structured output via JSON syntax:
   ```json
   {"classification": "DIRECT_ANSWER", "reasoning": "...", "score": 95}
   ```

3. **Pre-fill Technique**: Prompt ends with `{"classification": "` to force the model into JSON completion mode, preventing early EOS prediction

4. **Stop Sequences**: Configured with `["<|end|>", "<|user|>"]` to prevent run-on generation

5. **Score Parsing**: `parse_score()` reconstructs full JSON by prepending the pre-fill, extracts the `score` field, normalizes 0-100 → 0.0-1.0

6. **Fallback Chain**: If JSON parsing fails, falls back to text-based score extraction (finds "score" followed by a number)

**Scoring Rubric**:
- 90-100: Directly answers with specific info
- 70-89: Partially answers
- 50-69: Related but doesn't answer
- 25-49: Same topic, not useful
- 0-24: Unrelated

**Penalties Applied**:
- Prefaces/intros: MAX 40
- Table of contents: MAX 30
- Keywords but no answer: MAX 65

**Debugging Reference**: See `docs/RERANKER_DEBUGGING_POSTMORTEM.md` for detailed documentation of the prompt engineering process and lessons learned.

### MCP Tool Definitions

Tools are defined using the `#[tool]` attribute:

```rust
#[tool(description = "Search through uploaded documents using semantic similarity")]
async fn search_documents(&self, #[tool(aggr)] SearchRequest { query, top_k }: SearchRequest) -> Result<CallToolResult, McpError>
```

The `#[tool(aggr)]` attribute indicates parameter aggregation for structured input.

### Persistence Format

**Model-Partitioned Storage**: Each embedding model gets its own index file:
- Pattern: `chunks_{sanitized_model_name}.json` (e.g., `chunks_nomic-embed-text.json`)
- Model names are sanitized: slashes become underscores, special chars removed
- Legacy `chunks.json` is auto-migrated if it matches the current model
- Switching models preserves other models' indexes (no data loss)

Index file structure:
```json
{
  "version": 2,
  "model": "nomic-embed-text",
  "chunks": {
    "uuid": {
      "id": "chunk-uuid",
      "document_name": "example.pdf",
      "text": "chunk content...",
      "embedding": [0.1, 0.2, ...],
      "chunk_index": 0,
      "page_number": 1,
      "section": "Introduction",
      "metadata": {
        "page_range": [1, 2],
        "sentence_range": [0, 5],
        "section_title": "Chapter 1",
        "token_count": 150,
        "overlap_with_previous": 20
      }
    }
  },
  "needs_reindex": false,
  "document_hashes": {
    "example.pdf": "sha256_hash_string"
  }
}
```

**Hot-Swap Behavior**: When switching embedding models (e.g., `nomic-embed-text` → `mxbai-embed-large`):
1. Current model's index is saved to `chunks_nomic-embed-text.json`
2. New model looks for `chunks_mxbai-embed-large.json`
3. If not found, starts fresh with `needs_reindex=true`
4. Switching back instantly restores the previous index

### Logging System

- **Production**: JSON logs to file, automatic truncation at `LOG_MAX_MB`
- **Development**: Pretty console output when `DEV=true` or `DEVELOPMENT=true`
- Background task runs every 5 minutes to check and truncate logs

## Integration with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rust-local-rag": {
      "command": "/Users/username/.cargo/bin/rust-local-rag",
      "env": {
        "DATA_DIR": "/Users/username/Documents/data",
        "DOCUMENTS_DIR": "/Users/username/Documents/rag",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
        "OLLAMA_RERANK_MODEL": "llama3.1"
      }
    }
  }
}
```

**Note**: If `OLLAMA_RERANK_MODEL` is not set or the model is unavailable, the system gracefully falls back to embedding-only search.

MCP communication uses stdin/stdout transport via `server.serve(stdio()).await`.

## Key Dependencies

- **rmcp** (0.8): Official Rust MCP SDK with streamable HTTP transport
- **tokio** (1.x): Async runtime with "full" features
- **sqlx** (0.8): Async SQLite database with compile-time query verification
- **chrono** (0.4): Date/time handling for job timestamps
- **reqwest** (0.12): HTTP client for Ollama API
- **serde/serde_json**: Serialization for persistence and API
- **tracing**: Structured logging
- **uuid**: Chunk ID and job ID generation
- **walkdir**: Directory traversal for PDF discovery
- **sha2** (0.10): SHA-256 hashing for document fingerprinting
- **lopdf** (0.34): Pure-Rust PDF parsing for text extraction (primary backend)
- **axum** (0.8): HTTP framework for health endpoints
- **lru** (transitive): LRU cache for query embeddings
- **regex** (transitive): Sentence splitting and text parsing
- **futures** (transitive): Concurrent reranking with `join_all`
