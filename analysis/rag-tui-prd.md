# RAG-TUI: Terminal User Interface for rust-local-rag

## Product Requirements Document (PRD)

**Version:** 2.0 (Simplified)
**Date:** 2025-12-08
**Status:** Validated via CRASH + Codex review

---

## 1. Overview

RAG-TUI is a minimal terminal interface for monitoring and querying the rust-local-rag server.

### User Goals (All 5 addressed in MVP)
1. **View config** - See current configuration values (read-only)
2. **View ingestion status** - Job progress during indexing
3. **View document/chunk counts** - System statistics
4. **View models** - Which embedding/reranker models are active
5. **Query interface** - Type queries, see results

### Non-Goals
- Config editing (Phase 2)
- Document management
- GUI/web interface

---

## 2. MVP Scope

### 2.1 Files (4 total)

| File | Purpose | Est. LOC |
|------|---------|----------|
| `main.rs` | Entry, terminal setup, event loop | 80-150 |
| `app.rs` | State, update logic | 120-220 |
| `ui.rs` | All rendering (single file) | 180-260 |
| `api.rs` | HTTP client | 70-120 |

**Total: 450-750 LOC**

### 2.2 Dependencies (6 total)

```toml
[dependencies]
ratatui = "0.29"
crossterm = { version = "0.28", features = ["event-stream"] }
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
dotenv = "0.15"
```

**Cut from original PRD:**
- ~~tui-textarea~~ (use basic String input)
- ~~throbber-widgets-tui~~ (use ASCII spinner)
- ~~clap~~ (use env vars only)
- ~~widgets/ subdirectory~~ (keep in ui.rs)

### 2.3 Features (8 total)

| # | Feature | Implementation |
|---|---------|----------------|
| 1 | Connection status | `●`/`○` indicator from `/healthz` |
| 2 | Error/status line | Display last error message |
| 3 | Stats display | Docs, chunks, status from `/stats` |
| 4 | Models display | Embedding + reranker model names |
| 5 | Config summary | Read-only view of key config values |
| 6 | Job progress | Progress bar when indexing active |
| 7 | Search input | Basic String with char/backspace |
| 8 | Results list | Scrollable list with scores |

### 2.4 Key Bindings (5 total)

| Key | Action |
|-----|--------|
| `Enter` | Submit search |
| `Esc` | Cancel current search |
| `↑/↓` | Scroll results |
| `r` | Force refresh stats |
| `q` | Quit |

---

## 3. Design

### 3.1 UI Layout

```
┌─ RAG-TUI ─────────────────────────────── [●] Connected ─┐
│ Docs: 15 │ Chunks: 1247 │ Status: ready                 │
│ Embed: nomic-embed-text │ Rerank: phi4-mini             │
├─────────────────────────────────────────────────────────┤
│ Progress: [████████░░░░░░░░░] 45% (9/20 docs)          │
├─────────────────────────────────────────────────────────┤
│ Config: DATA_DIR=./data  OLLAMA_URL=localhost:11434     │
├─────────────────────────────────────────────────────────┤
│ Query: [algorithmic trading____________]   Searching 5s │
│                                                         │
│ 1. [0.95] trading.pdf (p.12) "Position sizing..."      │
│ 2. [0.87] market.pdf (p.3) "VWAP algorithms..."        │
│ 3. [0.81] quant.pdf (p.78) "Risk management..."        │
├─────────────────────────────────────────────────────────┤
│ Error: (none)                                           │
├─────────────────────────────────────────────────────────┤
│ Enter=Search  Esc=Cancel  ↑↓=Scroll  r=Refresh  q=Quit │
└─────────────────────────────────────────────────────────┘
```

### 3.2 State Structure

```rust
pub struct App {
    // Connection
    connected: bool,
    server_url: String,
    last_error: Option<String>,

    // Stats (from /stats endpoint)
    doc_count: usize,
    chunk_count: usize,
    status: String,  // "ready" or "reindexing"

    // Models (from env/config)
    embedding_model: String,
    reranker_model: Option<String>,

    // Job progress (if active)
    job_progress: Option<(u64, u64)>,  // (current, total)

    // Config values (read-only display)
    config_summary: String,

    // Search
    query_input: String,
    search_in_progress: bool,
    search_id: u64,
    search_started: Option<Instant>,
    results: Vec<SearchResult>,
    selected_result: usize,

    // Control
    should_quit: bool,
}
```

### 3.3 Event Loop (Simplified)

```rust
async fn run(mut app: App, api: ApiClient) -> Result<()> {
    let mut events = EventStream::new();
    let mut poll_timer = interval(Duration::from_secs(2));

    loop {
        tokio::select! {
            // Terminal input
            Some(Ok(Event::Key(key))) = events.next() => {
                handle_key(&mut app, key, &api).await;
            }

            // Stats polling (fixed 2s interval)
            _ = poll_timer.tick() => {
                if let Ok(stats) = api.get_stats().await {
                    app.update_stats(stats);
                }
            }
        }

        draw(&mut terminal, &app)?;

        if app.should_quit { break; }
    }
    Ok(())
}
```

**Simplifications from original PRD:**
- No `Command` enum - direct async calls in handlers
- Fixed 2s polling - no adaptive intervals
- No `Message` enum - direct state mutation
- No `SearchJob` struct with abort_handle - just track search_id

### 3.4 Search Handling

Long searches (30-60s) are handled simply:

1. Set `search_in_progress = true`, increment `search_id`
2. Spawn async task to call `/search`
3. Show spinner + elapsed time in UI
4. When result arrives, check if `search_id` matches
5. If match: update results. If stale: ignore.
6. `Esc` key: set `search_in_progress = false` (logical cancel)

```rust
async fn handle_search(app: &mut App, api: &ApiClient) {
    let id = app.search_id;
    let query = app.query_input.clone();
    app.search_in_progress = true;
    app.search_started = Some(Instant::now());

    // Spawn search task
    let result = api.search(&query, 5).await;

    // Only apply if still current search
    if app.search_id == id && app.search_in_progress {
        match result {
            Ok(results) => app.results = results,
            Err(e) => app.last_error = Some(e.to_string()),
        }
        app.search_in_progress = false;
    }
}
```

---

## 4. API Endpoints

| Endpoint | Method | Response | Polling |
|----------|--------|----------|---------|
| `/healthz` | GET | 200 OK | 5s |
| `/stats` | GET | `{"documents": N, "chunks": N, "status": "..."}` | 2s |
| `/search` | POST | `{"results": [...]}` | On demand |

---

## 5. Implementation Tasks

| # | Task | Size |
|---|------|------|
| 1 | Project setup: Cargo.toml, .env.example | S |
| 2 | Terminal init: alternate screen, raw mode | S |
| 3 | App state struct | S |
| 4 | API client: health, stats, search | M |
| 5 | Main event loop with tokio::select! | M |
| 6 | UI layout and rendering | M |
| 7 | Stats + models display | S |
| 8 | Config summary display | S |
| 9 | Job progress bar | S |
| 10 | Search input handling | M |
| 11 | Results list with scrolling | M |
| 12 | Error display | S |
| 13 | Keybindings | S |

**Total: ~13 tasks, 450-750 LOC**

---

## 6. Deferred to Phase 2

- Config editing/reset
- Documents list view
- Search history
- Auto-reconnect with backoff
- Tab navigation between panels
- tui-textarea for richer input
- Trigger reindex from TUI

---

## 7. Config Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_TUI_SERVER_URL` | `http://localhost:3046` | Server URL |
| `RAG_TUI_POLL_INTERVAL_S` | `2` | Stats polling interval |
| `RAG_TUI_SEARCH_TIMEOUT_S` | `120` | Search timeout |
| `RAG_TUI_TOP_K` | `5` | Default results count |

---

## 8. Validation Summary

**CRASH Analysis:**
- Identified 8 overengineering items to cut
- Simplified to 4 files, 6 deps, 8 features
- Estimated 450-750 LOC

**Codex Review:**
- Approved all cuts except: config viewing (added back)
- Recommended: error status line (added)
- Confirmed 450-750 LOC is realistic
- Validated basic String input over tui-textarea for MVP

**Result:** Pragmatic MVP that covers all 5 user goals with minimal complexity.
