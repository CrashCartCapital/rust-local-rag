# Settings Menu UX Improvements - Mini PRD

**Document Version**: 2.0
**Date**: 2024-12-09
**Status**: Final - Pragmatism Reviewed
**Author**: Claude (with CRASH + Codex + Gemini ensemble analysis)

---

## Ensemble Review Summary

### Round 1: Feature Review (Codex + Gemini)
**Verdict**: Approved with interaction changes

- Changed `Enter` to open dropdown (standard combobox behavior)
- Typed channel payload for error handling
- Guard fetch with Loading state check

### Round 2: Pragmatism Review (Gemini + CRASH)
**Verdict**: Scope reduced 50% - cut overengineering

| Proposed Feature | Decision | Rationale |
|------------------|----------|-----------|
| Type-to-filter | **CUT** | YAGNI - most users have <20 models, scrolling is faster |
| Section grouping | **CUT** | Overkill for 8 settings, flat list is cleaner |
| Jump nav (g/G, Tab) | **CUT** | Arrow keys sufficient for 8 items |
| Test Connection | **CUT** | Redundant - successful fetch proves connection |
| std::thread + mpsc | **CHANGED** | Use tokio::spawn + oneshot (already async) |

### Round 3: Architecture Simplification (Codex)
**Verdict**: Use tokio::spawn + oneshot channel

Since the TUI already uses tokio::select! event loop, stay in async world:
- `tokio::spawn` for non-blocking fetch
- `tokio::sync::oneshot` for result
- `tokio::time::timeout(3s)` for timeout
- Add one branch to existing `select!`

### Final Scope

| Phase | Hours | Status |
|-------|-------|--------|
| 1. Model Discovery | 3.5h | Simplified with tokio |
| 2. Dropdown Component | 4h | No filtering |
| 3. ~~Settings Organization~~ | ~~3h~~ | **REMOVED** |
| 4. Navigation | 0.5h | Reset key only |
| 5. Validation | 2h | Basic only |
| **Total** | **10h** | Down from 20.5h |

### Edge Cases (Retained)

| Scenario | Required Behavior |
|----------|-------------------|
| Ollama unreachable | Show error message, allow manual text entry fallback |
| Empty model list | Display "No models found" in dropdown |
| Very long model names | Truncate with ellipsis |

---

## 1. Overview

### 1.1 Problem Statement

The current RAG-TUI settings menu has several UX friction points:

1. **Model Selection is Error-Prone**: Users must manually type Ollama model names exactly. No discovery, no validation, easy to make typos.
2. **No Model Discovery**: Users don't know what models are available on their Ollama instance without checking separately.
3. **Flat List Structure**: All 8 settings presented equally with no logical grouping.
4. **Limited Navigation**: Only j/k for single-item movement, no jump keys or section navigation.
5. **No Validation Feedback**: Invalid values (non-existent paths, malformed URLs) only fail at runtime.

### 1.2 Proposed Solution

Transform model selection fields into dynamic dropdowns populated from Ollama's `/api/tags` endpoint, group settings into logical sections, and add UX polish including inline validation and improved navigation.

### 1.3 Success Metrics

- Model selection errors reduced to zero (dropdown prevents invalid entries)
- Settings discovery improved (users can see all available models)
- Time to configure reduced (fewer keystrokes, no need to look up model names)

---

## 2. Requirements

### 2.1 Functional Requirements

#### FR-1: Dynamic Model Discovery
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1.1 | Query Ollama `/api/tags` endpoint on settings screen entry | P0 |
| FR-1.2 | Parse model list including name, size, and family metadata | P0 |
| FR-1.3 | Handle Ollama unavailable gracefully (show error, allow manual entry) | P0 |
| FR-1.4 | Cache model list for session (don't re-fetch on every settings open) | P1 |
| FR-1.5 | Provide manual refresh option (e.g., `R` key) | P2 |

#### FR-2: Dropdown/Select Component
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-2.1 | Render dropdown as modal overlay on top of settings | P0 |
| FR-2.2 | Show model name prominently, metadata (size, family) secondary | P0 |
| FR-2.3 | Pre-select current value when opening dropdown | P0 |
| FR-2.4 | Support Up/Down navigation within dropdown | P0 |
| FR-2.5 | Enter to confirm selection, Esc to cancel | P0 |
| FR-2.6 | Show "No models found" for empty list | P0 |
| FR-2.7 | Enter opens dropdown on select fields (standard combobox) | P0 |
| FR-2.8 | Truncate long model names with ellipsis | P1 |

#### ~~FR-3: Settings Organization~~ (REMOVED - Pragmatism Review)
*Cut: Overkill for 8 settings. Flat list is cleaner.*

#### FR-4: Validation & Feedback
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-4.1 | Validate URL format for Ollama URL field | P1 |
| FR-4.2 | Check path existence for directory fields | P1 |
| FR-4.3 | Show validation state inline (color coding) | P1 |

### 2.2 Non-Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-1 | Model fetch must not block UI (background thread) | P0 |
| NFR-2 | Dropdown should render smoothly (no flicker) | P0 |
| NFR-3 | Settings changes should persist to .env file (existing) | P0 |
| NFR-4 | Support terminals as small as 80x24 | P1 |

---

## 3. Design

### 3.1 Architecture Overview (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│                        App State                            │
├─────────────────────────────────────────────────────────────┤
│  settings: Settings                                         │
│  model_fetch_state: ModelFetchState                         │
│  model_fetch_rx: Option<oneshot::Receiver<ModelFetchResult>>│
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   Existing select! loop  │    │   tokio::spawn task      │
│                          │    │   (on settings entry)    │
│  - Terminal events       │    │                          │
│  - Timers                │    │  1. timeout(3s, fetch)   │
│  - model_fetch_rx.recv() │◄───│  2. GET /api/tags        │
│                          │    │  3. Send via oneshot     │
└──────────────────────────┘    └──────────────────────────┘
```

**Key Simplification**: Uses existing tokio async infrastructure instead of adding std::thread.

### 3.2 New Types (Simplified)

```rust
/// Model information from Ollama API
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    #[serde(default)]
    pub size_bytes: u64,
    pub family: Option<String>,
}

/// State of async model fetching
#[derive(Debug, Clone, Default)]
pub enum ModelFetchState {
    #[default]
    Idle,
    Loading,
    Loaded(Vec<OllamaModel>),
    Error(String),
}

/// Channel message type for model fetch results
pub type ModelFetchResult = Result<Vec<OllamaModel>, String>;

/// Dropdown widget state (simplified - no filtering)
#[derive(Debug, Clone)]
pub struct DropdownState {
    pub open: bool,
    pub items: Vec<OllamaModel>,
    pub selected: usize,
    pub list_state: ListState,
}

/// Which dropdown is currently active
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveDropdown {
    EmbeddingModel,
    RerankerModel,
    Theme,
}
```

### 3.3 App Mode Extension

```rust
pub enum AppMode {
    Normal,
    Detail,
    Help,
    Settings,
    SettingsDropdown(ActiveDropdown),  // NEW: Modal dropdown state
}
```

### 3.4 Settings Layout (Simplified - No Grouping)

```
┌─────────────────── Settings ───────────────────┐
│                                                │
│  ▶ Ollama URL       [http://localhost:11434]   │
│    Embedding Model  [nomic-embed-text      ▼]  │
│    Reranker Model   [phi4-mini             ▼]  │
│    Documents Dir    [./documents/           ]  │
│    Data Directory   [./data/                ]  │
│    Theme            [dark                  ▼]  │
│    Default Top-K    [10                     ]  │
│    Poll Interval    [2                      ]  │
│                                                │
├────────────────────────────────────────────────┤
│ URL of the Ollama server.                      │
├────────────────────────────────────────────────┤
│ Enter=Open/Edit  r=Reset  S=Save  Esc=Back     │
└────────────────────────────────────────────────┘
```

*Section headers removed per pragmatism review - flat list is cleaner for 8 items.*

### 3.5 Dropdown Overlay Design

```
┌─────────────── Select Embedding Model ─────────────────┐
│                                                        │
│  ▶ nomic-embed-text              274 MB   embedding    │
│    mxbai-embed-large             669 MB   embedding    │
│    all-minilm                    46 MB    sentence     │
│    phi4-mini                     2.2 GB   phi          │
│    llama3.2:latest               2.0 GB   llama        │
│    mistral:instruct              4.1 GB   mistral      │
│                                                        │
├────────────────────────────────────────────────────────┤
│ ↑/↓ Navigate  Enter=Select  Esc=Cancel                 │
└────────────────────────────────────────────────────────┘
```

### 3.6 Ollama API Integration

**Endpoint**: `GET http://localhost:11434/api/tags`

**Response Structure**:
```json
{
  "models": [
    {
      "name": "nomic-embed-text:latest",
      "modified_at": "2024-12-01T...",
      "size": 274000000,
      "digest": "abc123...",
      "details": {
        "format": "gguf",
        "family": "nomic-bert",
        "parameter_size": "137M",
        "quantization_level": "Q4_0"
      }
    }
  ]
}
```

**Mapping to OllamaModel**:
- `name` → `name`
- `size` → `size_bytes`
- `details.family` → `family`
- `details.parameter_size` → `parameter_size`

---

## 4. Implementation Tasks

### Phase 1: Ollama Model Discovery (P0)

| Task | Description | Estimate |
|------|-------------|----------|
| 1.1 | Create `ollama.rs` module with OllamaModel struct and fetch function | 0.5h |
| 1.2 | Add ModelFetchState enum to app.rs | 0.25h |
| 1.3 | Implement async `fetch_models()` with tokio::time::timeout(3s) | 1h |
| 1.4 | Add oneshot channel receiver to App struct | 0.25h |
| 1.5 | Spawn tokio task on `enter_settings_mode()` with Loading guard | 0.5h |
| 1.6 | Add branch to existing select! loop for model results | 0.5h |
| 1.7 | Handle error states (timeout, connection refused, parse error) | 0.25h |
| 1.8 | Add "Loading models..." indicator in UI | 0.25h |

**Phase 1 Total**: ~3.5 hours

### Phase 2: Dropdown Component (P0)

| Task | Description | Estimate |
|------|-------------|----------|
| 2.1 | Create DropdownState struct with ListState | 0.25h |
| 2.2 | Add ActiveDropdown enum and AppMode::SettingsDropdown | 0.25h |
| 2.3 | Implement `draw_dropdown_overlay()` in ui.rs | 1.5h |
| 2.4 | Handle dropdown keyboard events (Up/Down/Enter/Esc) | 0.75h |
| 2.5 | Wire Enter key to open dropdown (standard combobox) | 0.25h |
| 2.6 | Pre-select current value when opening | 0.25h |
| 2.7 | Update Setting.value on selection confirm | 0.25h |
| 2.8 | Format model metadata in dropdown items (name, size, family) | 0.25h |
| 2.9 | Handle empty model list ("No models found") | 0.15h |
| 2.10 | Truncate long model names with ellipsis | 0.1h |

**Phase 2 Total**: ~4 hours

### ~~Phase 3: Settings Organization~~ (REMOVED)
*Cut per pragmatism review - overkill for 8 settings.*

### Phase 4: Navigation (P2)

| Task | Description | Estimate |
|------|-------------|----------|
| 4.1 | Add `r` key for reset current field to default | 0.25h |
| 4.2 | Update keybindings footer | 0.25h |

**Phase 4 Total**: ~0.5 hours

### Phase 5: Validation (P2)

| Task | Description | Estimate |
|------|-------------|----------|
| 5.1 | Add URL validation for Ollama URL field | 0.5h |
| 5.2 | Add path existence check for directory fields | 0.5h |
| 5.3 | Render validation state (color coding) | 0.5h |
| 5.4 | Handle terminal resize during dropdown (graceful close) | 0.5h |

**Phase 5 Total**: ~2 hours

---

## Implementation Summary

| Phase | Scope | Hours | Priority |
|-------|-------|-------|----------|
| 1. Model Discovery | tokio fetch, oneshot channel, error handling | 3.5h | P0 |
| 2. Dropdown Component | Modal overlay, metadata display | 4h | P0 |
| ~~3. Settings Organization~~ | ~~Grouping, headers~~ | ~~3h~~ | **REMOVED** |
| 4. Navigation | Reset key only | 0.5h | P2 |
| 5. Validation | URL/path validation | 2h | P2 |
| **Total** | | **10h** | |

---

## 5. Dependencies

### 5.1 Rust Crates (Already Available)

- `reqwest` (with `blocking` feature) - HTTP client for Ollama API
- `serde`, `serde_json` - JSON parsing
- `ratatui` - TUI framework (List, ListState, Block)
- `crossterm` - Terminal events

### 5.2 Crates to Add

- `bytesize` (optional) - Human-readable file sizes (e.g., "2.2 GB")
  - Alternative: Simple helper function

---

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ollama not running | Dropdown empty | Graceful fallback to manual text entry with warning |
| Many models (50+) | Dropdown unwieldy | P2: Add filter/search within dropdown |
| Slow Ollama response | UI feels stuck | Show "Loading..." spinner, don't block UI |
| Model names change | Saved value invalid | Validate on settings load, warn if model not found |

---

## 7. Future Considerations

- **Model Categories**: Filter by embedding vs. LLM models based on capabilities
- **Model Info Panel**: Show detailed model info (context length, capabilities) on hover
- **Favorites/Recent**: Pin frequently used models to top of dropdown
- **Remote Ollama**: Support non-localhost Ollama instances (already have URL field)

---

## 8. Appendix

### A. Ollama API Response Example

```json
{
  "models": [
    {
      "name": "nomic-embed-text:latest",
      "model": "nomic-embed-text:latest",
      "modified_at": "2024-11-15T10:30:00Z",
      "size": 274458880,
      "digest": "0a109f422b47e3a30ba2b10eca18548e...",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "nomic-bert",
        "families": ["nomic-bert"],
        "parameter_size": "137M",
        "quantization_level": "F16"
      }
    },
    {
      "name": "phi4-mini:latest",
      "model": "phi4-mini:latest",
      "modified_at": "2024-12-01T14:20:00Z",
      "size": 2200000000,
      "digest": "b2c3d4e5f6...",
      "details": {
        "parent_model": "",
        "format": "gguf",
        "family": "phi",
        "families": ["phi"],
        "parameter_size": "3.8B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

### B. Key Binding Summary (Final - Simplified)

| Key | Context | Action |
|-----|---------|--------|
| `j` / `Down` | Settings list | Move to next setting |
| `k` / `Up` | Settings list | Move to previous setting |
| `Enter` | Text field | Enter edit mode |
| `Enter` | Dropdown field | Open dropdown overlay |
| `r` | Any field | Reset to default value |
| `S` | Settings | Save to .env |
| `Esc` | Settings | Exit (prompt if unsaved) |
| `Up/Down` | Dropdown open | Navigate options |
| `Enter` | Dropdown open | Confirm selection |
| `Esc` | Dropdown open | Cancel, keep previous |

*Removed per pragmatism review: g/G jump, Tab section nav, type-to-filter, Test Connection*

---

## C. Deferred Features (Cut for MVP)

Features removed during pragmatism review. Can be added later if user feedback indicates need:

| Feature | Original Est | Rationale for Cut |
|---------|--------------|-------------------|
| Type-to-filter in dropdown | 1.5h | YAGNI - most users have <20 models |
| Section grouping with headers | 3h | Overkill for 8 settings |
| Jump navigation (g/G, Tab) | 1h | Arrow keys sufficient |
| Test Connection button | 1h | Redundant - fetch success proves connection |
| std::thread concurrency | - | Simplified to tokio (already async) |

**When to reconsider**:
- Type-to-filter: If users report having 30+ models locally
- Section grouping: If settings list grows beyond 12 items
- Test Connection: If users frequently misconfigure Ollama URL

---

*End of Document*
