use std::time::{Duration, Instant};

use crate::api::{SearchResult, Stats};
use crate::config::Config;
use crate::constants::{DEFAULT_TOP_K, INPUT_DEBOUNCE_MS, MAX_TOP_K, MIN_TOP_K, TOP_K_STEP};
use crate::settings::Settings;
use crate::theme::Theme;

/// App viewing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppMode {
    #[default]
    Normal,
    Detail,
    Help,
    Settings,
}

/// Server status enum for type-safe status handling
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ServerStatus {
    #[default]
    Connecting,
    Ready,
    Reindexing,
    Unknown(String),
}

impl ServerStatus {
    /// Parse server status from API response string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "ready" => Self::Ready,
            "reindexing" => Self::Reindexing,
            "connecting" | "connecting..." => Self::Connecting,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// Check if server is in a ready state (can accept queries)
    #[allow(dead_code)]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Check if server is reindexing
    pub fn is_reindexing(&self) -> bool {
        matches!(self, Self::Reindexing)
    }
}

impl std::fmt::Display for ServerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connecting => write!(f, "connecting..."),
            Self::Ready => write!(f, "ready"),
            Self::Reindexing => write!(f, "reindexing"),
            Self::Unknown(s) => write!(f, "{}", s),
        }
    }
}

/// Message enum for Elm-style update pattern
/// Centralizes all state mutations for testability
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Msg {
    // Lifecycle
    Quit,

    // Navigation
    ScrollUp,
    ScrollDown,
    JumpToFirst,
    JumpToLast,
    EnterDetail,
    ExitDetail,
    ToggleHelp,

    // Input
    InputChar(char),
    InputBackspace,
    ClearQuery,

    // Top-K adjustment
    IncreaseTopK,
    DecreaseTopK,

    // Search
    StartSearch,
    SearchCompleted { search_id: u64, results: Vec<SearchResult> },
    SearchFailed(String),
    CancelSearch,

    // Connection
    SetConnected(bool),
    SetError(Option<String>),
    UpdateStats(Stats),

    // Reindex / Jobs
    ReindexStarted { job_id: String },
    JobProgress { progress: i64, total: i64 },
    JobCompleted,
    JobFailed(Option<String>),

    // Detail view
    DetailScrollUp,
    DetailScrollDown(usize),
}

#[allow(dead_code)]
pub struct App {
    // Connection
    pub connected: bool,
    pub server_url: String,
    pub last_error: Option<String>,

    // Stats (from /stats endpoint)
    pub doc_count: usize,
    pub chunk_count: usize,
    pub status: ServerStatus,

    // Models (from stats)
    pub embedding_model: String,
    pub reranker_model: Option<String>,

    // Job progress (if active)
    pub job_progress: Option<(u64, u64)>,
    pub active_job_id: Option<String>,
    pub reindex_in_progress: bool,

    // Config values (read-only display)
    pub config_summary: String,

    // Search
    pub query_input: String,
    pub last_searched_query: String, // Track what was actually searched (for Enter key logic)
    pub search_in_progress: bool,
    pub search_id: u64,
    pub search_started: Option<Instant>,
    pub results: Vec<SearchResult>,
    pub selected_result: usize,
    pub top_k: usize,

    // Input debouncing (for search-as-you-type)
    pub last_input_time: Option<Instant>,
    pub debounce_pending: bool,

    // UI Mode
    pub mode: AppMode,
    pub detail_scroll: usize,

    // Terminal size (for responsive layouts)
    pub terminal_size: (u16, u16),

    // Theming
    pub theme: Theme,

    // Settings
    pub settings: Settings,
    /// Message to show after settings save
    pub settings_message: Option<(String, bool)>, // (message, is_error)

    // Control
    pub should_quit: bool,
}

impl App {
    /// Create a new App with configuration from a Config struct
    pub fn new_with_config(config: &Config) -> Self {
        Self {
            connected: false,
            server_url: config.server_url.clone(),
            last_error: None,
            doc_count: 0,
            chunk_count: 0,
            status: ServerStatus::Connecting,
            embedding_model: "loading...".to_string(),
            reranker_model: None,
            job_progress: None,
            active_job_id: None,
            reindex_in_progress: false,
            config_summary: config.summary(),
            query_input: String::new(),
            last_searched_query: String::new(),
            search_in_progress: false,
            search_id: 0,
            search_started: None,
            results: Vec::new(),
            selected_result: 0,
            top_k: config.top_k,
            last_input_time: None,
            debounce_pending: false,
            mode: AppMode::Normal,
            detail_scroll: 0,
            terminal_size: (80, 24), // Default, updated on first resize event
            theme: Theme::from_name(&config.theme),
            settings: Settings::new(),
            settings_message: None,
            should_quit: false,
        }
    }

    /// Create a new App with server URL (backwards compatible)
    #[allow(dead_code)]
    pub fn new(server_url: String) -> Self {
        // Build config summary from environment
        let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string());
        let docs_dir = std::env::var("DOCUMENTS_DIR").unwrap_or_else(|_| "./documents".to_string());
        let ollama_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "localhost:11434".to_string());

        let config_summary = format!(
            "DATA_DIR={data_dir}  DOCS_DIR={docs_dir}  OLLAMA={ollama_url}"
        );

        // Get top_k from env or default
        let top_k = std::env::var("RAG_TUI_TOP_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TOP_K);

        Self {
            connected: false,
            server_url,
            last_error: None,
            doc_count: 0,
            chunk_count: 0,
            status: ServerStatus::Connecting,
            embedding_model: "loading...".to_string(),
            reranker_model: None,
            job_progress: None,
            active_job_id: None,
            reindex_in_progress: false,
            config_summary,
            query_input: String::new(),
            last_searched_query: String::new(),
            search_in_progress: false,
            search_id: 0,
            search_started: None,
            results: Vec::new(),
            selected_result: 0,
            top_k,
            last_input_time: None,
            debounce_pending: false,
            mode: AppMode::Normal,
            detail_scroll: 0,
            terminal_size: (80, 24), // Default, updated on first resize event
            theme: Theme::default(),
            settings: Settings::new(),
            settings_message: None,
            should_quit: false,
        }
    }

    /// Update terminal size on resize
    pub fn set_terminal_size(&mut self, width: u16, height: u16) {
        self.terminal_size = (width, height);
    }

    /// Check if terminal is "compact" (for adaptive layouts)
    #[allow(dead_code)]
    pub fn is_compact_mode(&self) -> bool {
        self.terminal_size.0 < 100 || self.terminal_size.1 < 30
    }

    /// Record input event for debouncing (call on each keystroke)
    #[allow(dead_code)]
    pub fn record_input(&mut self) {
        self.last_input_time = Some(Instant::now());
        self.debounce_pending = true;
    }

    /// Check if debounce period has elapsed since last input
    /// Returns true if enough time has passed and there was pending input
    #[allow(dead_code)]
    pub fn should_trigger_debounced_search(&mut self) -> bool {
        if !self.debounce_pending {
            return false;
        }

        if let Some(last_time) = self.last_input_time {
            if last_time.elapsed() >= Duration::from_millis(INPUT_DEBOUNCE_MS) {
                self.debounce_pending = false;
                return true;
            }
        }
        false
    }

    /// Clear debounce state (e.g., on explicit search submission)
    #[allow(dead_code)]
    pub fn clear_debounce(&mut self) {
        self.debounce_pending = false;
        self.last_input_time = None;
    }

    /// Set active job from API response
    pub fn set_active_job(&mut self, job_id: String, progress: i64, total: i64) {
        self.active_job_id = Some(job_id);
        self.reindex_in_progress = true;
        self.job_progress = Some((progress as u64, total as u64));
    }

    /// Clear active job (completed or failed)
    pub fn clear_active_job(&mut self) {
        self.active_job_id = None;
        self.reindex_in_progress = false;
        self.job_progress = None;
    }

    /// Update job progress
    pub fn update_job_progress(&mut self, progress: i64, total: i64) {
        self.job_progress = Some((progress as u64, total as u64));
    }

    pub fn update_stats(&mut self, stats: Stats) {
        self.doc_count = stats.documents;
        self.chunk_count = stats.chunks;
        self.status = ServerStatus::from_str(&stats.status);

        if let Some(model) = stats.embedding_model {
            self.embedding_model = model;
        }
        self.reranker_model = stats.reranker_model;

        // Check for reindexing progress
        if !self.status.is_reindexing() {
            // Clear job progress if not reindexing
            self.job_progress = None;
        }
    }

    pub fn set_connected(&mut self, connected: bool) {
        let was_disconnected = !self.connected;
        self.connected = connected;
        if connected && was_disconnected {
            // Clear error on reconnection
            self.last_error = None;
        } else if !connected {
            self.last_error = Some("Connection lost".to_string());
        }
    }

    pub fn set_error(&mut self, error: Option<String>) {
        self.last_error = error;
    }

    pub fn start_search(&mut self) {
        if self.query_input.trim().is_empty() {
            return;
        }
        self.search_id += 1;
        self.search_in_progress = true;
        self.search_started = Some(Instant::now());
        self.last_searched_query = self.query_input.clone(); // Track what we searched
        self.results.clear();
        self.selected_result = 0;
    }

    /// Check if query has been modified since last search
    /// Used to determine Enter key behavior (search vs open detail)
    pub fn query_changed_since_search(&self) -> bool {
        self.query_input != self.last_searched_query
    }

    pub fn complete_search(&mut self, search_id: u64, results: Vec<SearchResult>) {
        if self.search_id == search_id && self.search_in_progress {
            self.results = results;
            self.search_in_progress = false;
            self.search_started = None;
            self.selected_result = 0;
        }
    }

    pub fn cancel_search(&mut self) {
        self.search_in_progress = false;
        self.search_started = None;
    }

    pub fn scroll_up(&mut self) {
        if self.selected_result > 0 {
            self.selected_result -= 1;
        }
    }

    pub fn scroll_down(&mut self) {
        if !self.results.is_empty() && self.selected_result < self.results.len() - 1 {
            self.selected_result += 1;
        }
    }

    /// Scroll up by a page (10 items)
    pub fn scroll_page_up(&mut self) {
        self.selected_result = self.selected_result.saturating_sub(10);
    }

    /// Scroll down by a page (10 items)
    pub fn scroll_page_down(&mut self) {
        if !self.results.is_empty() {
            self.selected_result = (self.selected_result + 10).min(self.results.len() - 1);
        }
    }

    pub fn input_char(&mut self, c: char) {
        self.query_input.push(c);
    }

    pub fn input_backspace(&mut self) {
        self.query_input.pop();
    }

    pub fn clear_query(&mut self) {
        self.query_input.clear();
    }

    pub fn search_elapsed_secs(&self) -> Option<u64> {
        self.search_started.map(|start| start.elapsed().as_secs())
    }

    // Mode switching
    pub fn enter_detail_mode(&mut self) {
        if !self.results.is_empty() {
            self.mode = AppMode::Detail;
            self.detail_scroll = 0;
        }
    }

    pub fn exit_detail_mode(&mut self) {
        self.mode = AppMode::Normal;
        self.detail_scroll = 0;
    }

    // Help mode
    pub fn toggle_help(&mut self) {
        match self.mode {
            AppMode::Help => self.mode = AppMode::Normal,
            _ => self.mode = AppMode::Help,
        }
    }

    pub fn exit_help_mode(&mut self) {
        if self.mode == AppMode::Help {
            self.mode = AppMode::Normal;
        }
    }

    // Settings mode
    pub fn enter_settings_mode(&mut self) {
        self.mode = AppMode::Settings;
        self.settings_message = None;
    }

    pub fn exit_settings_mode(&mut self) {
        if self.mode == AppMode::Settings {
            // Cancel any in-progress edit
            self.settings.cancel_edit();
            self.mode = AppMode::Normal;
        }
    }

    /// Save settings and update theme if needed
    pub fn save_settings(&mut self) {
        match self.settings.save_to_env() {
            Ok(had_restart_required) => {
                // save_to_env returns whether restart-required settings were modified
                // (checked BEFORE marking as saved, so this is accurate)
                let msg = if had_restart_required {
                    "Settings saved! Restart server for changes to take effect."
                } else {
                    "Settings saved!"
                };
                self.settings_message = Some((msg.to_string(), false));

                // Apply TUI-only settings immediately
                if let Some(theme_setting) = self.settings.items.iter()
                    .find(|s| s.env_var == "RAG_TUI_THEME")
                {
                    self.theme = Theme::from_name(&theme_setting.value);
                }
            }
            Err(e) => {
                self.settings_message = Some((format!("Save failed: {e}"), true));
            }
        }
    }

    #[allow(dead_code)]
    pub fn toggle_mode(&mut self) {
        match self.mode {
            AppMode::Normal => self.enter_detail_mode(),
            AppMode::Detail => self.exit_detail_mode(),
            AppMode::Help => self.exit_help_mode(),
            AppMode::Settings => self.exit_settings_mode(),
        }
    }

    // Detail scroll (for full text view)
    pub fn detail_scroll_up(&mut self) {
        if self.detail_scroll > 0 {
            self.detail_scroll -= 1;
        }
    }

    pub fn detail_scroll_down(&mut self, max_lines: usize) {
        if self.detail_scroll < max_lines.saturating_sub(1) {
            self.detail_scroll += 1;
        }
    }

    /// Scroll detail view up by a page (10 lines)
    pub fn detail_scroll_page_up(&mut self) {
        self.detail_scroll = self.detail_scroll.saturating_sub(10);
    }

    /// Scroll detail view down by a page (10 lines)
    pub fn detail_scroll_page_down(&mut self, max_lines: usize) {
        self.detail_scroll = (self.detail_scroll + 10).min(max_lines.saturating_sub(1));
    }

    // Top-k adjustment
    pub fn increase_top_k(&mut self) {
        if self.top_k < MAX_TOP_K {
            self.top_k = (self.top_k + TOP_K_STEP).min(MAX_TOP_K);
        }
    }

    pub fn decrease_top_k(&mut self) {
        if self.top_k > MIN_TOP_K {
            self.top_k = self.top_k.saturating_sub(TOP_K_STEP).max(MIN_TOP_K);
        }
    }

    // Jump navigation
    pub fn jump_to_first(&mut self) {
        self.selected_result = 0;
    }

    pub fn jump_to_last(&mut self) {
        if !self.results.is_empty() {
            self.selected_result = self.results.len() - 1;
        }
    }

    // Get selected result for detail view
    pub fn selected_result_ref(&self) -> Option<&SearchResult> {
        self.results.get(self.selected_result)
    }

    /// Elm-style update function - centralizes all state mutations
    /// Returns true if a search should be triggered (for async handling)
    #[allow(dead_code)]
    pub fn update(&mut self, msg: Msg) -> bool {
        match msg {
            // Lifecycle
            Msg::Quit => {
                self.should_quit = true;
            }

            // Navigation
            Msg::ScrollUp => self.scroll_up(),
            Msg::ScrollDown => self.scroll_down(),
            Msg::JumpToFirst => self.jump_to_first(),
            Msg::JumpToLast => self.jump_to_last(),
            Msg::EnterDetail => self.enter_detail_mode(),
            Msg::ExitDetail => self.exit_detail_mode(),
            Msg::ToggleHelp => self.toggle_help(),

            // Input
            Msg::InputChar(c) => self.input_char(c),
            Msg::InputBackspace => self.input_backspace(),
            Msg::ClearQuery => self.clear_query(),

            // Top-K
            Msg::IncreaseTopK => self.increase_top_k(),
            Msg::DecreaseTopK => self.decrease_top_k(),

            // Search
            Msg::StartSearch => {
                self.start_search();
                return true; // Signal to trigger async search
            }
            Msg::SearchCompleted { search_id, results } => {
                self.complete_search(search_id, results);
            }
            Msg::SearchFailed(err) => {
                self.cancel_search();
                self.set_error(Some(format!("Search failed: {err}")));
            }
            Msg::CancelSearch => self.cancel_search(),

            // Connection
            Msg::SetConnected(connected) => self.set_connected(connected),
            Msg::SetError(err) => self.set_error(err),
            Msg::UpdateStats(stats) => self.update_stats(stats),

            // Reindex / Jobs
            Msg::ReindexStarted { job_id } => {
                self.set_active_job(job_id, 0, 0);
            }
            Msg::JobProgress { progress, total } => {
                self.update_job_progress(progress, total);
            }
            Msg::JobCompleted => {
                self.clear_active_job();
            }
            Msg::JobFailed(err) => {
                self.clear_active_job();
                self.set_error(err.or(Some("Reindex failed".to_string())));
            }

            // Detail view
            Msg::DetailScrollUp => self.detail_scroll_up(),
            Msg::DetailScrollDown(max) => self.detail_scroll_down(max),
        }
        false // No async action needed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_result(doc: &str, score: f32) -> SearchResult {
        SearchResult {
            document: doc.to_string(),
            text: "Test content".to_string(),
            score,
            page_number: 1,
            chunk_id: "test-chunk".to_string(),
            section: None,
        }
    }

    #[test]
    fn test_app_initial_state() {
        let app = App::new("http://localhost:3046".to_string());
        assert!(!app.connected);
        assert!(!app.search_in_progress);
        assert_eq!(app.search_id, 0);
        assert!(app.results.is_empty());
        assert!(app.query_input.is_empty());
        assert!(!app.should_quit);
    }

    #[test]
    fn test_input_handling() {
        let mut app = App::new("http://localhost:3046".to_string());

        // Type characters
        app.input_char('h');
        app.input_char('e');
        app.input_char('l');
        app.input_char('l');
        app.input_char('o');
        assert_eq!(app.query_input, "hello");

        // Backspace
        app.input_backspace();
        assert_eq!(app.query_input, "hell");

        // Backspace on empty should not panic
        let mut empty_app = App::new("http://localhost:3046".to_string());
        empty_app.input_backspace(); // Should not panic
        assert!(empty_app.query_input.is_empty());
    }

    #[test]
    fn test_search_id_tracking() {
        let mut app = App::new("http://localhost:3046".to_string());
        app.query_input = "test query".to_string();

        // Start first search
        app.start_search();
        assert_eq!(app.search_id, 1);
        assert!(app.search_in_progress);
        let first_id = app.search_id;

        // Start second search (simulating user typing new query)
        app.query_input = "new query".to_string();
        app.search_in_progress = false; // Reset for test
        app.start_search();
        assert_eq!(app.search_id, 2);
        let second_id = app.search_id;

        // Complete with STALE search_id - should be ignored
        let stale_results = vec![make_test_result("stale.pdf", 0.5)];
        app.complete_search(first_id, stale_results);
        assert!(app.results.is_empty()); // Stale results ignored

        // Complete with CURRENT search_id - should update
        let current_results = vec![make_test_result("current.pdf", 0.9)];
        app.complete_search(second_id, current_results);
        assert_eq!(app.results.len(), 1);
        assert_eq!(app.results[0].document, "current.pdf");
        assert!(!app.search_in_progress);
    }

    #[test]
    fn test_result_navigation() {
        let mut app = App::new("http://localhost:3046".to_string());

        // Add some results
        app.results = vec![
            make_test_result("doc1.pdf", 0.9),
            make_test_result("doc2.pdf", 0.8),
            make_test_result("doc3.pdf", 0.7),
        ];
        app.selected_result = 0;

        // Scroll down
        app.scroll_down();
        assert_eq!(app.selected_result, 1);
        app.scroll_down();
        assert_eq!(app.selected_result, 2);

        // Can't scroll past end
        app.scroll_down();
        assert_eq!(app.selected_result, 2);

        // Scroll up
        app.scroll_up();
        assert_eq!(app.selected_result, 1);
        app.scroll_up();
        assert_eq!(app.selected_result, 0);

        // Can't scroll past beginning
        app.scroll_up();
        assert_eq!(app.selected_result, 0);
    }

    #[test]
    fn test_scroll_empty_results() {
        let mut app = App::new("http://localhost:3046".to_string());
        // Empty results - scrolling should not panic
        app.scroll_down();
        app.scroll_up();
        assert_eq!(app.selected_result, 0);
    }

    #[test]
    fn test_search_cancel() {
        let mut app = App::new("http://localhost:3046".to_string());
        app.query_input = "test".to_string();
        app.start_search();
        assert!(app.search_in_progress);

        app.cancel_search();
        assert!(!app.search_in_progress);
        assert!(app.search_started.is_none());
    }

    #[test]
    fn test_empty_query_start_search() {
        let mut app = App::new("http://localhost:3046".to_string());
        app.query_input = "   ".to_string(); // Whitespace only
        app.start_search();
        assert_eq!(app.search_id, 0); // Should not increment
        assert!(!app.search_in_progress);
    }

    #[test]
    fn test_stats_update() {
        let mut app = App::new("http://localhost:3046".to_string());

        let stats = Stats {
            documents: 15,
            chunks: 1247,
            status: "ready".to_string(),
            embedding_model: Some("nomic-embed-text".to_string()),
            reranker_model: Some("phi4-mini".to_string()),
        };

        app.update_stats(stats);
        assert_eq!(app.doc_count, 15);
        assert_eq!(app.chunk_count, 1247);
        assert_eq!(app.status, ServerStatus::Ready);
        assert_eq!(app.embedding_model, "nomic-embed-text");
        assert_eq!(app.reranker_model, Some("phi4-mini".to_string()));
    }

    #[test]
    fn test_connection_state() {
        let mut app = App::new("http://localhost:3046".to_string());

        // Initial connection
        app.set_connected(true);
        assert!(app.connected);
        assert!(app.last_error.is_none()); // Error cleared on reconnection

        // Disconnect
        app.set_connected(false);
        assert!(!app.connected);
        assert!(app.last_error.is_some()); // Should set error on disconnect

        // Reconnect - should clear error
        app.set_connected(true);
        assert!(app.connected);
        assert!(app.last_error.is_none()); // Error cleared on reconnection
    }

    #[test]
    fn test_error_handling() {
        let mut app = App::new("http://localhost:3046".to_string());

        app.set_error(Some("Test error".to_string()));
        assert_eq!(app.last_error, Some("Test error".to_string()));

        app.set_error(None);
        assert!(app.last_error.is_none());
    }

    #[test]
    fn test_unicode_input() {
        let mut app = App::new("http://localhost:3046".to_string());

        // Unicode characters
        app.input_char('日');
        app.input_char('本');
        app.input_char('語');
        assert_eq!(app.query_input, "日本語");

        app.input_backspace();
        assert_eq!(app.query_input, "日本");
    }

    // Tests for Msg update pattern
    #[test]
    fn test_msg_quit() {
        let mut app = App::new("http://localhost:3046".to_string());
        assert!(!app.should_quit);
        app.update(Msg::Quit);
        assert!(app.should_quit);
    }

    #[test]
    fn test_msg_navigation() {
        let mut app = App::new("http://localhost:3046".to_string());
        app.results = vec![
            make_test_result("doc1.pdf", 0.9),
            make_test_result("doc2.pdf", 0.8),
        ];

        app.update(Msg::ScrollDown);
        assert_eq!(app.selected_result, 1);

        app.update(Msg::ScrollUp);
        assert_eq!(app.selected_result, 0);

        app.update(Msg::JumpToLast);
        assert_eq!(app.selected_result, 1);

        app.update(Msg::JumpToFirst);
        assert_eq!(app.selected_result, 0);
    }

    #[test]
    fn test_msg_input() {
        let mut app = App::new("http://localhost:3046".to_string());

        app.update(Msg::InputChar('h'));
        app.update(Msg::InputChar('i'));
        assert_eq!(app.query_input, "hi");

        app.update(Msg::InputBackspace);
        assert_eq!(app.query_input, "h");

        app.update(Msg::ClearQuery);
        assert!(app.query_input.is_empty());
    }

    #[test]
    fn test_msg_search_flow() {
        let mut app = App::new("http://localhost:3046".to_string());
        app.query_input = "test query".to_string();

        // StartSearch returns true to signal async action needed
        let needs_async = app.update(Msg::StartSearch);
        assert!(needs_async);
        assert!(app.search_in_progress);

        // Complete search
        let results = vec![make_test_result("result.pdf", 0.85)];
        app.update(Msg::SearchCompleted { search_id: app.search_id, results });
        assert!(!app.search_in_progress);
        assert_eq!(app.results.len(), 1);
    }

    #[test]
    fn test_msg_mode_toggle() {
        let mut app = App::new("http://localhost:3046".to_string());
        assert_eq!(app.mode, AppMode::Normal);

        app.update(Msg::ToggleHelp);
        assert_eq!(app.mode, AppMode::Help);

        app.update(Msg::ToggleHelp);
        assert_eq!(app.mode, AppMode::Normal);

        app.results = vec![make_test_result("doc.pdf", 0.9)];
        app.update(Msg::EnterDetail);
        assert_eq!(app.mode, AppMode::Detail);

        app.update(Msg::ExitDetail);
        assert_eq!(app.mode, AppMode::Normal);
    }

    #[test]
    fn test_msg_job_lifecycle() {
        let mut app = App::new("http://localhost:3046".to_string());
        assert!(!app.reindex_in_progress);

        app.update(Msg::ReindexStarted { job_id: "job-123".to_string() });
        assert!(app.reindex_in_progress);
        assert_eq!(app.active_job_id, Some("job-123".to_string()));

        app.update(Msg::JobProgress { progress: 5, total: 10 });
        assert_eq!(app.job_progress, Some((5, 10)));

        app.update(Msg::JobCompleted);
        assert!(!app.reindex_in_progress);
        assert!(app.active_job_id.is_none());
    }

    // ServerStatus tests
    #[test]
    fn test_server_status_from_str() {
        assert_eq!(ServerStatus::from_str("ready"), ServerStatus::Ready);
        assert_eq!(ServerStatus::from_str("Ready"), ServerStatus::Ready);
        assert_eq!(ServerStatus::from_str("READY"), ServerStatus::Ready);

        assert_eq!(ServerStatus::from_str("reindexing"), ServerStatus::Reindexing);
        assert_eq!(ServerStatus::from_str("Reindexing"), ServerStatus::Reindexing);

        assert_eq!(ServerStatus::from_str("connecting..."), ServerStatus::Connecting);
        assert_eq!(ServerStatus::from_str("connecting"), ServerStatus::Connecting);

        assert_eq!(
            ServerStatus::from_str("custom"),
            ServerStatus::Unknown("custom".to_string())
        );
    }

    #[test]
    fn test_server_status_display() {
        assert_eq!(ServerStatus::Ready.to_string(), "ready");
        assert_eq!(ServerStatus::Reindexing.to_string(), "reindexing");
        assert_eq!(ServerStatus::Connecting.to_string(), "connecting...");
        assert_eq!(
            ServerStatus::Unknown("custom".to_string()).to_string(),
            "custom"
        );
    }

    #[test]
    fn test_server_status_predicates() {
        assert!(ServerStatus::Ready.is_ready());
        assert!(!ServerStatus::Reindexing.is_ready());
        assert!(!ServerStatus::Connecting.is_ready());

        assert!(ServerStatus::Reindexing.is_reindexing());
        assert!(!ServerStatus::Ready.is_reindexing());
        assert!(!ServerStatus::Connecting.is_reindexing());
    }

    #[test]
    fn test_server_status_default() {
        assert_eq!(ServerStatus::default(), ServerStatus::Connecting);
    }
}
