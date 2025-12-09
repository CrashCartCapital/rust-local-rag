use std::time::Instant;

use crate::api::{SearchResult, Stats};

/// App viewing mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppMode {
    #[default]
    Normal,
    Detail,
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
    pub status: String,

    // Models (from stats)
    pub embedding_model: String,
    pub reranker_model: Option<String>,

    // Job progress (if active)
    pub job_progress: Option<(u64, u64)>,

    // Config values (read-only display)
    pub config_summary: String,

    // Search
    pub query_input: String,
    pub search_in_progress: bool,
    pub search_id: u64,
    pub search_started: Option<Instant>,
    pub results: Vec<SearchResult>,
    pub selected_result: usize,
    pub top_k: usize,

    // UI Mode
    pub mode: AppMode,
    pub detail_scroll: usize,

    // Control
    pub should_quit: bool,
}

impl App {
    pub fn new(server_url: String) -> Self {
        // Build config summary from environment
        let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string());
        let docs_dir = std::env::var("DOCUMENTS_DIR").unwrap_or_else(|_| "./documents".to_string());
        let ollama_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| "localhost:11434".to_string());

        let config_summary = format!(
            "DATA_DIR={data_dir}  DOCS_DIR={docs_dir}  OLLAMA={ollama_url}"
        );

        // Get top_k from env or default to 10
        let top_k = std::env::var("RAG_TUI_TOP_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        Self {
            connected: false,
            server_url,
            last_error: None,
            doc_count: 0,
            chunk_count: 0,
            status: "connecting...".to_string(),
            embedding_model: "loading...".to_string(),
            reranker_model: None,
            job_progress: None,
            config_summary,
            query_input: String::new(),
            search_in_progress: false,
            search_id: 0,
            search_started: None,
            results: Vec::new(),
            selected_result: 0,
            top_k,
            mode: AppMode::Normal,
            detail_scroll: 0,
            should_quit: false,
        }
    }

    pub fn update_stats(&mut self, stats: Stats) {
        self.doc_count = stats.documents;
        self.chunk_count = stats.chunks;
        self.status = stats.status;

        if let Some(model) = stats.embedding_model {
            self.embedding_model = model;
        }
        self.reranker_model = stats.reranker_model;

        // Check for reindexing progress
        if self.status == "reindexing" {
            // We'd need job status endpoint for progress
            // For now, just show "reindexing"
        } else {
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
        self.results.clear();
        self.selected_result = 0;
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

    #[allow(dead_code)]
    pub fn toggle_mode(&mut self) {
        match self.mode {
            AppMode::Normal => self.enter_detail_mode(),
            AppMode::Detail => self.exit_detail_mode(),
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

    // Top-k adjustment
    pub fn increase_top_k(&mut self) {
        if self.top_k < 100 {
            self.top_k = (self.top_k + 5).min(100);
        }
    }

    pub fn decrease_top_k(&mut self) {
        if self.top_k > 1 {
            self.top_k = self.top_k.saturating_sub(5).max(1);
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
        assert_eq!(app.status, "ready");
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
}
