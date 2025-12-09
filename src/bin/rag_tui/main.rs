mod api;
mod app;
mod ui;

use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;
use tokio::time::{interval, timeout};

use api::{ApiClient, SearchResult};
use app::{App, AppMode};

/// RAII guard for terminal cleanup - ensures terminal is restored even on panic
struct TuiGuard {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TuiGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }
}

impl Drop for TuiGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        );
        let _ = self.terminal.show_cursor();
    }
}

/// Search result message for async channel
type SearchOutcome = Result<(u64, Vec<SearchResult>), String>;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present
    dotenv::dotenv().ok();

    // Get server URL from environment
    let server_url = std::env::var("RAG_TUI_SERVER_URL")
        .unwrap_or_else(|_| "http://localhost:3046".to_string());

    // Setup terminal with RAII guard (cleanup happens automatically on drop)
    let mut tui = TuiGuard::new()?;

    // Create app and API client
    let app = App::new(server_url.clone());
    let api = ApiClient::new(server_url);

    // Run the app (any panic will still trigger TuiGuard::drop)
    let result = run_app(&mut tui.terminal, app, api).await;

    if let Err(e) = &result {
        eprintln!("Error: {e}");
    }

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: App,
    api: ApiClient,
) -> Result<()> {
    // Polling interval from env or default 2s
    let poll_interval_secs = std::env::var("RAG_TUI_POLL_INTERVAL_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let mut poll_timer = interval(Duration::from_secs(poll_interval_secs));
    let mut health_timer = interval(Duration::from_secs(5));

    // Channel for async search results (non-blocking search)
    let (search_tx, mut search_rx) = mpsc::unbounded_channel::<SearchOutcome>();

    // Short timeout for health/stats calls to keep UI responsive (5 seconds)
    const API_TIMEOUT: Duration = Duration::from_secs(5);

    // Initial health check and stats fetch (with timeout)
    let health_result = timeout(API_TIMEOUT, api.health_check()).await;
    app.set_connected(health_result.ok().and_then(|r| r.ok()).unwrap_or(false));
    if app.connected {
        if let Ok(Ok(stats)) = timeout(API_TIMEOUT, api.get_stats()).await {
            app.update_stats(stats);
        }
    }

    loop {
        // Draw UI
        terminal.draw(|f| ui::draw(f, &app))?;

        // Handle events with timeout to allow polling
        tokio::select! {
            // Terminal input events
            result = tokio::task::spawn_blocking(|| {
                if event::poll(Duration::from_millis(100)).unwrap_or(false) {
                    event::read().ok()
                } else {
                    None
                }
            }) => {
                if let Ok(Some(Event::Key(key))) = result {
                    // Mode-specific keybindings
                    match app.mode {
                        AppMode::Normal => {
                            match (key.code, key.modifiers) {
                                // Quit
                                (KeyCode::Char('q'), KeyModifiers::NONE) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                    app.should_quit = true;
                                }

                                // Enter: open detail view OR submit search
                                (KeyCode::Enter, _) => {
                                    if !app.results.is_empty() {
                                        // Open detail view for selected result
                                        app.enter_detail_mode();
                                    } else if !app.search_in_progress && !app.query_input.is_empty() {
                                        // Submit search
                                        app.start_search();
                                        let search_id = app.search_id;
                                        let query = app.query_input.clone();
                                        let top_k = app.top_k;
                                        let api_clone = api.clone();
                                        let tx = search_tx.clone();

                                        tokio::spawn(async move {
                                            let outcome = match api_clone.search(&query, top_k).await {
                                                Ok(results) => Ok((search_id, results)),
                                                Err(e) => Err(e.to_string()),
                                            };
                                            let _ = tx.send(outcome);
                                        });
                                    }
                                }

                                // Escape: cancel search or clear results
                                (KeyCode::Esc, _) => {
                                    if app.search_in_progress {
                                        app.cancel_search();
                                    } else {
                                        app.results.clear();
                                        app.selected_result = 0;
                                    }
                                }

                                // vim-style navigation (j/k) - only when results exist
                                (KeyCode::Up, _) => {
                                    app.scroll_up();
                                }
                                (KeyCode::Down, _) => {
                                    app.scroll_down();
                                }
                                (KeyCode::Char('k'), KeyModifiers::NONE) if !app.results.is_empty() => {
                                    app.scroll_up();
                                }
                                (KeyCode::Char('j'), KeyModifiers::NONE) if !app.results.is_empty() => {
                                    app.scroll_down();
                                }

                                // Jump to first/last (g/G) - only when results exist
                                (KeyCode::Char('g'), KeyModifiers::NONE) if !app.results.is_empty() => {
                                    app.jump_to_first();
                                }
                                (KeyCode::Char('G'), KeyModifiers::SHIFT) if !app.results.is_empty() => {
                                    app.jump_to_last();
                                }

                                // Adjust top_k with [ and ]
                                (KeyCode::Char('['), _) => {
                                    app.decrease_top_k();
                                }
                                (KeyCode::Char(']'), _) => {
                                    app.increase_top_k();
                                }

                                // Force refresh
                                (KeyCode::Char('r'), KeyModifiers::NONE) => {
                                    if let Ok(Ok(stats)) = timeout(API_TIMEOUT, api.get_stats()).await {
                                        app.update_stats(stats);
                                        app.set_error(None);
                                    }
                                }

                                // Clear query (Ctrl+U)
                                (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                                    if !app.search_in_progress {
                                        app.clear_query();
                                    }
                                }

                                // Text input - j/k/g allowed when no results, q/r always reserved
                                (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                                    if !app.search_in_progress && !matches!(c, 'q' | 'r') {
                                        app.input_char(c);
                                    }
                                }
                                (KeyCode::Backspace, _) => {
                                    if !app.search_in_progress {
                                        app.input_backspace();
                                    }
                                }

                                _ => {}
                            }
                        }

                        AppMode::Detail => {
                            match (key.code, key.modifiers) {
                                // Quit
                                (KeyCode::Char('q'), KeyModifiers::NONE) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                    app.should_quit = true;
                                }

                                // Escape: back to normal mode
                                (KeyCode::Esc, _) => {
                                    app.exit_detail_mode();
                                }

                                // j/k: scroll detail text
                                (KeyCode::Char('j'), KeyModifiers::NONE) => {
                                    app.detail_scroll_down(100); // Approximate max lines
                                }
                                (KeyCode::Char('k'), KeyModifiers::NONE) => {
                                    app.detail_scroll_up();
                                }

                                // Up/Down arrows: switch between results
                                (KeyCode::Up, _) => {
                                    app.scroll_up();
                                    app.detail_scroll = 0; // Reset scroll on result change
                                }
                                (KeyCode::Down, _) => {
                                    app.scroll_down();
                                    app.detail_scroll = 0;
                                }

                                // y: copy to clipboard (placeholder - needs clipboard crate)
                                (KeyCode::Char('y'), KeyModifiers::NONE) => {
                                    // TODO: Implement clipboard copy
                                    // For now, just visual feedback
                                    app.set_error(Some("Copy not yet implemented".to_string()));
                                }

                                _ => {}
                            }
                        }
                    }
                }
            }

            // Receive search results (non-blocking)
            Some(outcome) = search_rx.recv() => {
                match outcome {
                    Ok((search_id, results)) => {
                        app.complete_search(search_id, results);
                        app.set_error(None);
                    }
                    Err(e) => {
                        app.cancel_search();
                        app.set_error(Some(format!("Search failed: {e}")));
                    }
                }
            }

            // Stats polling (with timeout)
            _ = poll_timer.tick() => {
                if app.connected {
                    match timeout(API_TIMEOUT, api.get_stats()).await {
                        Ok(Ok(stats)) => {
                            app.update_stats(stats);
                            app.set_error(None); // Clear error on successful stats
                        }
                        Ok(Err(e)) => {
                            app.set_error(Some(format!("Stats: {e}")));
                        }
                        Err(_) => {
                            app.set_error(Some("Stats: request timed out".to_string()));
                        }
                    }
                }
            }

            // Health check polling (with timeout)
            _ = health_timer.tick() => {
                let health_result = timeout(API_TIMEOUT, api.health_check()).await;
                let connected = health_result.ok().and_then(|r| r.ok()).unwrap_or(false);
                app.set_connected(connected);
            }
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyModifiers;

    #[test]
    fn test_keymodifier_pattern_matching() {
        // Verify OR pattern works with bitflags KeyModifiers
        fn matches_none_or_shift(m: KeyModifiers) -> bool {
            matches!(m, KeyModifiers::NONE | KeyModifiers::SHIFT)
        }

        // NONE should match
        assert!(matches_none_or_shift(KeyModifiers::NONE), "NONE should match");
        
        // SHIFT should match
        assert!(matches_none_or_shift(KeyModifiers::SHIFT), "SHIFT should match");
        
        // CONTROL should NOT match
        assert!(!matches_none_or_shift(KeyModifiers::CONTROL), "CONTROL should not match");
        
        // ALT should NOT match
        assert!(!matches_none_or_shift(KeyModifiers::ALT), "ALT should not match");
        
        // SHIFT+CONTROL should NOT match (combination)
        assert!(!matches_none_or_shift(KeyModifiers::SHIFT | KeyModifiers::CONTROL), "SHIFT+CONTROL should not match");
    }
}
