mod api;
mod app;
mod config;
mod constants;
mod ollama;
mod settings;
mod theme;
mod ui;

use std::io;
use std::time::Duration;

use constants::{
    API_TIMEOUT, DETAIL_MAX_SCROLL_ESTIMATE, HEALTH_CHECK_INTERVAL_SECS, JOB_POLL_INTERVAL_SECS,
    SEARCH_CHANNEL_CAPACITY,
};

use anyhow::Result;
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt;
use ratatui::{Terminal, backend::CrosstermBackend};
use tokio::sync::mpsc;
use tokio::time::{interval, timeout};

use api::{ApiClient, SearchResult};
#[allow(unused_imports)]
use app::Msg;
use app::{App, AppMode}; // Msg enum for future message-based architecture

/// RAII guard for terminal cleanup - ensures terminal is restored even on panic
struct TuiGuard {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TuiGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        // Note: Mouse capture intentionally disabled to allow terminal text selection
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }
}

impl Drop for TuiGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

/// Search result message for async channel
type SearchOutcome = Result<(u64, Vec<SearchResult>), String>;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present
    dotenv::dotenv().ok();

    // Load configuration from environment
    let config = config::Config::from_env();

    // Setup terminal with RAII guard (cleanup happens automatically on drop)
    let mut tui = TuiGuard::new()?;

    // Create app and API client
    let app = App::new_with_config(&config);
    let api = ApiClient::new(config.server_url.clone());

    // Run the app (any panic will still trigger TuiGuard::drop)
    let result = run_app(&mut tui.terminal, app, api, &config).await;

    if let Err(e) = &result {
        eprintln!("Error: {e}");
    }

    result
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut app: App,
    api: ApiClient,
    config: &config::Config,
) -> Result<()> {
    let mut poll_timer = interval(Duration::from_secs(config.poll_interval_secs));
    let mut health_timer = interval(Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS));
    let mut job_timer = interval(Duration::from_secs(JOB_POLL_INTERVAL_SECS));

    // Channel for async search results (bounded for backpressure)
    let (search_tx, mut search_rx) = mpsc::channel::<SearchOutcome>(SEARCH_CHANNEL_CAPACITY);

    // Initial health check and stats fetch (with timeout)
    let health_result = timeout(API_TIMEOUT, api.health_check()).await;
    app.set_connected(health_result.ok().and_then(|r| r.ok()).unwrap_or(false));
    if app.connected {
        if let Ok(Ok(stats)) = timeout(API_TIMEOUT, api.get_stats()).await {
            app.update_stats(stats);
        }
        // Check for active reindex job
        if let Ok(Ok(Some(job))) = timeout(API_TIMEOUT, api.get_active_job()).await {
            app.set_active_job(job.job_id, job.progress, job.total);
        }
    }

    // Create async event stream for terminal input (no more spawn_blocking!)
    let mut events = EventStream::new();

    loop {
        // Draw UI
        terminal.draw(|f| ui::draw(f, &app))?;

        // Handle events with timeout to allow polling
        // biased; ensures input events are checked first to prevent dropped keystrokes
        tokio::select! {
            biased;

            // Terminal input events (async stream - no dropped events!)
            Some(event_result) = events.next() => {
                // Handle resize events
                if let Ok(Event::Resize(width, height)) = event_result {
                    app.set_terminal_size(width, height);
                    // Close dropdown on resize (prevents UI glitches)
                    if app.is_dropdown_open() {
                        app.close_dropdown();
                    }
                    // UI will redraw on next loop iteration
                }
                // Handle key events
                else if let Ok(Event::Key(key)) = event_result {
                    // Mode-specific keybindings
                    match app.mode {
                        AppMode::Normal => {
                            match (key.code, key.modifiers) {
                                // === CONTROL KEY COMMANDS (always work) ===

                                // Quit: Ctrl+C only (frees 'q' for typing)
                                (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                    app.should_quit = true;
                                }

                                // Refresh stats: Ctrl+R (frees 'r' for typing)
                                (KeyCode::Char('r'), KeyModifiers::CONTROL) => {
                                    if let Ok(Ok(stats)) = timeout(API_TIMEOUT, api.get_stats()).await {
                                        app.update_stats(stats);
                                        app.set_error(None);
                                    }
                                }

                                // Clear query: Ctrl+U
                                (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                                    if !app.search_in_progress {
                                        app.clear_query();
                                    }
                                }

                                // === SPECIAL KEYS ===

                                // Help toggle (only when query is empty, so ? can be typed in queries)
                                (KeyCode::Char('?'), _) if app.query_input.is_empty() => {
                                    app.toggle_help();
                                }

                                // Enter: submit search (if query changed) OR open detail view
                                // Priority: 1) New/modified query → search, 2) Results exist → detail
                                (KeyCode::Enter, _) => {
                                    let should_search = !app.search_in_progress
                                        && !app.query_input.is_empty()
                                        && (app.results.is_empty() || app.query_changed_since_search());

                                    if should_search {
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
                                            let _ = tx.send(outcome).await;
                                        });
                                    } else if !app.results.is_empty() {
                                        app.enter_detail_mode();
                                    }
                                }

                                // Escape: cancel search or clear results
                                (KeyCode::Esc, _) => {
                                    if app.search_in_progress {
                                        app.cancel_search();
                                    } else if !app.results.is_empty() {
                                        app.results.clear();
                                        app.selected_result = 0;
                                    } else if !app.query_input.is_empty() {
                                        app.clear_query();
                                    }
                                }

                                // Backspace: always goes to query
                                (KeyCode::Backspace, _) => {
                                    if !app.search_in_progress {
                                        app.input_backspace();
                                    }
                                }

                                // === NAVIGATION (arrow keys) ===

                                (KeyCode::Up, _) => {
                                    app.scroll_up();
                                }
                                (KeyCode::Down, _) => {
                                    app.scroll_down();
                                }
                                (KeyCode::PageUp, _) => {
                                    app.scroll_page_up();
                                }
                                (KeyCode::PageDown, _) => {
                                    app.scroll_page_down();
                                }
                                (KeyCode::Home, _) => {
                                    app.jump_to_first();
                                }
                                (KeyCode::End, _) => {
                                    app.jump_to_last();
                                }

                                // === VIM NAVIGATION (only when results exist) ===

                                (KeyCode::Char('k'), KeyModifiers::NONE) if !app.results.is_empty() => {
                                    app.scroll_up();
                                }
                                (KeyCode::Char('j'), KeyModifiers::NONE) if !app.results.is_empty() => {
                                    app.scroll_down();
                                }
                                (KeyCode::Char('g'), KeyModifiers::NONE) if !app.results.is_empty() => {
                                    app.jump_to_first();
                                }
                                (KeyCode::Char('G'), KeyModifiers::SHIFT) if !app.results.is_empty() => {
                                    app.jump_to_last();
                                }

                                // === TOP_K ADJUSTMENT (always available) ===

                                (KeyCode::Char('['), KeyModifiers::NONE) => {
                                    app.decrease_top_k();
                                }
                                (KeyCode::Char(']'), KeyModifiers::NONE) => {
                                    app.increase_top_k();
                                }

                                // === REINDEX (Shift+R - uppercase only) ===

                                (KeyCode::Char('R'), KeyModifiers::SHIFT) => {
                                    if !app.reindex_in_progress {
                                        match timeout(API_TIMEOUT, api.start_reindex()).await {
                                            Ok(Ok(resp)) => {
                                                app.set_active_job(resp.job_id, 0, 0);
                                                app.set_error(None);
                                            }
                                            Ok(Err(e)) => {
                                                app.set_error(Some(format!("Reindex: {e}")));
                                            }
                                            Err(_) => {
                                                app.set_error(Some("Reindex: request timed out".to_string()));
                                            }
                                        }
                                    }
                                }

                                // === SETTINGS (Shift+S - uppercase only) ===

                                (KeyCode::Char('S'), KeyModifiers::SHIFT) => {
                                    app.enter_settings_mode();
                                }

                                // === TEXT INPUT (all other characters) ===
                                // j/k/g go to input when no results, all letters work

                                (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                                    if !app.search_in_progress {
                                        app.input_char(c);
                                    }
                                }

                                _ => {}
                            }
                        }

                        AppMode::Detail => {
                            match (key.code, key.modifiers) {
                                // Quit app: Ctrl+C only
                                (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                    app.should_quit = true;
                                }

                                // Exit detail view: q or Escape
                                (KeyCode::Char('q'), KeyModifiers::NONE) | (KeyCode::Esc, _) => {
                                    app.exit_detail_mode();
                                }

                                // j/k: scroll detail text (single line)
                                (KeyCode::Char('j'), KeyModifiers::NONE) => {
                                    app.detail_scroll_down(DETAIL_MAX_SCROLL_ESTIMATE);
                                }
                                (KeyCode::Char('k'), KeyModifiers::NONE) => {
                                    app.detail_scroll_up();
                                }

                                // PageUp/PageDown: scroll detail text (page at a time)
                                (KeyCode::PageUp, _) => {
                                    app.detail_scroll_page_up();
                                }
                                (KeyCode::PageDown, _) => {
                                    app.detail_scroll_page_down(DETAIL_MAX_SCROLL_ESTIMATE);
                                }

                                // Home/End: jump to top/bottom of detail
                                (KeyCode::Home, _) => {
                                    app.detail_scroll = 0;
                                }
                                (KeyCode::End, _) => {
                                    app.detail_scroll = DETAIL_MAX_SCROLL_ESTIMATE.saturating_sub(1);
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

                                // y: copy to clipboard
                                (KeyCode::Char('y'), KeyModifiers::NONE) => {
                                    if let Some(result) = app.selected_result_ref() {
                                        let text = result.text.clone();
                                        match arboard::Clipboard::new() {
                                            Ok(mut clipboard) => {
                                                match clipboard.set_text(&text) {
                                                    Ok(()) => {
                                                        app.set_error(Some("Copied to clipboard".to_string()));
                                                    }
                                                    Err(e) => {
                                                        app.set_error(Some(format!("Copy failed: {e}")));
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                app.set_error(Some(format!("Clipboard unavailable: {e}")));
                                            }
                                        }
                                    }
                                }

                                // Help
                                (KeyCode::Char('?'), _) => {
                                    app.toggle_help();
                                }

                                _ => {}
                            }
                        }

                        AppMode::Help => {
                            match (key.code, key.modifiers) {
                                // Quit app: Ctrl+C only
                                (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                    app.should_quit = true;
                                }

                                // Close help: q, Escape, or ?
                                (KeyCode::Char('q'), KeyModifiers::NONE)
                                    | (KeyCode::Esc, _)
                                    | (KeyCode::Char('?'), _) => {
                                    app.exit_help_mode();
                                }

                                _ => {}
                            }
                        }

                        AppMode::Settings => {
                            // Check if dropdown is open (highest priority)
                            if app.is_dropdown_open() {
                                match (key.code, key.modifiers) {
                                    // Navigate dropdown
                                    (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                                        app.dropdown_up();
                                    }
                                    (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                                        app.dropdown_down();
                                    }
                                    // Confirm selection
                                    (KeyCode::Enter, _) => {
                                        app.dropdown_confirm();
                                    }
                                    // Cancel dropdown
                                    (KeyCode::Esc, _) | (KeyCode::Char('q'), KeyModifiers::NONE) => {
                                        app.close_dropdown();
                                    }
                                    _ => {}
                                }
                            // Check if we're in edit mode for a setting
                            } else if app.settings.editing {
                                match (key.code, key.modifiers) {
                                    // Confirm edit
                                    (KeyCode::Enter, _) => {
                                        app.settings.confirm_edit();
                                    }
                                    // Cancel edit
                                    (KeyCode::Esc, _) => {
                                        app.settings.cancel_edit();
                                    }
                                    // Text input
                                    (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                                        app.settings.input_char(c);
                                    }
                                    (KeyCode::Backspace, _) => {
                                        app.settings.backspace();
                                    }
                                    (KeyCode::Delete, _) => {
                                        app.settings.delete();
                                    }
                                    // Cursor movement
                                    (KeyCode::Left, _) => {
                                        app.settings.cursor_left();
                                    }
                                    (KeyCode::Right, _) => {
                                        app.settings.cursor_right();
                                    }
                                    (KeyCode::Home, _) => {
                                        app.settings.cursor_home();
                                    }
                                    (KeyCode::End, _) => {
                                        app.settings.cursor_end();
                                    }
                                    _ => {}
                                }
                            } else {
                                match (key.code, key.modifiers) {
                                    // Quit app: Ctrl+C only
                                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                        app.should_quit = true;
                                    }

                                    // Exit settings: q or Escape
                                    (KeyCode::Char('q'), KeyModifiers::NONE) | (KeyCode::Esc, _) => {
                                        app.exit_settings_mode();
                                    }

                                    // Navigate settings
                                    (KeyCode::Up, _) | (KeyCode::Char('k'), KeyModifiers::NONE) => {
                                        app.settings.prev();
                                    }
                                    (KeyCode::Down, _) | (KeyCode::Char('j'), KeyModifiers::NONE) => {
                                        app.settings.next();
                                    }

                                    // Edit current setting
                                    (KeyCode::Enter, _) => {
                                        // Check if this setting has dropdown capability
                                        if app.current_setting_has_dropdown() {
                                            // Open dropdown overlay
                                            app.open_dropdown();
                                        } else {
                                            // Open text edit for other settings
                                            app.settings.start_edit();
                                        }
                                    }

                                    // Cycle options with Tab (for dropdown-style)
                                    (KeyCode::Tab, _) => {
                                        if app.settings.current().map(|s| s.options.is_some()).unwrap_or(false) {
                                            app.settings.cycle_option(true);
                                        }
                                    }
                                    (KeyCode::BackTab, _) => {
                                        if app.settings.current().map(|s| s.options.is_some()).unwrap_or(false) {
                                            app.settings.cycle_option(false);
                                        }
                                    }

                                    // Save settings: Ctrl+S
                                    (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                                        app.save_settings();
                                    }

                                    // Reset current field to original value: r
                                    (KeyCode::Char('r'), KeyModifiers::NONE) => {
                                        if let Some(setting) = app.settings.current() {
                                            let name = setting.display_name.clone();
                                            app.settings.reset_current();
                                            app.settings_message = Some((format!("{name} reset to default"), false));
                                        }
                                    }

                                    // Reset all to original values: R (Shift+r)
                                    (KeyCode::Char('R'), KeyModifiers::SHIFT) => {
                                        app.settings.reset_all();
                                        app.settings_message = Some(("All settings reset to original values".to_string(), false));
                                    }

                                    _ => {}
                                }
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

            // Receive model fetch results (for settings dropdowns)
            // Uses if-guard pattern: branch only active when receiver exists
            result = async {
                app.model_fetch_rx.as_mut().unwrap().await
            }, if app.model_fetch_rx.is_some() => {
                // Clear the receiver since it's consumed
                app.model_fetch_rx = None;
                match result {
                    Ok(models_result) => {
                        app.handle_model_fetch_result(models_result);
                    }
                    Err(_) => {
                        // Sender dropped (shouldn't happen normally)
                        app.handle_model_fetch_result(Err("Model fetch cancelled".to_string()));
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

            // Job progress polling (only when reindex is in progress)
            _ = job_timer.tick(), if app.reindex_in_progress => {
                if let Some(ref job_id) = app.active_job_id.clone() {
                    match timeout(API_TIMEOUT, api.get_job_status(job_id)).await {
                        Ok(Ok(job)) => {
                            app.update_job_progress(job.progress, job.total);
                            // Check if job completed
                            if job.status == "completed" || job.status == "failed" {
                                app.clear_active_job();
                                if job.status == "failed" {
                                    app.set_error(job.error.or(Some("Reindex failed".to_string())));
                                } else {
                                    // Refresh stats after successful reindex
                                    if let Ok(Ok(stats)) = timeout(API_TIMEOUT, api.get_stats()).await {
                                        app.update_stats(stats);
                                    }
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            app.set_error(Some(format!("Job status: {e}")));
                        }
                        Err(_) => {
                            // Timeout, will retry next tick
                        }
                    }
                }
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
        assert!(
            matches_none_or_shift(KeyModifiers::NONE),
            "NONE should match"
        );

        // SHIFT should match
        assert!(
            matches_none_or_shift(KeyModifiers::SHIFT),
            "SHIFT should match"
        );

        // CONTROL should NOT match
        assert!(
            !matches_none_or_shift(KeyModifiers::CONTROL),
            "CONTROL should not match"
        );

        // ALT should NOT match
        assert!(
            !matches_none_or_shift(KeyModifiers::ALT),
            "ALT should not match"
        );

        // SHIFT+CONTROL should NOT match (combination)
        assert!(
            !matches_none_or_shift(KeyModifiers::SHIFT | KeyModifiers::CONTROL),
            "SHIFT+CONTROL should not match"
        );
    }
}
