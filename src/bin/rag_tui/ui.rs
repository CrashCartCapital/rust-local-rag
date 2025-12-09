use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, AppMode, ServerStatus};
use crate::constants::{
    DOC_NAME_MAX_CHARS, ERROR_TRUNCATE_CHARS, MODEL_NAME_MAX_CHARS, PREVIEW_CHARS,
    RERANKER_NAME_MAX_CHARS, SCORE_THRESHOLD_HIGH, SCORE_THRESHOLD_MEDIUM,
};

pub fn draw(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),  // Compact status bar
            Constraint::Length(3),  // Query input
            Constraint::Min(8),     // Results (maximized)
            Constraint::Length(1),  // Keybindings
        ])
        .split(frame.area());

    draw_status_bar(frame, app, chunks[0]);
    draw_query_input(frame, app, chunks[1]);

    // Different layout based on mode
    match app.mode {
        AppMode::Normal => {
            draw_results(frame, app, chunks[2]);
            draw_keybindings_normal(frame, app, chunks[3]);
        }
        AppMode::Detail => {
            draw_split_pane(frame, app, chunks[2]);
            draw_keybindings_detail(frame, chunks[3]);
        }
        AppMode::Help => {
            draw_results(frame, app, chunks[2]);
            draw_keybindings_normal(frame, app, chunks[3]);
            // Overlay help on top
            draw_help_overlay(frame);
        }
    }
}

/// Centered help overlay showing all keybindings
fn draw_help_overlay(frame: &mut Frame) {
    let area = frame.area();

    // Calculate centered area (60% width, 70% height)
    let popup_width = (area.width * 60 / 100).min(70);
    let popup_height = (area.height * 70 / 100).min(24);
    let popup_x = (area.width.saturating_sub(popup_width)) / 2;
    let popup_y = (area.height.saturating_sub(popup_height)) / 2;

    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear the background
    let clear = Block::default().style(Style::default().bg(Color::Black));
    frame.render_widget(clear, popup_area);

    let help_text = vec![
        Line::from(vec![
            Span::styled("RAG-TUI Help", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("── Search Mode ──", Style::default().fg(Color::Yellow)),
        ]),
        Line::from("  Enter        Execute search / Open detail view"),
        Line::from("  Esc          Cancel search / Clear results"),
        Line::from("  Ctrl+U       Clear query input"),
        Line::from("  [ / ]        Decrease / Increase top_k"),
        Line::from(""),
        Line::from(vec![
            Span::styled("── Navigation ──", Style::default().fg(Color::Yellow)),
        ]),
        Line::from("  j/k, ↑/↓     Scroll up/down (one item)"),
        Line::from("  PgUp/PgDn    Scroll up/down (page)"),
        Line::from("  Home/End     Jump to first/last"),
        Line::from("  g / G        Jump to first/last (vim)"),
        Line::from(""),
        Line::from(vec![
            Span::styled("── Detail View ──", Style::default().fg(Color::Yellow)),
        ]),
        Line::from("  Enter        Open detail view (from results)"),
        Line::from("  Esc          Return to list view"),
        Line::from("  j / k        Scroll detail text"),
        Line::from("  y            Copy to clipboard"),
        Line::from(""),
        Line::from(vec![
            Span::styled("── General ──", Style::default().fg(Color::Yellow)),
        ]),
        Line::from("  ?            Toggle this help"),
        Line::from("  Ctrl+R       Refresh stats"),
        Line::from("  Shift+R      Trigger reindex"),
        Line::from("  Ctrl+C       Quit"),
        Line::from(""),
        Line::from(vec![
            Span::styled("Press ? or Esc to close", Style::default().fg(Color::DarkGray)),
        ]),
    ];

    let help = Paragraph::new(help_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Help ")
                .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(help, popup_area);
}

/// Compact status bar: mode, connection indicator, docs, chunks, models, status, and error
fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    // Mode indicator with distinct styling
    let (mode_str, mode_color) = match app.mode {
        AppMode::Normal => ("NORMAL", Color::Blue),
        AppMode::Detail => ("DETAIL", Color::Magenta),
        AppMode::Help => ("HELP", Color::Yellow),
    };

    let connection = if app.connected {
        Span::styled("● ", Style::default().fg(Color::Green))
    } else {
        Span::styled("○ ", Style::default().fg(Color::Red))
    };

    // Truncate model names to fit (Unicode-safe)
    let embed_short = truncate_str(&app.embedding_model, MODEL_NAME_MAX_CHARS);
    let rerank_short = app
        .reranker_model
        .as_ref()
        .map(|m| truncate_str(m, RERANKER_NAME_MAX_CHARS))
        .unwrap_or_else(|| "none".to_string());

    // Format counts compactly (e.g., "85k" instead of "85492")
    let chunks_str = format_count(app.chunk_count);

    // Status indicator with color (prioritize reindex_in_progress for accurate TUI state)
    let (status_str, status_color): (String, Color) = if app.reindex_in_progress {
        if let Some((progress, total)) = app.job_progress {
            if total > 0 {
                let pct = (progress * 100) / total;
                (format!("reindexing {}%", pct), Color::Yellow)
            } else {
                ("reindexing...".to_string(), Color::Yellow)
            }
        } else {
            ("reindexing...".to_string(), Color::Yellow)
        }
    } else {
        match &app.status {
            ServerStatus::Ready => ("ready".to_string(), Color::Green),
            ServerStatus::Reindexing => ("reindexing...".to_string(), Color::Yellow),
            ServerStatus::Connecting => ("connecting".to_string(), Color::Yellow),
            ServerStatus::Unknown(s) => (s.clone(), Color::DarkGray),
        }
    };

    let mut spans = vec![
        Span::styled(
            format!("[{mode_str}]"),
            Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        connection,
        Span::raw(format!("{}docs │ {} │ ", app.doc_count, chunks_str)),
        Span::styled(&embed_short, Style::default().fg(Color::Cyan)),
        Span::raw(" │ "),
        Span::styled(&rerank_short, Style::default().fg(Color::Cyan)),
        Span::raw(" │ "),
        Span::styled(&status_str, Style::default().fg(status_color)),
    ];

    // Show error inline if present
    if let Some(ref err) = app.last_error {
        let err_short = if err.len() > ERROR_TRUNCATE_CHARS {
            format!("{}...", &err[..ERROR_TRUNCATE_CHARS.saturating_sub(3)])
        } else {
            err.clone()
        };
        spans.push(Span::raw(" │ "));
        spans.push(Span::styled(err_short, Style::default().fg(Color::Red)));
    }

    let status_bar = Paragraph::new(Line::from(spans));
    frame.render_widget(status_bar, area);
}

/// Truncate string for compact display (Unicode-safe)
fn truncate_str(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{truncated}…")
    }
}

/// Format large counts compactly (e.g., 85492 -> "85k")
fn format_count(count: usize) -> String {
    if count >= 1000 {
        format!("{}k", count / 1000)
    } else {
        count.to_string()
    }
}

#[allow(dead_code)]
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let connection_indicator = if app.connected {
        Span::styled("●", Style::default().fg(Color::Green))
    } else {
        Span::styled("○", Style::default().fg(Color::Red))
    };

    let connection_text = if app.connected {
        " Connected"
    } else {
        " Disconnected"
    };

    let header = Paragraph::new(Line::from(vec![
        Span::raw(format!(
            "Docs: {} │ Chunks: {} │ Status: {}",
            app.doc_count, app.chunk_count, app.status
        )),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(vec![
                Span::raw(" RAG-TUI "),
                connection_indicator,
                Span::raw(connection_text),
                Span::raw(" "),
            ]),
    );

    frame.render_widget(header, area);
}

#[allow(dead_code)]
fn draw_models(frame: &mut Frame, app: &App, area: Rect) {
    let reranker = app.reranker_model.as_deref().unwrap_or("none");

    let models = Paragraph::new(format!(
        "Embed: {} │ Rerank: {}",
        app.embedding_model, reranker
    ))
    .block(Block::default().borders(Borders::ALL).title(" Models "));

    frame.render_widget(models, area);
}

#[allow(dead_code)]
fn draw_progress(frame: &mut Frame, app: &App, area: Rect) {
    if let Some((current, total)) = app.job_progress {
        let ratio = if total > 0 {
            current as f64 / total as f64
        } else {
            0.0
        };

        let gauge = Gauge::default()
            .block(Block::default().borders(Borders::ALL).title(" Progress "))
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(ratio)
            .label(format!("{current}/{total} docs"));

        frame.render_widget(gauge, area);
    } else if app.status.is_reindexing() {
        // Show indeterminate progress
        let progress = Paragraph::new("Reindexing in progress...")
            .style(Style::default().fg(Color::Yellow))
            .block(Block::default().borders(Borders::ALL).title(" Progress "));
        frame.render_widget(progress, area);
    } else {
        // Empty block when no progress
        let empty = Block::default().borders(Borders::ALL).title(" Progress ");
        frame.render_widget(empty, area);
    }
}

#[allow(dead_code)]
fn draw_config(frame: &mut Frame, app: &App, area: Rect) {
    let config = Paragraph::new(app.config_summary.as_str())
        .block(Block::default().borders(Borders::ALL).title(" Config "));

    frame.render_widget(config, area);
}

fn draw_query_input(frame: &mut Frame, app: &App, area: Rect) {
    let search_status = if app.search_in_progress {
        let elapsed = app.search_elapsed_secs().unwrap_or(0);
        // Simple ASCII spinner
        let spinner = match elapsed % 4 {
            0 => "|",
            1 => "/",
            2 => "-",
            _ => "\\",
        };
        format!(" {spinner} {elapsed}s")
    } else {
        String::new()
    };

    let input = Paragraph::new(Line::from(vec![
        Span::raw(&app.query_input),
        Span::styled("_", Style::default().add_modifier(Modifier::SLOW_BLINK)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Query{search_status} ")),
    );

    frame.render_widget(input, area);
}

fn draw_results(frame: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let provenance = if result.page_number > 0 {
                format!("{} (p.{})", result.document, result.page_number)
            } else {
                result.document.clone()
            };

            // Truncate text for display
            let text_preview: String = result
                .text
                .chars()
                .take(PREVIEW_CHARS)
                .collect::<String>()
                .replace('\n', " ");

            // Format score with color coding and handle zero scores
            let (score_str, score_color) = if result.score <= 0.0 {
                ("--".to_string(), Color::DarkGray)
            } else if result.score >= SCORE_THRESHOLD_HIGH {
                (format!("{:.2}", result.score), Color::Green)
            } else if result.score >= SCORE_THRESHOLD_MEDIUM {
                (format!("{:.2}", result.score), Color::Yellow)
            } else {
                (format!("{:.2}", result.score), Color::DarkGray)
            };

            let is_selected = i == app.selected_result;

            // Build spans with score color
            let idx = i + 1;
            let spans = vec![
                Span::raw(format!("{idx}. [")),
                Span::styled(score_str, Style::default().fg(score_color)),
                Span::raw(format!("] {provenance} \"{text_preview}...\"")),
            ];

            let style = if is_selected {
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(spans)).style(style)
        })
        .collect();

    let results_title = if app.results.is_empty() && !app.search_in_progress {
        " Results (no results) "
    } else {
        " Results "
    };

    let results = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(results_title));

    frame.render_widget(results, area);
}

#[allow(dead_code)]
fn draw_error(frame: &mut Frame, app: &App, area: Rect) {
    let error_text = app.last_error.as_deref().unwrap_or("(none)");
    let style = if app.last_error.is_some() {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let error = Paragraph::new(format!("Error: {error_text}"))
        .style(style)
        .block(Block::default().borders(Borders::ALL));

    frame.render_widget(error, area);
}

fn draw_keybindings_normal(frame: &mut Frame, app: &App, area: Rect) {
    let bindings = Paragraph::new(format!(
        "Enter=Search/Detail  j/k=Nav  [/]=top_k({})  C-U=Clear  C-R=Refresh  C-c=Quit  ?=Help",
        app.top_k
    ))
    .style(Style::default().fg(Color::DarkGray));

    frame.render_widget(bindings, area);
}

fn draw_keybindings_detail(frame: &mut Frame, area: Rect) {
    let bindings = Paragraph::new(
        "Esc/q=Back  j/k=Scroll  ↑↓=Prev/Next result  y=Copy  C-c=Quit"
    )
    .style(Style::default().fg(Color::DarkGray));

    frame.render_widget(bindings, area);
}

/// Split-pane layout: results list on left, detail view on right
fn draw_split_pane(frame: &mut Frame, app: &App, area: Rect) {
    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(35),  // Results list (compact)
            Constraint::Percentage(65),  // Detail view (larger)
        ])
        .split(area);

    draw_results_compact(frame, app, panes[0]);
    draw_detail_view(frame, app, panes[1]);
}

/// Compact results list for detail mode (shorter text)
fn draw_results_compact(frame: &mut Frame, app: &App, area: Rect) {
    let items: Vec<ListItem> = app
        .results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            // Very compact: just score and truncated doc name
            let (score_str, score_color) = if result.score <= 0.0 {
                ("--".to_string(), Color::DarkGray)
            } else if result.score >= SCORE_THRESHOLD_HIGH {
                (format!("{:.2}", result.score), Color::Green)
            } else if result.score >= SCORE_THRESHOLD_MEDIUM {
                (format!("{:.2}", result.score), Color::Yellow)
            } else {
                (format!("{:.2}", result.score), Color::DarkGray)
            };

            // Truncate document name (Unicode-safe)
            let doc_name = truncate_str(&result.document, DOC_NAME_MAX_CHARS);

            let is_selected = i == app.selected_result;
            let marker = if is_selected { "▶" } else { " " };

            let spans = vec![
                Span::raw(format!("{marker} [")),
                Span::styled(score_str, Style::default().fg(score_color)),
                Span::raw(format!("] {doc_name}")),
            ];

            let style = if is_selected {
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };

            ListItem::new(Line::from(spans)).style(style)
        })
        .collect();

    let results = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Results "));

    frame.render_widget(results, area);
}

/// Detail view showing full text of selected result
fn draw_detail_view(frame: &mut Frame, app: &App, area: Rect) {
    if let Some(result) = app.selected_result_ref() {
        // Header with metadata
        let provenance = if result.page_number > 0 {
            format!("{} (p.{})", result.document, result.page_number)
        } else {
            result.document.clone()
        };

        let (score_str, score_color) = if result.score <= 0.0 {
            ("--".to_string(), Color::DarkGray)
        } else {
            (format!("{:.2}", result.score), Color::Green)
        };

        let section_info = result.section.as_ref()
            .map(|s| format!("  §{s}"))
            .unwrap_or_default();

        let header = Line::from(vec![
            Span::styled(&provenance, Style::default().fg(Color::Cyan)),
            Span::raw("  Score: "),
            Span::styled(score_str, Style::default().fg(score_color)),
            Span::styled(section_info, Style::default().fg(Color::DarkGray)),
        ]);

        // Full text with scroll
        let text_lines: Vec<Line> = std::iter::once(header)
            .chain(std::iter::once(Line::from("─".repeat(area.width.saturating_sub(2) as usize))))
            .chain(result.text.lines().map(|line| Line::from(line.to_string())))
            .skip(app.detail_scroll)
            .collect();

        let detail = Paragraph::new(text_lines)
            .block(Block::default().borders(Borders::ALL).title(" Detail "))
            .wrap(Wrap { trim: false });

        frame.render_widget(detail, area);
    } else {
        let empty = Paragraph::new("No result selected")
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title(" Detail "));
        frame.render_widget(empty, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::SearchResult;
    use crate::app::App;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    /// Helper to create a test app with configurable state
    fn make_test_app() -> App {
        App::new("http://localhost:3046".to_string())
    }

    fn make_test_result(doc: &str, score: f32, text: &str) -> SearchResult {
        SearchResult {
            document: doc.to_string(),
            text: text.to_string(),
            score,
            page_number: 1,
            chunk_id: "test-chunk".to_string(),
            section: None,
        }
    }

    /// Helper to render app to a TestBackend buffer
    fn render_to_buffer(app: &App, width: u16, height: u16) -> ratatui::buffer::Buffer {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        terminal.backend().buffer().clone()
    }

    /// Helper to check if a string appears anywhere in the buffer
    fn buffer_contains(buffer: &ratatui::buffer::Buffer, needle: &str) -> bool {
        let area = buffer.area();
        for y in area.top()..area.bottom() {
            let mut line = String::new();
            for x in area.left()..area.right() {
                line.push_str(buffer.cell((x, y)).unwrap().symbol());
            }
            if line.contains(needle) {
                return true;
            }
        }
        false
    }

    #[test]
    fn test_render_initial_state() {
        let app = make_test_app();
        let buffer = render_to_buffer(&app, 80, 30);

        // Should show disconnected indicator (empty circle)
        assert!(buffer_contains(&buffer, "○"));
        // Should show keybindings (updated to new Ctrl-modifier scheme)
        assert!(buffer_contains(&buffer, "Enter=Search"));
        assert!(buffer_contains(&buffer, "C-c=Quit"));
    }

    #[test]
    fn test_render_connected_state() {
        let mut app = make_test_app();
        app.connected = true;
        app.doc_count = 15;
        app.chunk_count = 1247;
        app.status = ServerStatus::Ready;

        let buffer = render_to_buffer(&app, 80, 30);

        // Should show connected indicator (filled circle)
        assert!(buffer_contains(&buffer, "●"));
        // Should show stats in compact format
        assert!(buffer_contains(&buffer, "15docs"));
        assert!(buffer_contains(&buffer, "1k")); // 1247 -> "1k"
        assert!(buffer_contains(&buffer, "ready"));
    }

    #[test]
    fn test_render_with_models() {
        let mut app = make_test_app();
        app.embedding_model = "nomic-embed-text".to_string();
        app.reranker_model = Some("phi4-mini".to_string());

        let buffer = render_to_buffer(&app, 80, 30);

        // Should show model names in compact status bar (truncated to fit)
        // "nomic-embed-text" (16 chars) gets truncated to "nomic-embed-te…" (15 chars)
        assert!(buffer_contains(&buffer, "nomic-embed"));
        assert!(buffer_contains(&buffer, "phi4-mini"));
    }

    #[test]
    fn test_render_with_query_input() {
        let mut app = make_test_app();
        app.query_input = "algorithmic trading".to_string();

        let buffer = render_to_buffer(&app, 80, 30);

        // Should show the query
        assert!(buffer_contains(&buffer, "algorithmic trading"));
    }

    #[test]
    fn test_render_with_results() {
        let mut app = make_test_app();
        app.results = vec![
            make_test_result("trading.pdf", 0.95, "Position sizing strategies..."),
            make_test_result("market.pdf", 0.87, "VWAP algorithms..."),
        ];

        let buffer = render_to_buffer(&app, 100, 30);

        // Should show results with scores
        assert!(buffer_contains(&buffer, "trading.pdf"));
        assert!(buffer_contains(&buffer, "0.95"));
        assert!(buffer_contains(&buffer, "market.pdf"));
    }

    #[test]
    fn test_render_with_error() {
        let mut app = make_test_app();
        app.last_error = Some("Connection timeout".to_string());

        let buffer = render_to_buffer(&app, 80, 30);

        // Should show error message
        assert!(buffer_contains(&buffer, "Connection timeout"));
    }

    #[test]
    fn test_render_reindexing_state() {
        let mut app = make_test_app();
        app.status = ServerStatus::Reindexing;

        let buffer = render_to_buffer(&app, 80, 30);

        // Should show reindexing in compact status bar
        assert!(buffer_contains(&buffer, "reindexing..."));
    }

    #[test]
    fn test_render_no_reranker() {
        let mut app = make_test_app();
        app.embedding_model = "nomic-embed-text".to_string();
        app.reranker_model = None;

        let buffer = render_to_buffer(&app, 80, 30);

        // Should show "none" for reranker in compact status bar
        assert!(buffer_contains(&buffer, "none"));
    }

    #[test]
    fn test_render_small_terminal() {
        // Test that UI doesn't panic on small terminal sizes
        let app = make_test_app();

        // Very small terminal - should not panic
        let _buffer = render_to_buffer(&app, 40, 15);

        // Minimum viable size
        let _buffer = render_to_buffer(&app, 60, 20);
    }

    #[test]
    fn test_render_large_terminal() {
        // Test that UI scales to large terminal sizes
        let mut app = make_test_app();
        app.results = vec![
            make_test_result("doc1.pdf", 0.9, "Content 1"),
            make_test_result("doc2.pdf", 0.8, "Content 2"),
            make_test_result("doc3.pdf", 0.7, "Content 3"),
        ];

        let buffer = render_to_buffer(&app, 200, 60);

        // Should still contain expected elements (compact status bar + results)
        assert!(buffer_contains(&buffer, "doc1.pdf"));
        assert!(buffer_contains(&buffer, "doc2.pdf"));
        assert!(buffer_contains(&buffer, "doc3.pdf"));
    }

    #[test]
    fn test_render_long_query() {
        let mut app = make_test_app();
        app.query_input = "This is a very long query that might overflow the input field in a narrow terminal window".to_string();

        // Should not panic with long query
        let _buffer = render_to_buffer(&app, 80, 30);
    }

    #[test]
    fn test_render_long_result_text() {
        let mut app = make_test_app();
        app.results = vec![make_test_result(
            "document.pdf",
            0.95,
            "This is a very long result text that contains many words and might need to be truncated when displayed in the results list to prevent overflow issues",
        )];

        // Should not panic with long result text
        let _buffer = render_to_buffer(&app, 80, 30);
    }

    #[test]
    fn test_render_empty_results_message() {
        let mut app = make_test_app();
        app.results = vec![]; // Empty results

        let buffer = render_to_buffer(&app, 80, 30);

        // Should show "no results" indicator
        assert!(buffer_contains(&buffer, "no results"));
    }
}
