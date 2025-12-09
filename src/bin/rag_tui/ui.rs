use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, AppMode};

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
    }
}

/// Compact status bar: connection indicator, docs, chunks, models, status, and error
fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let connection = if app.connected {
        Span::styled("● ", Style::default().fg(Color::Green))
    } else {
        Span::styled("○ ", Style::default().fg(Color::Red))
    };

    // Truncate model names to fit (Unicode-safe)
    let embed_short = truncate_str(&app.embedding_model, 15);
    let rerank_short = app.reranker_model.as_ref()
        .map(|m| truncate_str(m, 12))
        .unwrap_or_else(|| "none".to_string());

    // Format counts compactly (e.g., "85k" instead of "85492")
    let chunks_str = format_count(app.chunk_count);

    // Status indicator with color
    let (status_str, status_color) = match app.status.as_str() {
        "ready" => ("ready", Color::Green),
        "reindexing" => ("reindexing...", Color::Yellow),
        "connecting..." => ("connecting", Color::Yellow),
        _ => (app.status.as_str(), Color::DarkGray),
    };

    let mut spans = vec![
        connection,
        Span::raw(format!("{}docs │ {} │ ", app.doc_count, chunks_str)),
        Span::styled(&embed_short, Style::default().fg(Color::Cyan)),
        Span::raw(" │ "),
        Span::styled(&rerank_short, Style::default().fg(Color::Cyan)),
        Span::raw(" │ "),
        Span::styled(status_str, Style::default().fg(status_color)),
    ];

    // Show error inline if present
    if let Some(ref err) = app.last_error {
        let err_short = if err.len() > 30 { format!("{}...", &err[..27]) } else { err.clone() };
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
        format!("{}…", truncated)
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
    let reranker = app
        .reranker_model
        .as_ref()
        .map(|s| s.as_str())
        .unwrap_or("none");

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
            .label(format!("{}/{} docs", current, total));

        frame.render_widget(gauge, area);
    } else if app.status == "reindexing" {
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
        format!(" {} {}s", spinner, elapsed)
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
            .title(format!(" Query{} ", search_status)),
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
                .take(60)
                .collect::<String>()
                .replace('\n', " ");

            // Format score with color coding and handle zero scores
            let (score_str, score_color) = if result.score <= 0.0 {
                ("--".to_string(), Color::DarkGray)
            } else if result.score >= 0.7 {
                (format!("{:.2}", result.score), Color::Green)
            } else if result.score >= 0.4 {
                (format!("{:.2}", result.score), Color::Yellow)
            } else {
                (format!("{:.2}", result.score), Color::DarkGray)
            };

            let is_selected = i == app.selected_result;

            // Build spans with score color
            let spans = vec![
                Span::raw(format!("{}. [", i + 1)),
                Span::styled(score_str, Style::default().fg(score_color)),
                Span::raw(format!("] {} \"{}...\"", provenance, text_preview)),
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

    let error = Paragraph::new(format!("Error: {}", error_text))
        .style(style)
        .block(Block::default().borders(Borders::ALL));

    frame.render_widget(error, area);
}

fn draw_keybindings_normal(frame: &mut Frame, app: &App, area: Rect) {
    let bindings = Paragraph::new(format!(
        "Enter=Detail  j/k=Nav  [/]=top_k({})  Ctrl+U=Clear  r=Refresh  q=Quit",
        app.top_k
    ))
    .style(Style::default().fg(Color::DarkGray));

    frame.render_widget(bindings, area);
}

fn draw_keybindings_detail(frame: &mut Frame, area: Rect) {
    let bindings = Paragraph::new(
        "Esc=Back  j/k=Scroll  ↑↓=Prev/Next result  y=Copy  q=Quit"
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
            } else if result.score >= 0.7 {
                (format!("{:.2}", result.score), Color::Green)
            } else if result.score >= 0.4 {
                (format!("{:.2}", result.score), Color::Yellow)
            } else {
                (format!("{:.2}", result.score), Color::DarkGray)
            };

            // Truncate document name (Unicode-safe)
            let doc_name = truncate_str(&result.document, 20);

            let is_selected = i == app.selected_result;
            let marker = if is_selected { "▶" } else { " " };

            let spans = vec![
                Span::raw(format!("{} [", marker)),
                Span::styled(score_str, Style::default().fg(score_color)),
                Span::raw(format!("] {}", doc_name)),
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
            .map(|s| format!("  §{}", s))
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
        // Should show keybindings
        assert!(buffer_contains(&buffer, "Enter=Detail"));
        assert!(buffer_contains(&buffer, "Quit"));
    }

    #[test]
    fn test_render_connected_state() {
        let mut app = make_test_app();
        app.connected = true;
        app.doc_count = 15;
        app.chunk_count = 1247;
        app.status = "ready".to_string();

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
        app.status = "reindexing".to_string();

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
