use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Wrap},
};

use crate::app::{ActiveDropdown, App, AppMode, ModelFetchState, ServerStatus};
use crate::constants::{
    DOC_NAME_MAX_CHARS, ERROR_TRUNCATE_CHARS, MODEL_NAME_MAX_CHARS, PREVIEW_CHARS,
    RERANKER_NAME_MAX_CHARS, SCORE_THRESHOLD_HIGH, SCORE_THRESHOLD_MEDIUM,
};
use crate::settings::ValidationState;

pub fn draw(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Compact status bar
            Constraint::Length(3), // Query input
            Constraint::Min(8),    // Results (maximized)
            Constraint::Length(1), // Keybindings
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
        AppMode::Settings => {
            draw_results(frame, app, chunks[2]);
            draw_keybindings_settings(frame, app, chunks[3]);
            // Overlay settings on top
            draw_settings_overlay(frame, app);
            // Overlay dropdown on top of settings if open
            if app.is_dropdown_open() {
                draw_dropdown_overlay(frame, app);
            }
        }
    }
}

/// Centered help overlay showing all keybindings
fn draw_help_overlay(frame: &mut Frame) {
    let area = frame.area();

    // Skip rendering on very small terminals
    if area.width < 20 || area.height < 10 {
        return;
    }

    // Calculate centered area (60% width, 70% height) with minimum sizes
    let popup_width = (area.width * 60 / 100).clamp(20, 70);
    let popup_height = (area.height * 70 / 100).clamp(10, 24);
    let popup_x = (area.width.saturating_sub(popup_width)) / 2;
    let popup_y = (area.height.saturating_sub(popup_height)) / 2;

    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear the background
    let clear = Block::default().style(Style::default().bg(Color::Black));
    frame.render_widget(clear, popup_area);

    let help_text = vec![
        Line::from(vec![Span::styled(
            "RAG-TUI Help",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "── Search Mode ──",
            Style::default().fg(Color::Yellow),
        )]),
        Line::from("  Enter        Execute search / Open detail view"),
        Line::from("  Esc          Cancel search / Clear results"),
        Line::from("  Ctrl+U       Clear query input"),
        Line::from("  [ / ]        Decrease / Increase top_k"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "── Navigation ──",
            Style::default().fg(Color::Yellow),
        )]),
        Line::from("  j/k, ↑/↓     Scroll up/down (one item)"),
        Line::from("  PgUp/PgDn    Scroll up/down (page)"),
        Line::from("  Home/End     Jump to first/last"),
        Line::from("  g / G        Jump to first/last (vim)"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "── Detail View ──",
            Style::default().fg(Color::Yellow),
        )]),
        Line::from("  Enter        Open detail view (from results)"),
        Line::from("  Esc          Return to list view"),
        Line::from("  j / k        Scroll detail text"),
        Line::from("  y            Copy to clipboard"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "── General ──",
            Style::default().fg(Color::Yellow),
        )]),
        Line::from("  ?            Toggle this help"),
        Line::from("  Shift+S      Open settings"),
        Line::from("  Ctrl+R       Refresh stats"),
        Line::from("  Shift+R      Trigger reindex"),
        Line::from("  Ctrl+C       Quit"),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Press ? or Esc to close",
            Style::default().fg(Color::DarkGray),
        )]),
    ];

    let help = Paragraph::new(help_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(" Help ")
                .title_style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
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
        AppMode::Settings => ("SETTINGS", Color::Green),
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
                (format!("reindexing {pct}%"), Color::Yellow)
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

    let header = Paragraph::new(Line::from(vec![Span::raw(format!(
        "Docs: {} │ Chunks: {} │ Status: {}",
        app.doc_count, app.chunk_count, app.status
    ))]))
    .block(Block::default().borders(Borders::ALL).title(vec![
        Span::raw(" RAG-TUI "),
        connection_indicator,
        Span::raw(connection_text),
        Span::raw(" "),
    ]));

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

    let results =
        List::new(items).block(Block::default().borders(Borders::ALL).title(results_title));

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
        "Enter=Search  j/k=Nav  [/]=top_k({})  Shift+S=Settings  ?=Help  C-c=Quit",
        app.top_k
    ))
    .style(Style::default().fg(Color::DarkGray));

    frame.render_widget(bindings, area);
}

fn draw_keybindings_detail(frame: &mut Frame, area: Rect) {
    let bindings = Paragraph::new("Esc/q=Back  j/k=Scroll  ↑↓=Prev/Next result  y=Copy  C-c=Quit")
        .style(Style::default().fg(Color::DarkGray));

    frame.render_widget(bindings, area);
}

fn draw_keybindings_settings(frame: &mut Frame, app: &App, area: Rect) {
    let bindings = if app.is_dropdown_open() {
        "j/k=Nav  Enter=Select  Esc=Cancel"
    } else if app.settings.editing {
        "Enter=Confirm  Esc=Cancel  ←→=Cursor  Type to edit"
    } else if app.settings.has_changes() {
        "j/k=Nav  Enter=Edit  r=Reset  R=Reset All  C-S=Save  Esc=Back  [UNSAVED]"
    } else {
        "j/k=Nav  Enter=Edit  r=Reset  C-S=Save  Esc=Back"
    };

    let style = if app.settings.has_changes() {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    frame.render_widget(Paragraph::new(bindings).style(style), area);
}

/// Settings overlay showing editable configuration
fn draw_settings_overlay(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Skip rendering on very small terminals
    if area.width < 30 || area.height < 15 {
        return;
    }

    // Calculate centered area (70% width, 80% height) with minimum sizes
    let popup_width = (area.width * 70 / 100).clamp(30, 80);
    let popup_height = (area.height * 80 / 100).clamp(15, 30);
    let popup_x = (area.width.saturating_sub(popup_width)) / 2;
    let popup_y = (area.height.saturating_sub(popup_height)) / 2;

    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    // Clear the background
    let clear = Block::default().style(Style::default().bg(Color::Black));
    frame.render_widget(clear, popup_area);

    // Build settings list
    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(
                "Settings",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            if app.settings.has_restart_required() {
                Span::styled(
                    "⚠ Restart required for some changes",
                    Style::default().fg(Color::Yellow),
                )
            } else if app.settings.has_changes() {
                Span::styled("● Unsaved changes", Style::default().fg(Color::Yellow))
            } else {
                Span::styled("", Style::default())
            },
        ]),
        Line::from(""),
    ];

    // Show model fetch status (for future dropdown functionality)
    match &app.model_fetch_state {
        ModelFetchState::Idle => {}
        ModelFetchState::Loading => {
            lines.push(Line::from(vec![
                Span::styled("⟳ ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    "Loading available models from Ollama...",
                    Style::default().fg(Color::Yellow),
                ),
            ]));
            lines.push(Line::from(""));
        }
        ModelFetchState::Loaded => {
            let count = app.available_models.len();
            lines.push(Line::from(vec![
                Span::styled("✓ ", Style::default().fg(Color::Green)),
                Span::styled(
                    format!("{count} models available"),
                    Style::default().fg(Color::Green),
                ),
            ]));
            lines.push(Line::from(""));
        }
        ModelFetchState::Failed(err) => {
            lines.push(Line::from(vec![
                Span::styled("✗ ", Style::default().fg(Color::Red)),
                Span::styled(
                    format!("Model fetch failed: {err}"),
                    Style::default().fg(Color::Red),
                ),
            ]));
            lines.push(Line::from(""));
        }
    }

    // Show settings message if any
    if let Some((ref msg, is_error)) = app.settings_message {
        let color = if is_error { Color::Red } else { Color::Green };
        lines.push(Line::from(Span::styled(
            msg.as_str(),
            Style::default().fg(color),
        )));
        lines.push(Line::from(""));
    }

    // Setting items
    for (i, setting) in app.settings.items.iter().enumerate() {
        let is_selected = i == app.settings.selected;
        let is_modified = setting.is_modified();

        // Marker and name
        let marker = if is_selected { "▶ " } else { "  " };
        let name_style = if is_selected {
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        // Value display - using owned strings for Span to avoid lifetime issues
        let value_display: Vec<Span> = if app.settings.editing && is_selected {
            // Show edit buffer with cursor (edit_buffer is Vec<char>)
            let before_cursor: String = app
                .settings
                .edit_buffer
                .iter()
                .take(app.settings.cursor_pos)
                .collect();
            let at_cursor: String = app
                .settings
                .edit_buffer
                .iter()
                .skip(app.settings.cursor_pos)
                .take(1)
                .collect();
            let after_cursor: String = app
                .settings
                .edit_buffer
                .iter()
                .skip(app.settings.cursor_pos + 1)
                .collect();

            let cursor_char = if at_cursor.is_empty() {
                " ".to_string()
            } else {
                at_cursor
            };

            vec![
                Span::raw(before_cursor),
                Span::styled(
                    cursor_char,
                    Style::default().bg(Color::White).fg(Color::Black),
                ),
                Span::raw(after_cursor),
            ]
        } else {
            // Show current value with validation-aware coloring
            let (value_color, validation_indicator) = match &setting.validation_state {
                ValidationState::None => {
                    let color = if is_modified {
                        Color::Yellow
                    } else {
                        Color::Green
                    };
                    (color, Span::raw(""))
                }
                ValidationState::Valid => {
                    let color = if is_modified {
                        Color::Yellow
                    } else {
                        Color::Green
                    };
                    (color, Span::styled(" ✓", Style::default().fg(Color::Green)))
                }
                ValidationState::Invalid(_) => (
                    Color::Red,
                    Span::styled(" ✗", Style::default().fg(Color::Red)),
                ),
                ValidationState::Warning(_) => (
                    Color::Yellow,
                    Span::styled(" ⚠", Style::default().fg(Color::Yellow)),
                ),
            };

            let modified_marker = if is_modified { " *" } else { "" };

            // Show options hint if available
            let options_hint = if setting.options.is_some() && is_selected {
                " [Tab to cycle]"
            } else {
                ""
            };

            vec![
                Span::styled(setting.value.clone(), Style::default().fg(value_color)),
                validation_indicator,
                Span::styled(modified_marker, Style::default().fg(Color::Yellow)),
                Span::styled(options_hint, Style::default().fg(Color::Gray)),
            ]
        };

        // Restart indicator
        let restart_indicator = if setting.requires_restart && is_modified {
            Span::styled(" (restart)", Style::default().fg(Color::Red))
        } else {
            Span::raw("")
        };

        let mut spans = vec![
            Span::raw(marker),
            Span::styled(&setting.display_name, name_style),
            Span::raw(": "),
        ];
        spans.extend(value_display);
        spans.push(restart_indicator);

        let line_style = if is_selected {
            Style::default().bg(Color::DarkGray)
        } else {
            Style::default()
        };

        lines.push(Line::from(spans).style(line_style));

        // Show description for selected item
        if is_selected {
            lines.push(Line::from(vec![
                Span::raw("     "),
                Span::styled(
                    &setting.description,
                    Style::default()
                        .fg(Color::Gray)
                        .add_modifier(Modifier::ITALIC),
                ),
            ]));

            // Show validation message if present
            match &setting.validation_state {
                ValidationState::Invalid(msg) => {
                    lines.push(Line::from(vec![
                        Span::raw("     "),
                        Span::styled(format!("✗ {msg}"), Style::default().fg(Color::Red)),
                    ]));
                }
                ValidationState::Warning(msg) => {
                    lines.push(Line::from(vec![
                        Span::raw("     "),
                        Span::styled(format!("⚠ {msg}"), Style::default().fg(Color::Yellow)),
                    ]));
                }
                _ => {}
            }
        }
    }

    // Footer with .env path
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("Config file: ", Style::default().fg(Color::Gray)),
        Span::styled(
            app.settings.env_path().display().to_string(),
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::ITALIC),
        ),
    ]));

    let settings = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Green))
                .title(" Settings ")
                .title_style(
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(settings, popup_area);
}

/// Dropdown overlay for model selection
fn draw_dropdown_overlay(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Get dropdown info
    let active = match &app.active_dropdown {
        Some(dropdown) => dropdown,
        None => return,
    };

    // Determine dropdown title and items based on type
    let (title, items): (&str, Vec<&str>) = match active {
        ActiveDropdown::EmbeddingModel => {
            let models: Vec<&str> = app
                .available_models
                .iter()
                .map(|m| m.name.as_str())
                .collect();
            ("Embedding Model", models)
        }
        ActiveDropdown::RerankerModel => {
            let models: Vec<&str> = app
                .available_models
                .iter()
                .map(|m| m.name.as_str())
                .collect();
            ("Reranker Model", models)
        }
        ActiveDropdown::Theme => {
            // Source theme options from settings data (not hard-coded)
            let theme_options: Vec<&str> = app
                .settings
                .items
                .get(app.dropdown_state.setting_index)
                .and_then(|s| s.options.as_ref())
                .map(|opts| opts.iter().map(|s| s.as_str()).collect())
                .unwrap_or_else(|| vec!["dark", "light"]); // Fallback if no options defined
            ("Theme", theme_options)
        }
    };

    // Handle empty items (e.g., no models fetched yet)
    let items = if items.is_empty() {
        match active {
            ActiveDropdown::EmbeddingModel | ActiveDropdown::RerankerModel => {
                vec!["(loading models...)"]
            }
            ActiveDropdown::Theme => vec!["dark", "light"], // Fallback
        }
    } else {
        items
    };

    // Calculate dropdown size
    let max_item_width = items.iter().map(|s| s.len()).max().unwrap_or(10);
    let dropdown_width = (max_item_width as u16 + 4).max(20).min(area.width - 4);
    let visible_items = 8.min(items.len()); // Show up to 8 items at once
    let dropdown_height = (visible_items as u16 + 2).min(area.height - 4);

    // Center the dropdown
    let dropdown_x = (area.width.saturating_sub(dropdown_width)) / 2;
    let dropdown_y = (area.height.saturating_sub(dropdown_height)) / 2;
    let dropdown_area = Rect::new(dropdown_x, dropdown_y, dropdown_width, dropdown_height);

    // Clear background
    let clear = Block::default().style(Style::default().bg(Color::Black));
    frame.render_widget(clear, dropdown_area);

    // Build dropdown items with scrolling support
    let inner_height = dropdown_height.saturating_sub(2) as usize;

    // Defensive clamping: ensure scroll_offset and selected are within bounds
    let scroll_offset = app
        .dropdown_state
        .scroll_offset
        .min(items.len().saturating_sub(inner_height));
    let selected_index = app
        .dropdown_state
        .selected
        .min(items.len().saturating_sub(1));

    let mut lines: Vec<Line> = vec![];

    // Calculate scroll indicator
    let has_scroll_up = scroll_offset > 0;
    let has_scroll_down = scroll_offset + inner_height < items.len();

    // Render visible items
    for (i, item) in items
        .iter()
        .enumerate()
        .skip(scroll_offset)
        .take(inner_height)
    {
        let is_selected = i == selected_index;
        let marker = if is_selected { "▶ " } else { "  " };
        let style = if is_selected {
            Style::default()
                .fg(Color::Cyan)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White)
        };

        lines.push(Line::from(vec![
            Span::raw(marker),
            Span::styled(*item, style),
        ]));
    }

    // Build title with scroll indicators
    let title_text = if has_scroll_up && has_scroll_down {
        format!(" {title} ↑↓ ")
    } else if has_scroll_up {
        format!(" {title} ↑ ")
    } else if has_scroll_down {
        format!(" {title} ↓ ")
    } else {
        format!(" {title} ")
    };

    let dropdown = Paragraph::new(lines).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan))
            .title(title_text)
            .title_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
    );

    frame.render_widget(dropdown, dropdown_area);
}

/// Split-pane layout: results list on left, detail view on right
fn draw_split_pane(frame: &mut Frame, app: &App, area: Rect) {
    let panes = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(35), // Results list (compact)
            Constraint::Percentage(65), // Detail view (larger)
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

    let results = List::new(items).block(Block::default().borders(Borders::ALL).title(" Results "));

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
            (format!("{:.3}", result.score), Color::Green)
        };

        let section_info = result
            .section
            .as_ref()
            .map(|s| format!("  §{s}"))
            .unwrap_or_default();

        let header = Line::from(vec![
            Span::styled(&provenance, Style::default().fg(Color::Cyan)),
            Span::raw("  Score: "),
            Span::styled(score_str, Style::default().fg(score_color)),
            Span::styled(section_info, Style::default().fg(Color::DarkGray)),
        ]);

        // Build score breakdown line if detailed scores are available
        let score_breakdown = build_score_breakdown_line(result);

        // Full text with scroll
        let mut lines: Vec<Line> = vec![header];

        // Add score breakdown if available
        if let Some(breakdown) = score_breakdown {
            lines.push(breakdown);
        }

        lines.push(Line::from(
            "─".repeat(area.width.saturating_sub(2) as usize),
        ));
        lines.extend(result.text.lines().map(|line| Line::from(line.to_string())));

        let text_lines: Vec<Line> = lines.into_iter().skip(app.detail_scroll).collect();

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

/// Build a line showing the score breakdown for a result
fn build_score_breakdown_line(result: &crate::api::SearchResult) -> Option<Line<'static>> {
    // Only show breakdown if we have reranker data
    let has_detailed_scores = result.embedding_score.is_some()
        || result.reranker_score.is_some()
        || result.yes_logprob.is_some();

    if !has_detailed_scores {
        return None;
    }

    let mut spans = vec![Span::styled(
        "  Scores: ",
        Style::default().fg(Color::DarkGray),
    )];

    // Embedding score
    if let Some(embed) = result.embedding_score {
        spans.push(Span::styled("embed=", Style::default().fg(Color::DarkGray)));
        spans.push(Span::styled(
            format!("{embed:.3}"),
            Style::default().fg(Color::Blue),
        ));
        spans.push(Span::raw("  "));
    }

    // Lexical score
    if let Some(lex) = result.lexical_score {
        spans.push(Span::styled("lex=", Style::default().fg(Color::DarkGray)));
        spans.push(Span::styled(
            format!("{lex:.3}"),
            Style::default().fg(Color::Blue),
        ));
        spans.push(Span::raw("  "));
    }

    // Initial (combined) score
    if let Some(initial) = result.initial_score {
        spans.push(Span::styled("init=", Style::default().fg(Color::DarkGray)));
        spans.push(Span::styled(
            format!("{initial:.3}"),
            Style::default().fg(Color::Yellow),
        ));
        spans.push(Span::raw("  "));
    }

    // Reranker score with logprobs if available
    if let Some(rerank) = result.reranker_score {
        spans.push(Span::styled(
            "rerank=",
            Style::default().fg(Color::DarkGray),
        ));
        spans.push(Span::styled(
            format!("{rerank:.3}"),
            Style::default().fg(Color::Magenta),
        ));

        // Show logprobs if available
        if let (Some(yes_lp), Some(no_lp)) = (result.yes_logprob, result.no_logprob) {
            spans.push(Span::styled(
                format!(" (yes:{yes_lp:.2} no:{no_lp:.2})"),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    Some(Line::from(spans))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::SearchResult;
    use crate::app::App;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

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
            embedding_score: None,
            lexical_score: None,
            initial_score: None,
            reranker_score: None,
            yes_logprob: None,
            no_logprob: None,
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

    #[test]
    fn test_render_settings_screen() {
        let mut app = make_test_app();
        app.mode = AppMode::Settings;

        let buffer = render_to_buffer(&app, 100, 40);

        // Should show Settings title
        assert!(buffer_contains(&buffer, "Settings"));
        // Should show setting names
        assert!(buffer_contains(&buffer, "Embedding Model"));
        assert!(buffer_contains(&buffer, "Documents Directory"));
        assert!(buffer_contains(&buffer, "Ollama URL"));
        // Should show keybindings for settings mode
        assert!(buffer_contains(&buffer, "Esc"));
    }

    #[test]
    fn test_render_settings_edit_mode() {
        let mut app = make_test_app();
        app.mode = AppMode::Settings;
        app.settings.start_edit();

        let buffer = render_to_buffer(&app, 100, 40);

        // Should show edit mode keybindings
        assert!(buffer_contains(&buffer, "Enter"));
        assert!(buffer_contains(&buffer, "Esc"));
    }

    #[test]
    fn test_render_settings_with_changes() {
        let mut app = make_test_app();
        app.mode = AppMode::Settings;
        // Modify a setting
        app.settings.items[0].value = "modified-model".to_string();

        let buffer = render_to_buffer(&app, 100, 40);

        // Should show modified indicator
        assert!(buffer_contains(&buffer, "modified"));
    }

    #[test]
    fn test_render_settings_validation_states() {
        let mut app = make_test_app();
        app.mode = AppMode::Settings;

        // Find Ollama URL setting and set invalid value
        if let Some(setting) = app
            .settings
            .items
            .iter_mut()
            .find(|s| s.env_var == "OLLAMA_URL")
        {
            setting.value = "not-a-url".to_string();
            setting.validate();
        }

        // Should not panic when rendering validation states
        let _buffer = render_to_buffer(&app, 100, 40);
    }

    #[test]
    fn test_render_settings_dropdown_open() {
        let mut app = make_test_app();
        app.mode = AppMode::Settings;
        // Find theme setting (has options) and select it
        if let Some(idx) = app
            .settings
            .items
            .iter()
            .position(|s| s.env_var == "RAG_TUI_THEME")
        {
            app.settings.selected = idx;
        }
        app.open_dropdown();

        let buffer = render_to_buffer(&app, 100, 40);

        // Should show dropdown options
        assert!(buffer_contains(&buffer, "dark") || buffer_contains(&buffer, "light"));
    }

    #[test]
    fn test_render_settings_small_terminal() {
        let mut app = make_test_app();
        app.mode = AppMode::Settings;

        // Should not panic on small terminal
        let _buffer = render_to_buffer(&app, 60, 25);
    }

    #[test]
    fn test_render_settings_with_message() {
        let mut app = make_test_app();
        app.mode = AppMode::Settings;
        app.settings_message = Some(("Settings saved!".to_string(), false));

        let buffer = render_to_buffer(&app, 100, 40);

        // Should show the message
        assert!(buffer_contains(&buffer, "Settings saved"));
    }
}
