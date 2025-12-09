//! Settings management for the RAG TUI
//!
//! Handles editable configuration that can be saved to .env file.

use reqwest::Url;
use std::fs;
use std::path::PathBuf;

/// Expand tilde (~) to home directory in path strings
/// Supports both Unix (HOME) and Windows (USERPROFILE) environments
fn expand_tilde(path: &str) -> String {
    // Get home directory from HOME (Unix) or USERPROFILE (Windows)
    let home_dir = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));

    if let Some(home) = home_dir {
        if path.starts_with("~/") {
            return format!("{}{}", home.to_string_lossy(), &path[1..]);
        } else if path == "~" {
            return home.to_string_lossy().to_string();
        }
    }
    path.to_string()
}

/// Validation state for a setting
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ValidationState {
    /// Not validated or validation not applicable
    #[default]
    None,
    /// Value is valid
    Valid,
    /// Value is invalid with reason
    Invalid(String),
    /// Value might be valid but couldn't be confirmed (e.g., path doesn't exist yet)
    Warning(String),
}

/// A single editable setting
#[derive(Debug, Clone)]
pub struct Setting {
    /// Environment variable name
    pub env_var: String,
    /// Display name in UI
    pub display_name: String,
    /// Current value
    pub value: String,
    /// Original value (for detecting changes)
    pub original_value: String,
    /// Description/help text
    pub description: String,
    /// Whether changing this requires server restart
    pub requires_restart: bool,
    /// Optional list of valid values (for dropdowns)
    pub options: Option<Vec<String>>,
    /// Validation state
    pub validation_state: ValidationState,
}

impl Setting {
    pub fn new(
        env_var: &str,
        display_name: &str,
        value: &str,
        description: &str,
        requires_restart: bool,
    ) -> Self {
        let mut setting = Self {
            env_var: env_var.to_string(),
            display_name: display_name.to_string(),
            value: value.to_string(),
            original_value: value.to_string(),
            description: description.to_string(),
            requires_restart,
            options: None,
            validation_state: ValidationState::None,
        };
        setting.validate();
        setting
    }

    pub fn with_options(mut self, options: Vec<&str>) -> Self {
        self.options = Some(options.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Check if value has been modified
    pub fn is_modified(&self) -> bool {
        self.value != self.original_value
    }

    /// Reset to original value
    pub fn reset(&mut self) {
        self.value = self.original_value.clone();
        self.validate();
    }

    /// Mark current value as saved (update original)
    pub fn mark_saved(&mut self) {
        self.original_value = self.value.clone();
    }

    /// Validate the current value based on setting type
    pub fn validate(&mut self) {
        self.validation_state = match self.env_var.as_str() {
            // URL validation for Ollama URL
            "OLLAMA_URL" => {
                let value = self.value.trim();
                if value.is_empty() {
                    ValidationState::Invalid("URL cannot be empty".to_string())
                } else if !value.starts_with("http://") && !value.starts_with("https://") {
                    ValidationState::Invalid("URL must start with http:// or https://".to_string())
                } else if Url::parse(value).is_err() {
                    ValidationState::Invalid("Invalid URL format".to_string())
                } else {
                    ValidationState::Valid
                }
            }

            // Path validation for directory fields
            "DATA_DIR" | "DOCUMENTS_DIR" => {
                let value = self.value.trim();
                if value.is_empty() {
                    ValidationState::Invalid("Path cannot be empty".to_string())
                } else {
                    // Expand tilde to home directory
                    let expanded = expand_tilde(value);
                    let path = PathBuf::from(&expanded);

                    if path.exists() {
                        if path.is_dir() {
                            ValidationState::Valid
                        } else {
                            ValidationState::Invalid(
                                "Path exists but is not a directory".to_string(),
                            )
                        }
                    } else {
                        // Check if tilde was used but expansion failed
                        if value.starts_with('~') && expanded == value {
                            ValidationState::Warning(
                                "Tilde (~) may not expand correctly; use absolute path".to_string(),
                            )
                        } else {
                            ValidationState::Warning(
                                "Directory does not exist (will be created)".to_string(),
                            )
                        }
                    }
                }
            }

            // Weight validation for score weight settings (0.0 to 1.0, must be finite)
            "RAG_EMBEDDING_WEIGHT"
            | "RAG_LEXICAL_WEIGHT"
            | "RAG_RERANKER_WEIGHT"
            | "RAG_INITIAL_SCORE_WEIGHT" => {
                let value = self.value.trim();
                if value.is_empty() {
                    ValidationState::Invalid("Weight cannot be empty".to_string())
                } else {
                    match value.parse::<f32>() {
                        Ok(w) if !w.is_finite() => ValidationState::Invalid(
                            "Weight must be a finite number (not NaN or Inf)".to_string(),
                        ),
                        Ok(w) if (0.0..=1.0).contains(&w) => ValidationState::Valid,
                        Ok(_) => ValidationState::Invalid(
                            "Weight must be between 0.0 and 1.0".to_string(),
                        ),
                        Err(_) => ValidationState::Invalid("Invalid number format".to_string()),
                    }
                }
            }

            // No validation for other settings
            _ => ValidationState::None,
        };
    }
}

/// All editable settings
#[derive(Debug, Clone)]
pub struct Settings {
    /// Ordered list of settings
    pub items: Vec<Setting>,
    /// Currently selected setting index
    pub selected: usize,
    /// Whether in edit mode for current setting
    pub editing: bool,
    /// Edit buffer as Vec<char> for safe Unicode handling
    /// (String::insert/remove need byte indices, but we track char indices)
    pub edit_buffer: Vec<char>,
    /// Cursor position in edit buffer (character index, not byte)
    pub cursor_pos: usize,
    /// Path to .env file
    env_path: PathBuf,
}

impl Default for Settings {
    fn default() -> Self {
        Self::new()
    }
}

impl Settings {
    pub fn new() -> Self {
        // Determine .env path (project root)
        let env_path = std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".env");

        // Load current values from environment
        let items = vec![
            Setting::new(
                "OLLAMA_EMBEDDING_MODEL",
                "Embedding Model",
                &std::env::var("OLLAMA_EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "nomic-embed-text".to_string()),
                "Model used for generating embeddings (requires reindex)",
                true,
            ),
            Setting::new(
                "OLLAMA_RERANK_MODEL",
                "Reranker Model",
                &std::env::var("OLLAMA_RERANK_MODEL").unwrap_or_else(|_| "phi4-mini".to_string()),
                "Model used for reranking search results",
                true,
            ),
            Setting::new(
                "DOCUMENTS_DIR",
                "Documents Directory",
                &std::env::var("DOCUMENTS_DIR").unwrap_or_else(|_| "./documents".to_string()),
                "Directory containing PDF documents to index",
                true,
            ),
            Setting::new(
                "DATA_DIR",
                "Data Directory",
                &std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string()),
                "Directory for storing embeddings and index data",
                true,
            ),
            Setting::new(
                "OLLAMA_URL",
                "Ollama URL",
                &std::env::var("OLLAMA_URL")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string()),
                "URL of the Ollama server",
                true,
            ),
            Setting::new(
                "RAG_TUI_THEME",
                "Theme",
                &std::env::var("RAG_TUI_THEME").unwrap_or_else(|_| "dark".to_string()),
                "UI color theme",
                false, // TUI-only, doesn't require server restart
            )
            .with_options(vec!["dark", "light", "high-contrast"]),
            Setting::new(
                "RAG_TUI_TOP_K",
                "Default Top-K",
                &std::env::var("RAG_TUI_TOP_K").unwrap_or_else(|_| "10".to_string()),
                "Default number of search results",
                false,
            ),
            Setting::new(
                "RAG_TUI_POLL_INTERVAL_S",
                "Poll Interval (sec)",
                &std::env::var("RAG_TUI_POLL_INTERVAL_S").unwrap_or_else(|_| "2".to_string()),
                "How often to poll server for stats updates",
                false,
            ),
            // Score weight settings (server-side, affects search scoring)
            Setting::new(
                "RAG_EMBEDDING_WEIGHT",
                "Embedding Weight",
                &std::env::var("RAG_EMBEDDING_WEIGHT").unwrap_or_else(|_| "0.7".to_string()),
                "Weight for embedding similarity in initial score (0.0-1.0)",
                true,
            ),
            Setting::new(
                "RAG_LEXICAL_WEIGHT",
                "Lexical Weight",
                &std::env::var("RAG_LEXICAL_WEIGHT").unwrap_or_else(|_| "0.3".to_string()),
                "Weight for lexical/BM25 score in initial score (0.0-1.0)",
                true,
            ),
            Setting::new(
                "RAG_RERANKER_WEIGHT",
                "Reranker Weight",
                &std::env::var("RAG_RERANKER_WEIGHT").unwrap_or_else(|_| "0.7".to_string()),
                "Weight for reranker score in final score (0.0-1.0)",
                true,
            ),
            Setting::new(
                "RAG_INITIAL_SCORE_WEIGHT",
                "Initial Score Weight",
                &std::env::var("RAG_INITIAL_SCORE_WEIGHT").unwrap_or_else(|_| "0.3".to_string()),
                "Weight for initial combined score in final score (0.0-1.0)",
                true,
            ),
        ];

        Self {
            items,
            selected: 0,
            editing: false,
            edit_buffer: Vec::new(),
            cursor_pos: 0,
            env_path,
        }
    }

    /// Navigate to previous setting
    pub fn prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    /// Navigate to next setting
    pub fn next(&mut self) {
        if self.selected < self.items.len() - 1 {
            self.selected += 1;
        }
    }

    /// Start editing current setting
    pub fn start_edit(&mut self) {
        if let Some(setting) = self.items.get(self.selected) {
            self.edit_buffer = setting.value.chars().collect();
            self.cursor_pos = self.edit_buffer.len();
            self.editing = true;
        }
    }

    /// Cancel editing
    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_buffer.clear();
        self.cursor_pos = 0;
    }

    /// Confirm edit
    pub fn confirm_edit(&mut self) {
        if self.editing {
            if let Some(setting) = self.items.get_mut(self.selected) {
                setting.value = self.edit_buffer.iter().collect();
                setting.validate();
            }
            self.editing = false;
            self.edit_buffer.clear();
            self.cursor_pos = 0;
        }
    }

    /// Cycle through options (for dropdown-style settings)
    pub fn cycle_option(&mut self, forward: bool) {
        if let Some(setting) = self.items.get_mut(self.selected) {
            if let Some(ref options) = setting.options {
                if let Some(current_idx) = options.iter().position(|o| o == &setting.value) {
                    let new_idx = if forward {
                        (current_idx + 1) % options.len()
                    } else if current_idx == 0 {
                        options.len() - 1
                    } else {
                        current_idx - 1
                    };
                    setting.value = options[new_idx].clone();
                } else {
                    // Current value not in options, set to first option
                    setting.value = options[0].clone();
                }
                setting.validate();
            }
        }
    }

    /// Input character in edit buffer (Unicode-safe with Vec<char>)
    pub fn input_char(&mut self, c: char) {
        self.edit_buffer.insert(self.cursor_pos, c);
        self.cursor_pos += 1;
    }

    /// Backspace in edit buffer (Unicode-safe with Vec<char>)
    pub fn backspace(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
            self.edit_buffer.remove(self.cursor_pos);
        }
    }

    /// Delete character at cursor (Unicode-safe with Vec<char>)
    pub fn delete(&mut self) {
        if self.cursor_pos < self.edit_buffer.len() {
            self.edit_buffer.remove(self.cursor_pos);
        }
    }

    /// Move cursor left
    pub fn cursor_left(&mut self) {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
        }
    }

    /// Move cursor right
    pub fn cursor_right(&mut self) {
        if self.cursor_pos < self.edit_buffer.len() {
            self.cursor_pos += 1;
        }
    }

    /// Move cursor to start
    pub fn cursor_home(&mut self) {
        self.cursor_pos = 0;
    }

    /// Move cursor to end
    pub fn cursor_end(&mut self) {
        self.cursor_pos = self.edit_buffer.len();
    }

    /// Check if any settings have been modified
    pub fn has_changes(&self) -> bool {
        self.items.iter().any(|s| s.is_modified())
    }

    /// Check if any modified settings require restart
    pub fn has_restart_required(&self) -> bool {
        self.items
            .iter()
            .any(|s| s.is_modified() && s.requires_restart)
    }

    /// Get current setting
    pub fn current(&self) -> Option<&Setting> {
        self.items.get(self.selected)
    }

    /// Reset all settings to original values
    pub fn reset_all(&mut self) {
        for setting in &mut self.items {
            setting.reset();
        }
    }

    /// Reset current setting to original value
    pub fn reset_current(&mut self) {
        if let Some(setting) = self.items.get_mut(self.selected) {
            setting.reset();
        }
    }

    /// Save settings to .env file, preserving unknown variables and comments
    pub fn save_to_env(&mut self) -> Result<bool, String> {
        // Track if any restart-required settings were modified BEFORE we mark them saved
        let had_restart_required = self.has_restart_required();

        // Read existing .env file if it exists, preserving structure
        let mut lines_to_write: Vec<String> = Vec::new();
        let mut keys_written: std::collections::HashSet<String> = std::collections::HashSet::new();

        if self.env_path.exists() {
            let content = fs::read_to_string(&self.env_path)
                .map_err(|e| format!("Failed to read .env: {e}"))?;

            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    // Preserve comments and blank lines as-is
                    lines_to_write.push(line.to_string());
                } else if let Some((key, _)) = trimmed.split_once('=') {
                    let key = key.trim();
                    // Check if this is one of our managed settings
                    if let Some(setting) = self.items.iter().find(|s| s.env_var == key) {
                        // Write our updated value
                        lines_to_write.push(format!("{}={}", key, setting.value));
                        keys_written.insert(key.to_string());
                    } else {
                        // Preserve unknown variables as-is
                        lines_to_write.push(line.to_string());
                    }
                } else {
                    // Preserve malformed lines
                    lines_to_write.push(line.to_string());
                }
            }
        }

        // Add any settings that weren't in the original file
        for setting in &self.items {
            if !keys_written.contains(&setting.env_var) {
                lines_to_write.push(format!("{}={}", setting.env_var, setting.value));
            }
        }

        // Write back with trailing newline
        let output = lines_to_write.join("\n") + "\n";

        fs::write(&self.env_path, output).map_err(|e| format!("Failed to write .env: {e}"))?;

        // Mark all as saved
        for setting in &mut self.items {
            setting.mark_saved();
        }

        // Return whether restart-required settings were modified
        Ok(had_restart_required)
    }

    /// Get path to .env file
    pub fn env_path(&self) -> &PathBuf {
        &self.env_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setting_modification() {
        let mut setting = Setting::new("TEST", "Test", "original", "desc", false);
        assert!(!setting.is_modified());

        setting.value = "changed".to_string();
        assert!(setting.is_modified());

        setting.reset();
        assert!(!setting.is_modified());
        assert_eq!(setting.value, "original");
    }

    #[test]
    fn test_settings_navigation() {
        let mut settings = Settings::new();
        assert_eq!(settings.selected, 0);

        settings.next();
        assert_eq!(settings.selected, 1);

        settings.prev();
        assert_eq!(settings.selected, 0);

        // Can't go below 0
        settings.prev();
        assert_eq!(settings.selected, 0);
    }

    #[test]
    fn test_edit_mode() {
        let mut settings = Settings::new();
        assert!(!settings.editing);

        settings.start_edit();
        assert!(settings.editing);
        assert!(!settings.edit_buffer.is_empty());

        settings.input_char('x');
        assert_eq!(settings.edit_buffer.last(), Some(&'x'));

        settings.cancel_edit();
        assert!(!settings.editing);
        assert!(settings.edit_buffer.is_empty());
    }

    #[test]
    fn test_option_cycling() {
        let mut settings = Settings::new();
        // Find theme setting (has options)
        let theme_idx = settings
            .items
            .iter()
            .position(|s| s.env_var == "RAG_TUI_THEME")
            .unwrap();
        settings.selected = theme_idx;

        let original = settings.items[theme_idx].value.clone();
        settings.cycle_option(true);
        assert_ne!(settings.items[theme_idx].value, original);
    }

    #[test]
    fn test_has_changes() {
        let mut settings = Settings::new();
        assert!(!settings.has_changes());

        settings.items[0].value = "changed".to_string();
        assert!(settings.has_changes());
    }
}
