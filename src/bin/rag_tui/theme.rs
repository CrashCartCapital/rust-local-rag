//! Theme configuration for the RAG TUI
//!
//! Centralizes color definitions for consistent UI styling.

use ratatui::style::Color;

/// Theme configuration for UI colors
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Theme {
    // Background
    pub bg_overlay: Color,

    // Primary accent (titles, highlights)
    pub accent: Color,

    // Status colors
    pub status_connected: Color,
    pub status_disconnected: Color,
    pub status_ready: Color,
    pub status_warning: Color,
    pub status_error: Color,

    // Score colors
    pub score_high: Color,
    pub score_medium: Color,
    pub score_low: Color,

    // Text colors
    pub text_muted: Color,
    pub text_section: Color,

    // Selection
    pub selection_bg: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// Dark theme (default)
    pub fn dark() -> Self {
        Self {
            bg_overlay: Color::Black,
            accent: Color::Cyan,
            status_connected: Color::Green,
            status_disconnected: Color::Red,
            status_ready: Color::Green,
            status_warning: Color::Yellow,
            status_error: Color::Red,
            score_high: Color::Green,
            score_medium: Color::Yellow,
            score_low: Color::DarkGray,
            text_muted: Color::DarkGray,
            text_section: Color::Yellow,
            selection_bg: Color::DarkGray,
        }
    }

    /// Light theme variant
    #[allow(dead_code)]
    pub fn light() -> Self {
        Self {
            bg_overlay: Color::White,
            accent: Color::Blue,
            status_connected: Color::Green,
            status_disconnected: Color::Red,
            status_ready: Color::Green,
            status_warning: Color::Rgb(255, 165, 0), // Orange
            status_error: Color::Red,
            score_high: Color::Green,
            score_medium: Color::Rgb(255, 165, 0), // Orange
            score_low: Color::Gray,
            text_muted: Color::Gray,
            text_section: Color::Blue,
            selection_bg: Color::LightBlue,
        }
    }

    /// High contrast theme for accessibility
    #[allow(dead_code)]
    pub fn high_contrast() -> Self {
        Self {
            bg_overlay: Color::Black,
            accent: Color::White,
            status_connected: Color::LightGreen,
            status_disconnected: Color::LightRed,
            status_ready: Color::LightGreen,
            status_warning: Color::LightYellow,
            status_error: Color::LightRed,
            score_high: Color::LightGreen,
            score_medium: Color::LightYellow,
            score_low: Color::White,
            text_muted: Color::White,
            text_section: Color::LightYellow,
            selection_bg: Color::White,
        }
    }

    /// Create theme from name (for config)
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "light" => Self::light(),
            "high_contrast" | "high-contrast" | "highcontrast" => Self::high_contrast(),
            _ => Self::dark(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_dark() {
        let default = Theme::default();
        let dark = Theme::dark();
        assert_eq!(default.accent, dark.accent);
        assert_eq!(default.bg_overlay, dark.bg_overlay);
    }

    #[test]
    fn test_from_name() {
        let dark = Theme::from_name("dark");
        assert_eq!(dark.accent, Color::Cyan);

        let light = Theme::from_name("light");
        assert_eq!(light.accent, Color::Blue);

        let hc = Theme::from_name("high-contrast");
        assert_eq!(hc.accent, Color::White);

        // Unknown defaults to dark
        let unknown = Theme::from_name("unknown");
        assert_eq!(unknown.accent, Color::Cyan);
    }

    #[test]
    fn test_theme_variants_have_distinct_colors() {
        let dark = Theme::dark();
        let light = Theme::light();
        let hc = Theme::high_contrast();

        // Each theme should have different accents
        assert_ne!(dark.accent, light.accent);
        assert_ne!(light.accent, hc.accent);
    }
}
