//! UI and configuration constants for the RAG TUI
//!
//! Centralizes magic numbers for maintainability and testability.

#![allow(dead_code)] // Some constants used in later tiers

use std::time::Duration;

// ============================================================================
// UI Layout Constants
// ============================================================================

/// Maximum characters to show in result text preview
pub const PREVIEW_CHARS: usize = 60;

/// Maximum characters for embedding model name display
pub const MODEL_NAME_MAX_CHARS: usize = 15;

/// Maximum characters for reranker model name display
pub const RERANKER_NAME_MAX_CHARS: usize = 12;

/// Maximum characters for error message display in status bar
pub const ERROR_TRUNCATE_CHARS: usize = 30;

/// Maximum characters for document name in compact view
pub const DOC_NAME_MAX_CHARS: usize = 20;

// ============================================================================
// Timing Constants
// ============================================================================

/// Timeout for API calls (health, stats) to keep UI responsive
pub const API_TIMEOUT: Duration = Duration::from_secs(5);

/// Default polling interval for stats updates
pub const DEFAULT_POLL_INTERVAL_SECS: u64 = 2;

/// Interval for health check polling
pub const HEALTH_CHECK_INTERVAL_SECS: u64 = 5;

/// Interval for job status polling during reindexing
pub const JOB_POLL_INTERVAL_SECS: u64 = 1;

/// Timeout for search requests (longer due to reranking)
pub const SEARCH_TIMEOUT: Duration = Duration::from_secs(120);

/// Terminal event polling interval
pub const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(100);

// ============================================================================
// Search & Results Constants
// ============================================================================

/// Default number of search results to return
pub const DEFAULT_TOP_K: usize = 10;

/// Minimum allowed top_k value
pub const MIN_TOP_K: usize = 1;

/// Maximum allowed top_k value
pub const MAX_TOP_K: usize = 100;

/// Step size for top_k adjustment
pub const TOP_K_STEP: usize = 5;

// ============================================================================
// Score Thresholds for Color Coding
// ============================================================================

/// Score threshold for green (high relevance)
pub const SCORE_THRESHOLD_HIGH: f32 = 0.7;

/// Score threshold for yellow (medium relevance)
pub const SCORE_THRESHOLD_MEDIUM: f32 = 0.4;

// ============================================================================
// Channel Constants
// ============================================================================

/// Capacity for bounded search result channel
pub const SEARCH_CHANNEL_CAPACITY: usize = 16;

/// Debounce delay for search-as-you-type (milliseconds)
pub const INPUT_DEBOUNCE_MS: u64 = 300;

// ============================================================================
// Detail View Constants
// ============================================================================

/// Number of lines to scroll per j/k keypress in detail view
pub const DETAIL_SCROLL_LINES: usize = 1;

/// Approximate max lines for detail scroll bounds
pub const DETAIL_MAX_SCROLL_ESTIMATE: usize = 100;
