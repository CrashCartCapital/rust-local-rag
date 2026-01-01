use std::path::{Path, PathBuf};

pub const INDEX_VERSION: u32 = 2;

/// Sanitizes model name for safe use as a filename.
/// Replaces path separators, special characters, and handles edge cases.
pub fn sanitize_model_name(model_name: &str) -> String {
    let trimmed = model_name.trim();

    if trimmed.is_empty() {
        return "default".to_string();
    }

    let sanitized: String = trimmed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if sanitized.is_empty() || sanitized.chars().all(|c| c == '_' || c == '.') {
        "default".to_string()
    } else {
        sanitized
    }
}

/// Generates the index file path for a specific model.
/// Uses sanitized model name to ensure filesystem safety.
pub fn index_path(data_dir: impl AsRef<Path>, model_name: &str) -> PathBuf {
    let sanitized = sanitize_model_name(model_name);
    data_dir.as_ref().join(format!("chunks_{sanitized}.json"))
}

/// Generates the legacy index file path (for migration support).
pub fn legacy_path(data_dir: impl AsRef<Path>) -> PathBuf {
    data_dir.as_ref().join("chunks.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_sanitize_model_name_basic() {
        assert_eq!(sanitize_model_name("nomic-embed-text"), "nomic-embed-text");
        assert_eq!(sanitize_model_name("all-MiniLM-L6-v2"), "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_sanitize_model_name_with_slashes() {
        assert_eq!(
            sanitize_model_name("sentence-transformers/all-MiniLM-L6-v2"),
            "sentence-transformers_all-MiniLM-L6-v2"
        );
        assert_eq!(
            sanitize_model_name("openai/text-embedding-3-large"),
            "openai_text-embedding-3-large"
        );
    }

    #[test]
    fn test_sanitize_model_name_path_traversal() {
        assert_eq!(sanitize_model_name("../etc/passwd"), ".._etc_passwd");
        assert_eq!(
            sanitize_model_name("..\\windows\\system32"),
            ".._windows_system32"
        );
        assert_eq!(sanitize_model_name("foo/../bar"), "foo_.._bar");
    }

    #[test]
    fn test_sanitize_model_name_special_chars() {
        assert_eq!(sanitize_model_name("model:v1"), "model_v1");
        assert_eq!(sanitize_model_name("model*test?"), "model_test_");
        assert_eq!(sanitize_model_name("model<>|"), "model___");
    }

    #[test]
    fn test_sanitize_model_name_empty_and_whitespace() {
        assert_eq!(sanitize_model_name(""), "default");
        assert_eq!(sanitize_model_name("   "), "default");
    }

    #[test]
    fn test_index_path_basic() {
        let path = index_path("/data", "nomic-embed-text");
        assert_eq!(path, PathBuf::from("/data/chunks_nomic-embed-text.json"));
    }

    #[test]
    fn test_index_path_with_slashes_in_model() {
        let path = index_path("/data", "sentence-transformers/all-MiniLM");
        assert_eq!(
            path,
            PathBuf::from("/data/chunks_sentence-transformers_all-MiniLM.json")
        );
    }

    #[test]
    fn test_index_path_stays_in_directory() {
        let path = index_path("/data", "../etc/passwd");
        assert!(path.starts_with("/data/"));
        assert_eq!(path, PathBuf::from("/data/chunks_.._etc_passwd.json"));
    }

    #[test]
    fn test_legacy_path() {
        let path = legacy_path("/data");
        assert_eq!(path, PathBuf::from("/data/chunks.json"));
    }
}
