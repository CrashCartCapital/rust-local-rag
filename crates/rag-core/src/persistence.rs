use crate::error::PersistenceError;
use crate::types::DocumentChunk;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const INDEX_VERSION: u32 = 2;

// ============================================================================
// EngineState: Serializable state for persistence
// ============================================================================

/// Complete engine state for persistence.
///
/// This struct contains all data needed to fully restore a RagEngine from disk.
/// It includes identity/invalidation keys per PRD D4.2 requirements.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct EngineState {
    /// Schema version for migration detection
    pub schema_version: u32,

    /// Embedding model identifier (invalidation key)
    pub embedding_model_id: String,

    /// Embedding dimension (invalidation key, derived from model)
    #[cfg_attr(feature = "persistence", serde(default))]
    pub embedding_dim: usize,

    /// All indexed chunks with their embeddings
    pub chunks: HashMap<String, DocumentChunk>,

    /// Document hashes for change detection
    #[cfg_attr(feature = "persistence", serde(default))]
    pub document_hashes: HashMap<String, String>,

    /// Whether a reindex is needed
    #[cfg_attr(feature = "persistence", serde(default))]
    pub needs_reindex: bool,
}

impl EngineState {
    /// Create a new EngineState with the given parameters.
    pub fn new(model_id: impl Into<String>, embedding_dim: usize) -> Self {
        Self {
            schema_version: INDEX_VERSION,
            embedding_model_id: model_id.into(),
            embedding_dim,
            chunks: HashMap::new(),
            document_hashes: HashMap::new(),
            needs_reindex: false,
        }
    }
}

// ============================================================================
// PersistenceBackend Trait
// ============================================================================

/// Trait for pluggable persistence backends.
///
/// Implementations must provide atomic save/load operations and schema versioning.
/// The default implementation is `JsonFileBackend`.
///
/// # Thread Safety
///
/// This trait requires `Send + Sync` to support concurrent access patterns.
/// Implementations should ensure:
///
/// - **Atomicity**: `save()` must not leave partial state on failure
/// - **Isolation**: Concurrent reads during write should see old or new state, never partial
///
/// For file-based backends, use the temp-file-then-rename pattern.
/// For database backends, use transactions.
///
/// # Blocking I/O
///
/// The methods in this trait are synchronous and may perform blocking I/O.
/// When using with async code, wrap calls in `tokio::task::spawn_blocking`:
///
/// ```rust,ignore
/// let state = tokio::task::spawn_blocking(move || backend.load()).await??;
/// ```
///
/// # Invariants
///
/// - `save()` must be atomic (temp file + rename pattern)
/// - `load()` must return `None` if no persisted state exists
/// - `schema_version()` must match the version in saved state
pub trait PersistenceBackend: Send + Sync {
    /// Save engine state to persistent storage.
    ///
    /// Implementations should use atomic write semantics to prevent corruption.
    fn save(&self, state: &EngineState) -> Result<(), PersistenceError>;

    /// Load engine state from persistent storage.
    ///
    /// Returns `Ok(None)` if no persisted state exists (new engine).
    /// Returns `Err` for actual errors (I/O, corruption, etc.).
    fn load(&self) -> Result<Option<EngineState>, PersistenceError>;

    /// Get the schema version this backend expects.
    fn schema_version(&self) -> u32 {
        INDEX_VERSION
    }
}

// ============================================================================
// JsonFileBackend: Default file-based persistence
// ============================================================================

/// JSON file-based persistence backend.
///
/// Stores engine state as a JSON file with atomic write semantics
/// (temp file + rename). Supports model-partitioned storage and
/// legacy format migration.
#[cfg(feature = "persistence")]
pub struct JsonFileBackend {
    /// Base directory for data storage
    data_dir: PathBuf,
    /// Model name for partitioned storage
    model_name: String,
}

#[cfg(feature = "persistence")]
impl JsonFileBackend {
    /// Create a new JsonFileBackend.
    ///
    /// # Arguments
    ///
    /// * `data_dir` - Directory where index files are stored
    /// * `model_name` - Embedding model name (used for file naming)
    pub fn new(data_dir: impl Into<PathBuf>, model_name: impl Into<String>) -> Self {
        Self {
            data_dir: data_dir.into(),
            model_name: model_name.into(),
        }
    }

    /// Get the path to the model-specific index file.
    pub fn index_path(&self) -> PathBuf {
        index_path(&self.data_dir, &self.model_name)
    }

    /// Get the path to the legacy index file.
    pub fn legacy_path(&self) -> PathBuf {
        legacy_path(&self.data_dir)
    }

    /// Check if legacy format exists and matches current model.
    fn try_load_legacy(&self) -> Result<Option<EngineState>, PersistenceError> {
        let legacy = self.legacy_path();
        if !legacy.exists() {
            return Ok(None);
        }

        let data = std::fs::read_to_string(&legacy)?;

        // Try to parse with model info
        #[derive(serde::Deserialize)]
        struct ModelOnly {
            model: String,
        }

        if let Ok(info) = serde_json::from_str::<ModelOnly>(&data) {
            if info.model == self.model_name {
                // Model matches, try full parse
                if let Ok(state) = serde_json::from_str::<LegacyState>(&data) {
                    return Ok(Some(state.into_engine_state(&self.model_name)));
                }
            }
        }

        Ok(None)
    }
}

/// Legacy persistence format for backward compatibility.
#[cfg(feature = "persistence")]
#[derive(serde::Deserialize)]
struct LegacyState {
    version: u32,
    #[allow(dead_code)]
    model: String,
    chunks: HashMap<String, DocumentChunk>,
    #[serde(default)]
    needs_reindex: bool,
    #[serde(default)]
    document_hashes: HashMap<String, String>,
}

#[cfg(feature = "persistence")]
impl LegacyState {
    fn into_engine_state(self, model_name: &str) -> EngineState {
        // Infer embedding dimension from chunks
        let embedding_dim = self
            .chunks
            .values()
            .next()
            .map(|c| c.embedding.len())
            .unwrap_or(0);

        EngineState {
            schema_version: self.version,
            embedding_model_id: model_name.to_string(),
            embedding_dim,
            chunks: self.chunks,
            document_hashes: self.document_hashes,
            needs_reindex: self.needs_reindex,
        }
    }
}

#[cfg(feature = "persistence")]
impl PersistenceBackend for JsonFileBackend {
    fn save(&self, state: &EngineState) -> Result<(), PersistenceError> {
        // Ensure directory exists
        if let Some(parent) = self.data_dir.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        if !self.data_dir.exists() {
            std::fs::create_dir_all(&self.data_dir)?;
        }

        let final_path = self.index_path();
        let temp_path = final_path.with_extension("json.tmp");

        // Serialize to JSON
        let data = serde_json::to_string_pretty(state)?;

        // Atomic write: temp file + rename
        std::fs::write(&temp_path, &data)?;
        std::fs::rename(&temp_path, &final_path)?;

        Ok(())
    }

    fn load(&self) -> Result<Option<EngineState>, PersistenceError> {
        let path = self.index_path();

        // Try model-specific path first
        if path.exists() {
            let data = std::fs::read_to_string(&path)?;
            match serde_json::from_str::<EngineState>(&data) {
                Ok(state) => return Ok(Some(state)),
                Err(_e) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        "Failed to parse index file at {:?}: {}. Starting fresh.",
                        path,
                        _e
                    );

                    // Try legacy format
                    if let Ok(legacy) = serde_json::from_str::<LegacyState>(&data) {
                        return Ok(Some(legacy.into_engine_state(&self.model_name)));
                    }
                    // Parse error - treat as no state
                    return Ok(None);
                }
            }
        }

        // Try legacy path
        self.try_load_legacy()
    }

    fn schema_version(&self) -> u32 {
        INDEX_VERSION
    }
}

// ============================================================================
// Path Utilities
// ============================================================================

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

    #[test]
    fn test_engine_state_new() {
        let state = EngineState::new("test-model", 384);
        assert_eq!(state.schema_version, INDEX_VERSION);
        assert_eq!(state.embedding_model_id, "test-model");
        assert_eq!(state.embedding_dim, 384);
        assert!(state.chunks.is_empty());
        assert!(!state.needs_reindex);
    }

    #[cfg(feature = "persistence")]
    mod persistence_tests {
        use super::*;
        use tempfile::TempDir;

        #[test]
        fn test_json_file_backend_save_load_roundtrip() {
            let temp_dir = TempDir::new().unwrap();
            let backend = JsonFileBackend::new(temp_dir.path(), "test-model");

            let mut state = EngineState::new("test-model", 384);
            state.chunks.insert(
                "chunk1".to_string(),
                DocumentChunk {
                    id: "chunk1".to_string(),
                    document_name: "test.pdf".to_string(),
                    text: "hello world".to_string(),
                    embedding: vec![1.0, 0.0, 0.0],
                    chunk_index: 0,
                    page_number: 1,
                    section: None,
                    metadata: crate::types::ChunkMetadata::default(),
                    tags: std::collections::HashSet::new(),
                    resolution: crate::types::Resolution::default(),
                    parent_id: None,
                },
            );
            state
                .document_hashes
                .insert("test.pdf".to_string(), "hash123".to_string());

            // Save
            backend.save(&state).unwrap();

            // Load
            let loaded = backend.load().unwrap().expect("should load state");

            assert_eq!(loaded.schema_version, state.schema_version);
            assert_eq!(loaded.embedding_model_id, state.embedding_model_id);
            assert_eq!(loaded.chunks.len(), 1);
            assert!(loaded.chunks.contains_key("chunk1"));
            assert_eq!(
                loaded.document_hashes.get("test.pdf"),
                Some(&"hash123".to_string())
            );
        }

        #[test]
        fn test_json_file_backend_load_nonexistent() {
            let temp_dir = TempDir::new().unwrap();
            let backend = JsonFileBackend::new(temp_dir.path(), "test-model");

            let result = backend.load().unwrap();
            assert!(result.is_none());
        }

        #[test]
        fn test_json_file_backend_atomic_write() {
            let temp_dir = TempDir::new().unwrap();
            let backend = JsonFileBackend::new(temp_dir.path(), "test-model");

            let state = EngineState::new("test-model", 384);
            backend.save(&state).unwrap();

            // Check that temp file doesn't exist
            let temp_path = backend.index_path().with_extension("json.tmp");
            assert!(!temp_path.exists());

            // Check that final file exists
            assert!(backend.index_path().exists());
        }

        #[test]
        fn test_json_file_backend_corrupted_file_returns_none() {
            let temp_dir = TempDir::new().unwrap();
            let backend = JsonFileBackend::new(temp_dir.path(), "test-model");

            // Write corrupted JSON
            let path = backend.index_path();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "{ invalid json }}}").unwrap();

            // Load should return None (graceful degradation)
            let result = backend.load().unwrap();
            assert!(
                result.is_none(),
                "Corrupted file should return None, not error"
            );
        }

        #[test]
        fn test_json_file_backend_corruption_recovery() {
            let temp_dir = TempDir::new().unwrap();
            let backend = JsonFileBackend::new(temp_dir.path(), "test-model");

            // Write truncated JSON (simulating crash during write)
            let path = backend.index_path();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(
                &path,
                r#"{"schema_version": 2, "embedding_model_id": "test-model""#,
            )
            .unwrap();

            // Load should return None (graceful degradation/automatic rebuild)
            let result = backend.load().unwrap();
            assert!(
                result.is_none(),
                "Truncated file should trigger automatic recovery (return None)"
            );
        }

        #[test]
        fn test_json_file_backend_empty_file_returns_none() {
            let temp_dir = TempDir::new().unwrap();
            let backend = JsonFileBackend::new(temp_dir.path(), "test-model");

            // Write empty file
            let path = backend.index_path();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "").unwrap();

            // Load should return None
            let result = backend.load().unwrap();
            assert!(result.is_none());
        }

        #[test]
        fn test_json_file_backend_overwrite_existing() {
            let temp_dir = TempDir::new().unwrap();
            let backend = JsonFileBackend::new(temp_dir.path(), "test-model");

            // Save initial state
            let mut state1 = EngineState::new("test-model", 384);
            state1.chunks.insert(
                "old-chunk".to_string(),
                DocumentChunk {
                    id: "old-chunk".to_string(),
                    document_name: "old.pdf".to_string(),
                    text: "old content".to_string(),
                    embedding: vec![1.0],
                    chunk_index: 0,
                    page_number: 1,
                    section: None,
                    metadata: crate::types::ChunkMetadata::default(),
                    tags: std::collections::HashSet::new(),
                    resolution: crate::types::Resolution::default(),
                    parent_id: None,
                },
            );
            backend.save(&state1).unwrap();

            // Save new state (overwrites)
            let mut state2 = EngineState::new("test-model", 384);
            state2.chunks.insert(
                "new-chunk".to_string(),
                DocumentChunk {
                    id: "new-chunk".to_string(),
                    document_name: "new.pdf".to_string(),
                    text: "new content".to_string(),
                    embedding: vec![2.0],
                    chunk_index: 0,
                    page_number: 1,
                    section: None,
                    metadata: crate::types::ChunkMetadata::default(),
                    tags: std::collections::HashSet::new(),
                    resolution: crate::types::Resolution::default(),
                    parent_id: None,
                },
            );
            backend.save(&state2).unwrap();

            // Load should return new state
            let loaded = backend.load().unwrap().unwrap();
            assert!(!loaded.chunks.contains_key("old-chunk"));
            assert!(loaded.chunks.contains_key("new-chunk"));
        }

        #[test]
        fn test_json_file_backend_preserves_embedding_precision() {
            let temp_dir = TempDir::new().unwrap();
            let backend = JsonFileBackend::new(temp_dir.path(), "test-model");

            // Use precise floating point values
            let precise_embedding = vec![0.123_456_79_f32, -0.987_654_3_f32, 0.000001f32];

            let mut state = EngineState::new("test-model", 3);
            state.chunks.insert(
                "chunk1".to_string(),
                DocumentChunk {
                    id: "chunk1".to_string(),
                    document_name: "test.pdf".to_string(),
                    text: "test".to_string(),
                    embedding: precise_embedding.clone(),
                    chunk_index: 0,
                    page_number: 1,
                    section: None,
                    metadata: crate::types::ChunkMetadata::default(),
                    tags: std::collections::HashSet::new(),
                    resolution: crate::types::Resolution::default(),
                    parent_id: None,
                },
            );
            backend.save(&state).unwrap();

            let loaded = backend.load().unwrap().unwrap();
            let loaded_embedding = &loaded.chunks.get("chunk1").unwrap().embedding;

            // Verify precision is maintained (within f32 limits)
            for (original, loaded) in precise_embedding.iter().zip(loaded_embedding.iter()) {
                assert!(
                    (original - loaded).abs() < 1e-6,
                    "Embedding precision lost: {} vs {}",
                    original,
                    loaded
                );
            }
        }

        #[test]
        fn test_json_file_backend_different_models_isolated() {
            let temp_dir = TempDir::new().unwrap();

            // Save state for model A
            let backend_a = JsonFileBackend::new(temp_dir.path(), "model-a");
            let mut state_a = EngineState::new("model-a", 128);
            state_a.chunks.insert(
                "chunk-a".to_string(),
                DocumentChunk {
                    id: "chunk-a".to_string(),
                    document_name: "a.pdf".to_string(),
                    text: "content a".to_string(),
                    embedding: vec![1.0; 128],
                    chunk_index: 0,
                    page_number: 1,
                    section: None,
                    metadata: crate::types::ChunkMetadata::default(),
                    tags: std::collections::HashSet::new(),
                    resolution: crate::types::Resolution::default(),
                    parent_id: None,
                },
            );
            backend_a.save(&state_a).unwrap();

            // Save state for model B
            let backend_b = JsonFileBackend::new(temp_dir.path(), "model-b");
            let mut state_b = EngineState::new("model-b", 256);
            state_b.chunks.insert(
                "chunk-b".to_string(),
                DocumentChunk {
                    id: "chunk-b".to_string(),
                    document_name: "b.pdf".to_string(),
                    text: "content b".to_string(),
                    embedding: vec![2.0; 256],
                    chunk_index: 0,
                    page_number: 1,
                    section: None,
                    metadata: crate::types::ChunkMetadata::default(),
                    tags: std::collections::HashSet::new(),
                    resolution: crate::types::Resolution::default(),
                    parent_id: None,
                },
            );
            backend_b.save(&state_b).unwrap();

            // Load each - should be isolated
            let loaded_a = backend_a.load().unwrap().unwrap();
            let loaded_b = backend_b.load().unwrap().unwrap();

            assert!(loaded_a.chunks.contains_key("chunk-a"));
            assert!(!loaded_a.chunks.contains_key("chunk-b"));
            assert!(loaded_b.chunks.contains_key("chunk-b"));
            assert!(!loaded_b.chunks.contains_key("chunk-a"));
            assert_eq!(loaded_a.embedding_dim, 128);
            assert_eq!(loaded_b.embedding_dim, 256);
        }
    }
}
