//! Collection management for document organization.
//!
//! Collections provide logical partitioning of documents within a RAG engine,
//! enabling multi-tenancy and per-collection configuration.
//!
//! # Overview
//!
//! - Each collection has its own `IndexSet` and configuration
//! - Documents belong to exactly one collection
//! - A "default" collection is created automatically for backward compatibility
//! - Collections can have different chunking and search weight settings

use crate::types::SearchWeights;

#[cfg(feature = "persistence")]
use serde::{Deserialize, Serialize};

/// Unique identifier for a collection.
///
/// Wraps a String to provide type safety and prevent mixing with document names.
#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CollectionId(String);

impl CollectionId {
    /// Create a new collection ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string reference.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The default collection ID for backward compatibility.
    pub const DEFAULT: &'static str = "default";

    /// Create the default collection ID.
    pub fn default_collection() -> Self {
        Self::new(Self::DEFAULT)
    }
}

impl Default for CollectionId {
    fn default() -> Self {
        Self::default_collection()
    }
}

impl std::fmt::Display for CollectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for CollectionId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for CollectionId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// Configuration for a collection.
///
/// Each collection can have its own chunking and search settings.
#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct CollectionConfig {
    /// Tokens per chunk (default: 200)
    pub chunk_tokens: usize,
    /// Sentence overlap between chunks (default: 2)
    pub sentence_overlap: usize,
    /// Search weights for this collection
    pub weights: SearchWeights,
    /// Enable semantic tag expansion (Phase 3 feature, placeholder)
    pub tag_expansion_enabled: bool,
    /// Tag expansion similarity threshold (Phase 3 feature, placeholder)
    pub tag_expansion_threshold: f32,
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            chunk_tokens: 200,
            sentence_overlap: 2,
            weights: SearchWeights::default(),
            tag_expansion_enabled: false,
            tag_expansion_threshold: 0.8,
        }
    }
}

impl CollectionConfig {
    /// Create a new configuration with custom chunk settings.
    pub fn with_chunking(mut self, tokens: usize, overlap: usize) -> Self {
        self.chunk_tokens = tokens;
        self.sentence_overlap = overlap;
        self
    }

    /// Create a new configuration with custom search weights.
    pub fn with_weights(mut self, weights: SearchWeights) -> Self {
        self.weights = weights;
        self
    }
}

/// A document collection with its configuration and metadata.
#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Collection {
    /// Unique identifier for this collection
    pub id: CollectionId,
    /// Human-readable name
    pub name: String,
    /// Optional description
    pub description: Option<String>,
    /// Collection-specific configuration
    pub config: CollectionConfig,
    /// Creation timestamp (ISO 8601 string for simplicity)
    pub created_at: String,
}

impl Collection {
    /// Create a new collection with the given ID and name.
    pub fn new(id: impl Into<CollectionId>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            config: CollectionConfig::default(),
            created_at: String::new(), // Will be set by engine
        }
    }

    /// Set the description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the configuration.
    pub fn with_config(mut self, config: CollectionConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the creation timestamp.
    pub fn with_created_at(mut self, timestamp: impl Into<String>) -> Self {
        self.created_at = timestamp.into();
        self
    }

    /// Create the default collection.
    pub fn default_collection() -> Self {
        Self::new(CollectionId::default_collection(), "Default Collection")
            .with_description("The default collection for backward compatibility")
    }
}

/// Registry of all collections, persisted to manifest.json.
#[cfg_attr(feature = "persistence", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct CollectionManifest {
    /// Schema version for future migrations
    pub version: u32,
    /// Map of collection ID to collection metadata
    pub collections: std::collections::HashMap<CollectionId, Collection>,
}

impl CollectionManifest {
    /// Current manifest schema version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new empty manifest.
    pub fn new() -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            collections: std::collections::HashMap::new(),
        }
    }

    /// Create a manifest with the default collection.
    pub fn with_default_collection() -> Self {
        let mut manifest = Self::new();
        manifest.add(Collection::default_collection());
        manifest
    }

    /// Add a collection to the manifest.
    pub fn add(&mut self, collection: Collection) {
        self.collections.insert(collection.id.clone(), collection);
    }

    /// Get a collection by ID.
    pub fn get(&self, id: &CollectionId) -> Option<&Collection> {
        self.collections.get(id)
    }

    /// Remove a collection by ID.
    pub fn remove(&mut self, id: &CollectionId) -> Option<Collection> {
        self.collections.remove(id)
    }

    /// Check if a collection exists.
    pub fn contains(&self, id: &CollectionId) -> bool {
        self.collections.contains_key(id)
    }

    /// Get all collection IDs.
    pub fn ids(&self) -> impl Iterator<Item = &CollectionId> {
        self.collections.keys()
    }

    /// Get all collections.
    pub fn iter(&self) -> impl Iterator<Item = &Collection> {
        self.collections.values()
    }

    /// Number of collections.
    pub fn len(&self) -> usize {
        self.collections.len()
    }

    /// Check if manifest is empty.
    pub fn is_empty(&self) -> bool {
        self.collections.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Phase 1 TDD Tests: Collections
    // ========================================================================

    #[test]
    fn test_collection_id_creation() {
        let id = CollectionId::new("research");
        assert_eq!(id.as_str(), "research");
        assert_eq!(id.to_string(), "research");
    }

    #[test]
    fn test_collection_id_default() {
        let id = CollectionId::default();
        assert_eq!(id.as_str(), "default");
        assert_eq!(id, CollectionId::default_collection());
    }

    #[test]
    fn test_collection_id_from_conversions() {
        let from_str: CollectionId = "test".into();
        let from_string: CollectionId = String::from("test").into();
        assert_eq!(from_str, from_string);
    }

    #[test]
    fn test_collection_id_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CollectionId::new("a"));
        set.insert(CollectionId::new("b"));
        set.insert(CollectionId::new("a")); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_collection_config_defaults() {
        let config = CollectionConfig::default();
        assert_eq!(config.chunk_tokens, 200);
        assert_eq!(config.sentence_overlap, 2);
        assert!(!config.tag_expansion_enabled);
        assert!((config.tag_expansion_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_collection_config_builder() {
        let config = CollectionConfig::default()
            .with_chunking(300, 3)
            .with_weights(SearchWeights {
                embedding: 0.9,
                lexical: 0.1,
                ..Default::default()
            });

        assert_eq!(config.chunk_tokens, 300);
        assert_eq!(config.sentence_overlap, 3);
        assert!((config.weights.embedding - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_collection_creation() {
        let collection = Collection::new("research", "Research Papers");
        assert_eq!(collection.id.as_str(), "research");
        assert_eq!(collection.name, "Research Papers");
        assert!(collection.description.is_none());
    }

    #[test]
    fn test_collection_builder() {
        let collection = Collection::new("legal", "Legal Documents")
            .with_description("Contracts and agreements")
            .with_created_at("2026-01-12T10:00:00Z");

        assert_eq!(collection.id.as_str(), "legal");
        assert_eq!(collection.description, Some("Contracts and agreements".into()));
        assert_eq!(collection.created_at, "2026-01-12T10:00:00Z");
    }

    #[test]
    fn test_collection_default_collection() {
        let collection = Collection::default_collection();
        assert_eq!(collection.id.as_str(), "default");
        assert_eq!(collection.name, "Default Collection");
        assert!(collection.description.is_some());
    }

    #[test]
    fn test_manifest_new_empty() {
        let manifest = CollectionManifest::new();
        assert!(manifest.is_empty());
        assert_eq!(manifest.version, CollectionManifest::CURRENT_VERSION);
    }

    #[test]
    fn test_manifest_with_default_collection() {
        let manifest = CollectionManifest::with_default_collection();
        assert_eq!(manifest.len(), 1);
        assert!(manifest.contains(&CollectionId::default_collection()));
    }

    #[test]
    fn test_manifest_add_and_get() {
        let mut manifest = CollectionManifest::new();
        manifest.add(Collection::new("research", "Research"));

        let collection = manifest.get(&CollectionId::new("research"));
        assert!(collection.is_some());
        assert_eq!(collection.unwrap().name, "Research");
    }

    #[test]
    fn test_manifest_remove() {
        let mut manifest = CollectionManifest::with_default_collection();
        let removed = manifest.remove(&CollectionId::default_collection());
        assert!(removed.is_some());
        assert!(manifest.is_empty());
    }

    #[test]
    fn test_manifest_iteration() {
        let mut manifest = CollectionManifest::new();
        manifest.add(Collection::new("a", "A"));
        manifest.add(Collection::new("b", "B"));

        let ids: Vec<_> = manifest.ids().collect();
        assert_eq!(ids.len(), 2);

        let collections: Vec<_> = manifest.iter().collect();
        assert_eq!(collections.len(), 2);
    }
}
