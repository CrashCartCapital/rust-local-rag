//! # Document Relationships
//!
//! Phase 4: Graph-based document linking for context expansion.
//!
//! Relationships are document-level (not chunk-level) and stored per collection.
//! Cross-collection edges are forbidden.
//!
//! ## Relationship Types
//!
//! - `Citation` (Directed): Document A references Document B
//! - `Version` (Directed): Document B supersedes Document A
//! - `Related` (Undirected): High semantic similarity
//! - `ParentChild` (Directed): Document B is part of Document A

use std::collections::{HashMap, HashSet};

/// Type alias for document names (consistent with existing codebase).
pub type DocumentName = String;

/// Types of relationships between documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RelationshipType {
    /// Document A cites/references Document B (directed).
    Citation,
    /// Document B supersedes Document A (directed).
    Version,
    /// Documents are semantically related (undirected).
    #[default]
    Related,
    /// Document B is a child of Document A (directed).
    ParentChild,
}

impl RelationshipType {
    /// Returns true if this relationship type is undirected.
    pub fn is_undirected(&self) -> bool {
        matches!(self, RelationshipType::Related)
    }
}

/// A relationship between two documents.
#[derive(Debug, Clone, PartialEq)]
pub struct Relationship {
    /// The source document.
    pub source: DocumentName,
    /// The target document.
    pub target: DocumentName,
    /// The type of relationship.
    pub rel_type: RelationshipType,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f32,
}

impl Relationship {
    /// Create a new relationship.
    pub fn new(
        source: impl Into<DocumentName>,
        target: impl Into<DocumentName>,
        rel_type: RelationshipType,
        confidence: f32,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            rel_type,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Configuration for document relationships per collection.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipConfig {
    /// Which relationship types are enabled.
    pub enabled_types: HashSet<RelationshipType>,
    /// Similarity threshold for auto-detection (default: 0.85).
    pub similarity_threshold: f32,
    /// Boost factor for related documents in search (default: 0.1).
    pub boost_factor: f32,
    /// Whether auto-detection is enabled.
    pub auto_detect: bool,
}

impl Default for RelationshipConfig {
    fn default() -> Self {
        Self {
            enabled_types: [
                RelationshipType::Citation,
                RelationshipType::Version,
                RelationshipType::Related,
                RelationshipType::ParentChild,
            ]
            .into_iter()
            .collect(),
            similarity_threshold: 0.85,
            boost_factor: 0.1,
            auto_detect: false,
        }
    }
}

impl RelationshipConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable only specific relationship types.
    pub fn with_enabled_types(mut self, types: impl IntoIterator<Item = RelationshipType>) -> Self {
        self.enabled_types = types.into_iter().collect();
        self
    }

    /// Set the similarity threshold for auto-detection.
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the boost factor for related documents.
    pub fn with_boost_factor(mut self, factor: f32) -> Self {
        self.boost_factor = factor.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable auto-detection.
    pub fn with_auto_detect(mut self, enabled: bool) -> Self {
        self.auto_detect = enabled;
        self
    }
}

/// Edge in the relationship graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    /// The connected document.
    pub document: DocumentName,
    /// The type of relationship.
    pub rel_type: RelationshipType,
    /// Confidence score.
    pub confidence: f32,
}

/// Index of document relationships using bidirectional adjacency lists.
///
/// For directed relationships:
/// - `outgoing[source]` contains edges to targets
/// - `incoming[target]` contains edges from sources
///
/// For undirected relationships (Related):
/// - Both outgoing and incoming are updated symmetrically
#[derive(Debug, Clone, Default)]
pub struct RelationIndex {
    /// Outgoing edges: source -> [(target, type, confidence)]
    outgoing: HashMap<DocumentName, Vec<Edge>>,
    /// Incoming edges: target -> [(source, type, confidence)]
    incoming: HashMap<DocumentName, Vec<Edge>>,
}

impl RelationIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a relationship to the index.
    ///
    /// For undirected relationships (Related), this automatically creates
    /// bidirectional edges (unless it's a self-relationship).
    ///
    /// Note: This method does not deduplicate. Adding the same relationship
    /// twice will create duplicate edges. Use `has_relationship_of_type()`
    /// to check before adding if deduplication is needed.
    pub fn add(&mut self, relationship: Relationship) {
        let Relationship {
            source,
            target,
            rel_type,
            confidence,
        } = relationship;

        let is_self_relationship = source == target;

        // Add outgoing edge from source to target
        self.outgoing.entry(source.clone()).or_default().push(Edge {
            document: target.clone(),
            rel_type,
            confidence,
        });

        // Add incoming edge from target to source
        self.incoming.entry(target.clone()).or_default().push(Edge {
            document: source.clone(),
            rel_type,
            confidence,
        });

        // For undirected relationships, also add the reverse
        // (but skip for self-relationships to avoid duplicate edges)
        if rel_type.is_undirected() && !is_self_relationship {
            self.outgoing.entry(target.clone()).or_default().push(Edge {
                document: source.clone(),
                rel_type,
                confidence,
            });

            self.incoming.entry(source).or_default().push(Edge {
                document: target,
                rel_type,
                confidence,
            });
        }
    }

    /// Get all outgoing edges from a document.
    pub fn outgoing(&self, document: &str) -> &[Edge] {
        self.outgoing
            .get(document)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all incoming edges to a document.
    pub fn incoming(&self, document: &str) -> &[Edge] {
        self.incoming
            .get(document)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all documents related to a given document (both directions).
    pub fn neighbors(&self, document: &str) -> HashSet<DocumentName> {
        let mut result = HashSet::new();

        for edge in self.outgoing(document) {
            result.insert(edge.document.clone());
        }

        for edge in self.incoming(document) {
            result.insert(edge.document.clone());
        }

        result
    }

    /// Get related documents of a specific type.
    pub fn neighbors_of_type(
        &self,
        document: &str,
        rel_type: RelationshipType,
    ) -> Vec<(DocumentName, f32)> {
        let mut result = Vec::new();

        // For outgoing edges
        for edge in self.outgoing(document) {
            if edge.rel_type == rel_type {
                result.push((edge.document.clone(), edge.confidence));
            }
        }

        // For incoming edges (if not already covered by undirected)
        if !rel_type.is_undirected() {
            for edge in self.incoming(document) {
                if edge.rel_type == rel_type {
                    result.push((edge.document.clone(), edge.confidence));
                }
            }
        }

        result
    }

    /// Remove all relationships for a document (cleanup on delete).
    pub fn remove_document(&mut self, document: &str) {
        // Remove from outgoing
        self.outgoing.remove(document);

        // Remove from all incoming lists
        for edges in self.incoming.values_mut() {
            edges.retain(|e| e.document != document);
        }

        // Remove from incoming
        self.incoming.remove(document);

        // Remove from all outgoing lists
        for edges in self.outgoing.values_mut() {
            edges.retain(|e| e.document != document);
        }

        // Clean up empty entries
        self.outgoing.retain(|_, v| !v.is_empty());
        self.incoming.retain(|_, v| !v.is_empty());
    }

    /// Check if a relationship exists between two documents.
    pub fn has_relationship(&self, source: &str, target: &str) -> bool {
        self.outgoing
            .get(source)
            .map(|edges| edges.iter().any(|e| e.document == target))
            .unwrap_or(false)
    }

    /// Check if a specific relationship type exists.
    pub fn has_relationship_of_type(
        &self,
        source: &str,
        target: &str,
        rel_type: RelationshipType,
    ) -> bool {
        self.outgoing
            .get(source)
            .map(|edges| {
                edges
                    .iter()
                    .any(|e| e.document == target && e.rel_type == rel_type)
            })
            .unwrap_or(false)
    }

    /// Get the number of relationships.
    pub fn len(&self) -> usize {
        self.outgoing.values().map(|v| v.len()).sum()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.outgoing.is_empty()
    }

    /// Clear all relationships.
    pub fn clear(&mut self) {
        self.outgoing.clear();
        self.incoming.clear();
    }

    /// Get all documents in the index.
    pub fn documents(&self) -> HashSet<DocumentName> {
        let mut docs: HashSet<DocumentName> = self.outgoing.keys().cloned().collect();
        docs.extend(self.incoming.keys().cloned());
        docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Phase 4 TDD Tests: RelationshipType
    // ========================================================================

    #[test]
    fn test_relationship_type_citation_directed() {
        assert!(!RelationshipType::Citation.is_undirected());
    }

    #[test]
    fn test_relationship_type_version_directed() {
        assert!(!RelationshipType::Version.is_undirected());
    }

    #[test]
    fn test_relationship_type_related_undirected() {
        assert!(RelationshipType::Related.is_undirected());
    }

    #[test]
    fn test_relationship_type_parent_child_directed() {
        assert!(!RelationshipType::ParentChild.is_undirected());
    }

    #[test]
    fn test_relationship_type_default() {
        assert_eq!(RelationshipType::default(), RelationshipType::Related);
    }

    // ========================================================================
    // Phase 4 TDD Tests: Relationship
    // ========================================================================

    #[test]
    fn test_relationship_creation() {
        let rel = Relationship::new("doc_a.pdf", "doc_b.pdf", RelationshipType::Citation, 0.95);
        assert_eq!(rel.source, "doc_a.pdf");
        assert_eq!(rel.target, "doc_b.pdf");
        assert_eq!(rel.rel_type, RelationshipType::Citation);
        assert!((rel.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_relationship_confidence_clamping() {
        let rel = Relationship::new("a", "b", RelationshipType::Related, 1.5);
        assert!((rel.confidence - 1.0).abs() < 1e-6);

        let rel2 = Relationship::new("a", "b", RelationshipType::Related, -0.5);
        assert!((rel2.confidence - 0.0).abs() < 1e-6);
    }

    // ========================================================================
    // Phase 4 TDD Tests: RelationshipConfig
    // ========================================================================

    #[test]
    fn test_config_defaults() {
        let config = RelationshipConfig::default();
        assert_eq!(config.enabled_types.len(), 4);
        assert!((config.similarity_threshold - 0.85).abs() < 1e-6);
        assert!((config.boost_factor - 0.1).abs() < 1e-6);
        assert!(!config.auto_detect);
    }

    #[test]
    fn test_config_builder() {
        let config = RelationshipConfig::new()
            .with_enabled_types([RelationshipType::Related, RelationshipType::Citation])
            .with_similarity_threshold(0.9)
            .with_boost_factor(0.2)
            .with_auto_detect(true);

        assert_eq!(config.enabled_types.len(), 2);
        assert!(config.enabled_types.contains(&RelationshipType::Related));
        assert!(config.enabled_types.contains(&RelationshipType::Citation));
        assert!((config.similarity_threshold - 0.9).abs() < 1e-6);
        assert!((config.boost_factor - 0.2).abs() < 1e-6);
        assert!(config.auto_detect);
    }

    #[test]
    fn test_config_threshold_clamping() {
        let config = RelationshipConfig::new().with_similarity_threshold(1.5);
        assert!((config.similarity_threshold - 1.0).abs() < 1e-6);

        let config2 = RelationshipConfig::new().with_similarity_threshold(-0.5);
        assert!((config2.similarity_threshold - 0.0).abs() < 1e-6);
    }

    // ========================================================================
    // Phase 4 TDD Tests: RelationIndex Basic Operations
    // ========================================================================

    #[test]
    fn test_relation_index_new_empty() {
        let index = RelationIndex::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_relation_index_add_directed() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new(
            "doc_a.pdf",
            "doc_b.pdf",
            RelationshipType::Citation,
            0.9,
        ));

        // Check outgoing from source
        let outgoing = index.outgoing("doc_a.pdf");
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].document, "doc_b.pdf");

        // Check incoming to target
        let incoming = index.incoming("doc_b.pdf");
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].document, "doc_a.pdf");

        // Check reverse is NOT present (directed)
        assert!(index.outgoing("doc_b.pdf").is_empty());
    }

    #[test]
    fn test_relation_index_add_undirected() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new(
            "doc_a.pdf",
            "doc_b.pdf",
            RelationshipType::Related,
            0.9,
        ));

        // Check both directions exist
        let outgoing_a = index.outgoing("doc_a.pdf");
        assert_eq!(outgoing_a.len(), 1);
        assert_eq!(outgoing_a[0].document, "doc_b.pdf");

        let outgoing_b = index.outgoing("doc_b.pdf");
        assert_eq!(outgoing_b.len(), 1);
        assert_eq!(outgoing_b[0].document, "doc_a.pdf");
    }

    #[test]
    fn test_relation_index_neighbors() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new("c", "a", RelationshipType::Version, 0.8));

        let neighbors = index.neighbors("a");
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains("b")); // a -> b
        assert!(neighbors.contains("c")); // c -> a
    }

    #[test]
    fn test_relation_index_neighbors_of_type() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new("a", "c", RelationshipType::Related, 0.8));
        index.add(Relationship::new("d", "a", RelationshipType::Citation, 0.7));

        // Only outgoing citations from a
        let citations = index.neighbors_of_type("a", RelationshipType::Citation);
        // Should include a->b (outgoing) and d->a (incoming)
        assert_eq!(citations.len(), 2);
    }

    #[test]
    fn test_relation_index_has_relationship() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));

        assert!(index.has_relationship("a", "b"));
        assert!(!index.has_relationship("b", "a")); // Directed, no reverse
        assert!(!index.has_relationship("a", "c")); // Doesn't exist
    }

    #[test]
    fn test_relation_index_has_relationship_of_type() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new("a", "b", RelationshipType::Related, 0.8));

        assert!(index.has_relationship_of_type("a", "b", RelationshipType::Citation));
        assert!(index.has_relationship_of_type("a", "b", RelationshipType::Related));
        assert!(!index.has_relationship_of_type("a", "b", RelationshipType::Version));
    }

    // ========================================================================
    // Phase 4 TDD Tests: RelationIndex Removal
    // ========================================================================

    #[test]
    fn test_relation_index_remove_document() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new("a", "c", RelationshipType::Related, 0.8));
        index.add(Relationship::new("d", "a", RelationshipType::Version, 0.7));

        // Remove document "a"
        index.remove_document("a");

        // a should be completely gone
        assert!(index.outgoing("a").is_empty());
        assert!(index.incoming("a").is_empty());
        assert!(!index.neighbors("b").contains("a"));
        assert!(!index.neighbors("c").contains("a"));
        assert!(!index.neighbors("d").contains("a"));
    }

    #[test]
    fn test_relation_index_clear() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new("c", "d", RelationshipType::Related, 0.8));

        assert!(!index.is_empty());

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_relation_index_documents() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new("c", "d", RelationshipType::Related, 0.8));

        let docs = index.documents();
        assert_eq!(docs.len(), 4);
        assert!(docs.contains("a"));
        assert!(docs.contains("b"));
        assert!(docs.contains("c"));
        assert!(docs.contains("d"));
    }

    // ========================================================================
    // Phase 4 TDD Tests: Complex Scenarios
    // ========================================================================

    #[test]
    fn test_relation_index_multiple_edges_same_pair() {
        let mut index = RelationIndex::new();

        // Add multiple relationship types between same pair
        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new("a", "b", RelationshipType::Related, 0.8));

        // Both should be present
        let outgoing = index.outgoing("a");
        assert_eq!(outgoing.len(), 2);
    }

    #[test]
    fn test_relation_index_self_relationship() {
        let mut index = RelationIndex::new();

        // Document relates to itself (edge case)
        index.add(Relationship::new("a", "a", RelationshipType::Related, 1.0));

        let neighbors = index.neighbors("a");
        assert!(neighbors.contains("a"));
    }

    #[test]
    fn test_relation_index_len() {
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        // Citation is directed, so 1 edge
        assert_eq!(index.len(), 1);

        index.add(Relationship::new("c", "d", RelationshipType::Related, 0.8));
        // Related is undirected, so 2 edges (c->d and d->c)
        assert_eq!(index.len(), 3);
    }

    // ========================================================================
    // Phase 4 Edge Case Tests (Codex Review Findings)
    // ========================================================================

    #[test]
    fn test_relation_index_duplicate_edges_allowed() {
        // Documented behavior: duplicates are allowed (not deduplicated)
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new(
            "a",
            "b",
            RelationshipType::Citation,
            0.95,
        ));

        // Both edges should be present
        let outgoing = index.outgoing("a");
        assert_eq!(outgoing.len(), 2);
    }

    #[test]
    fn test_relation_index_undirected_self_relationship_no_duplicate() {
        // Self-relationship with undirected type should NOT create duplicate edges
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "a", RelationshipType::Related, 1.0));

        // Should only have 1 outgoing edge (not 2)
        let outgoing = index.outgoing("a");
        assert_eq!(outgoing.len(), 1);

        // Should only have 1 incoming edge (not 2)
        let incoming = index.incoming("a");
        assert_eq!(incoming.len(), 1);

        // len() should be 1, not 2
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_relation_index_undirected_has_relationship_symmetric() {
        // Undirected relationships should be queryable from both directions
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Related, 0.9));

        // Both directions should report the relationship exists
        assert!(index.has_relationship("a", "b"));
        assert!(index.has_relationship("b", "a")); // Symmetric!
    }

    #[test]
    fn test_relation_index_directed_has_relationship_not_symmetric() {
        // Directed relationships should NOT be queryable from reverse direction
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));

        assert!(index.has_relationship("a", "b"));
        assert!(!index.has_relationship("b", "a")); // Not symmetric!
    }

    #[test]
    fn test_relation_index_remove_hub_node() {
        // Test removal of a hub node with mixed incoming/outgoing/undirected
        let mut index = RelationIndex::new();

        // a is a hub with various relationships
        index.add(Relationship::new("a", "b", RelationshipType::Citation, 0.9));
        index.add(Relationship::new("c", "a", RelationshipType::Version, 0.8));
        index.add(Relationship::new("a", "d", RelationshipType::Related, 0.7));

        // Verify initial state
        assert!(!index.is_empty());
        assert!(index.has_relationship("a", "b"));
        assert!(index.has_relationship("c", "a"));
        assert!(index.has_relationship("a", "d"));
        assert!(index.has_relationship("d", "a")); // Related is symmetric

        // Remove the hub
        index.remove_document("a");

        // a should be completely gone
        assert!(index.outgoing("a").is_empty());
        assert!(index.incoming("a").is_empty());

        // Other nodes should not reference a
        assert!(!index.neighbors("b").contains("a"));
        assert!(!index.neighbors("c").contains("a"));
        assert!(!index.neighbors("d").contains("a"));

        // has_relationship should return false
        assert!(!index.has_relationship("a", "b"));
        assert!(!index.has_relationship("c", "a"));
        assert!(!index.has_relationship("a", "d"));
    }

    #[test]
    fn test_relation_index_directed_self_relationship() {
        // Self-relationship with directed type
        let mut index = RelationIndex::new();

        index.add(Relationship::new("a", "a", RelationshipType::Citation, 1.0));

        // Should have exactly 1 edge each way
        assert_eq!(index.outgoing("a").len(), 1);
        assert_eq!(index.incoming("a").len(), 1);
        assert_eq!(index.len(), 1);
    }
}
