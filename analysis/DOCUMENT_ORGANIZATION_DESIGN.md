# Document Organization Architecture Design

**Status**: Design Complete | **Created**: 2026-01-12
**Methodology**: CRASH MCP (6 steps) + Codex consultation + Gemini consultations (8 rounds)

---

## Executive Summary

This document defines a comprehensive architecture for organizing, categorizing, and filtering documents within rust-local-rag. The design introduces five major feature areas implemented across six phases:

| Feature | Purpose | Phase |
|---------|---------|-------|
| **QuerySpec Contract** | Stable query interface for all features | 0 |
| **Collections** | Logical partitions with separate configs | 1 |
| **Tags + Filter DSL** | Metadata filtering with AND/OR/NOT | 2 |
| **Semantic Tag Expansion** | Query-time tag similarity boosting | 3 |
| **Document Relationships** | Graph-based document linking | 4 |
| **Multi-Resolution Indexing** | Doc/Section/Chunk hierarchy | 5 |

**Key Architectural Principle**: Everything plugs into `QuerySpec` without rewriting APIs.

---

## Part 1: Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Tools Layer                              │
│   search_documents, create_collection, add_tags, etc.           │
├─────────────────────────────────────────────────────────────────┤
│                    Domain Model Layer                            │
│   QuerySpec, SearchScope, FilterExpr, BoostSpec, ResolutionSpec │
├─────────────────────────────────────────────────────────────────┤
│                  Query Planning Layer                            │
│   QuerySpec → QueryPlan compilation, validation, optimization    │
├─────────────────────────────────────────────────────────────────┤
│                 Collection Runtime Layer                         │
│   Per-collection: IndexSet, Config, Persistence                  │
├─────────────────────────────────────────────────────────────────┤
│                  Index Abstractions Layer                        │
│   VectorIndex, TagIndex, RelationIndex traits                    │
├─────────────────────────────────────────────────────────────────┤
│                Scoring & Reranking Layer                         │
│   ScoreComponents, Reranker trait, boost normalization           │
├─────────────────────────────────────────────────────────────────┤
│               Persistence & Migration Layer                      │
│   Per-collection storage, versioned formats, atomic writes       │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

**Domain Model Layer**
- Define stable identifiers: `DocId`, `ChunkId`, `CollectionId`, `Tag`
- Represent user intent as pure data: `QuerySpec`, `FilterExpr`
- Keep serde-friendly and storage-agnostic

**Index Abstractions Layer**
- `trait VectorIndex { fn search(&self, embedding, top_k) -> Vec<ScoredChunk>; }`
- `trait TagIndex { fn docs_for(&self, predicate) -> BitSet; }`
- `trait RelationIndex { fn neighbors(&self, doc, rel_type) -> Vec<(DocId, f32)>; }`

**Collection Runtime Layer**
- Own per-collection configuration (chunk size, weights, thresholds)
- Own per-collection index set (vector, tags, relations, summaries)
- Expose uniform interface for query planner

**Query Planning Layer**
- Compile `QuerySpec` → `QueryPlan`
- Determine: which collections, which resolutions, which filters, which rerankers
- Validate and reject ambiguous combinations early

**Scoring & Reranking Layer**
- Single scoring pipeline with additive feature signals
- `ScoreComponents { vector, lexical, tag_boost, relation_boost, resolution_prior }`
- Caps and normalization to prevent boost instability

---

## Part 2: Core Types

### QuerySpec (Phase 0)

```rust
/// The universal query specification - all features plug into this.
pub struct QuerySpec {
    /// The semantic query text
    pub query: String,

    /// Which collections to search
    pub scope: SearchScope,

    /// Tag-based filtering (optional)
    pub filter: Option<FilterExpr>,

    /// Score boosting configuration
    pub boosts: BoostSpec,

    /// Resolution preferences (doc/section/chunk)
    pub resolution: ResolutionSpec,

    /// Number of results to return
    pub top_k: usize,

    /// MMR diversity factor (0.0 = pure relevance, 1.0 = max diversity)
    pub diversity_factor: f32,
}

pub enum SearchScope {
    /// Search all collections
    All,
    /// Search a specific collection
    Collection(CollectionId),
    /// Search multiple collections (fan-out then merge)
    Multi(Vec<CollectionId>),
}
```

### FilterExpr (Phase 2)

```rust
/// Tag filter expression supporting AND/OR/NOT
pub enum FilterExpr {
    /// Match exact tag
    Tag(String),
    /// Match tag prefix (e.g., "research/" matches "research/ml")
    Prefix(String),
    /// All conditions must match
    And(Vec<FilterExpr>),
    /// Any condition matches
    Or(Vec<FilterExpr>),
    /// Negate the inner expression
    Not(Box<FilterExpr>),
}
```

### Collection (Phase 1)

```rust
pub struct Collection {
    pub id: CollectionId,
    pub name: String,
    pub description: Option<String>,
    pub config: CollectionConfig,
    pub created_at: DateTime<Utc>,
}

pub struct CollectionConfig {
    /// Tokens per chunk (default: 200)
    pub chunk_tokens: usize,
    /// Sentence overlap between chunks
    pub sentence_overlap: usize,
    /// Search weights for this collection
    pub weights: SearchWeights,
    /// Enable semantic tag expansion
    pub tag_expansion_enabled: bool,
    /// Tag expansion similarity threshold
    pub tag_expansion_threshold: f32,
}
```

---

## Part 3: Filter DSL Specification

### String Syntax (Human-Friendly)

```
<expression> ::= <term> { " OR " <term> }
<term>       ::= <factor> { " " <factor> }           // Implicit AND
<factor>     ::= [ "-" | "NOT " ] <primary>
<primary>    ::= <tag_pattern> | "(" <expression> ")"
<tag_pattern> ::= <quoted_string> | <unquoted_string> [ "*" ]
```

### Examples

| Description | String Syntax | JSON Syntax |
|-------------|---------------|-------------|
| All tagged "research" | `research` | `{"tag": "research"}` |
| Tagged "ml" AND "python" | `ml python` | `{"and": [{"tag": "ml"}, {"tag": "python"}]}` |
| Any subtag of "research/" | `research/*` | `{"prefix": "research/"}` |
| Research but not outdated | `research/* -research/outdated` | `{"and": [{"prefix": "research/"}, {"not": {"tag": "research/outdated"}}]}` |
| Any priority level | `urgent OR important OR priority` | `{"or": [{"tag": "urgent"}, {"tag": "important"}, {"tag": "priority"}]}` |

### Parser Notes
- Tags are case-insensitive
- Operator precedence: NOT > AND > OR
- Use `nom` or `chumsky` crate for string parsing
- `*` wildcard only at end for prefix matching

---

## Part 4: MCP Tool Designs

### Collection Management

```rust
// Create a new collection
create_collection(name: String, description: Option<String>) -> Result<Collection>

// List all collections
list_collections() -> Vec<CollectionSummary>

// Delete a collection (documents remain, grouping removed)
delete_collection(name: String) -> Result<()>

// Add documents to a collection
add_to_collection(collection: String, documents: Vec<String>) -> Result<usize>

// Remove documents from a collection
remove_from_collection(collection: String, documents: Vec<String>) -> Result<usize>
```

### Tag Management

```rust
// Add tags to documents
add_tags(documents: Vec<String>, tags: Vec<String>) -> Result<()>

// Remove tags from documents
remove_tags(documents: Vec<String>, tags: Vec<String>) -> Result<()>

// List all unique tags with frequencies
list_tags() -> HashMap<String, usize>
```

### Enhanced Search

```rust
// Updated search_documents with optional filters
search_documents(
    query: String,
    top_k: Option<usize>,           // Default: 5
    collection: Option<String>,      // Filter to collection
    tags: Option<Vec<String>>,       // Filter by tags
    tag_filter: Option<String>,      // Complex filter expression
    diversity_factor: Option<f32>,   // MMR (default: 0.3)
    weights: Option<QueryWeights>,   // Score tuning
) -> Vec<SearchResult>
```

### Backward Compatibility

All new parameters are optional. Existing calls work unchanged:
```
search_documents(query="quantum computing", top_k=5)  // Still works
```

---

## Part 5: Implementation Phases

### Phase 0: QuerySpec Contract (Foundation)

**Goal**: Establish stable query interface before adding features.

**Deliverables**:
- [ ] Define `QuerySpec`, `SearchScope`, `FilterExpr`, `BoostSpec`, `ResolutionSpec` types
- [ ] Refactor `RagEngine::search()` to accept `QuerySpec` internally
- [ ] Add persistence versioning hooks
- [ ] Maintain backward-compatible `search()` API that builds `QuerySpec`

**Why First**: Everything else plugs into `QuerySpec` without rewriting APIs.

### Phase 1: Collections (Partitioning)

**Goal**: Logical multi-tenancy with per-collection configuration.

**Deliverables**:
- [ ] `CollectionState` and `CollectionConfig` types
- [ ] `HashMap<CollectionId, Collection>` in `RagEngine`
- [ ] `SearchScope::{All, Collection, Multi}` support
- [ ] Per-collection persistence directory layout
- [ ] MCP tools: `create_collection`, `list_collections`, `delete_collection`, `add_to_collection`, `remove_from_collection`
- [ ] Default "global" collection for backward compatibility

**Storage Layout**:
```
data/
├── manifest.json              # Collection registry
├── collections/
│   ├── default/
│   │   └── chunks_nomic-embed-text.json
│   ├── research/
│   │   └── chunks_nomic-embed-text.json
│   └── legal/
│       └── chunks_nomic-embed-text.json
```

### Phase 2: Tags + Filter DSL

**Goal**: Fast metadata-based filtering with expressive queries.

**Deliverables**:
- [ ] `tags: HashSet<String>` field on `DocumentChunk`
- [ ] Hierarchical tag explosion at ingestion ("a/b/c" → ["a", "a/b", "a/b/c"])
- [ ] `TagIndex`: inverted index `HashMap<Tag, HashSet<ChunkId>>`
- [ ] `FilterExpr` enum with parser (string + JSON)
- [ ] Filter evaluation to `BitSet` mask
- [ ] Wire filter mask into search pipeline (pre-filter before scoring)
- [ ] MCP tools: `add_tags`, `remove_tags`, `list_tags`
- [ ] Update `search_documents` with `tags`, `tag_filter` parameters

**Performance**: Use `roaring` crate for large indexes; set operations are O(n) on match count, not total chunks.

### Phase 3: Semantic Tag Expansion

**Goal**: Query-time semantic matching of tags without exact keywords.

**Deliverables**:
- [ ] Embed tags once per collection/model at ingestion
- [ ] Store tag embeddings in `TagIndex`
- [ ] At query time: embed query, find similar tags (cosine > threshold)
- [ ] Expansion modes: `soft_boost` (add to BoostSpec) or `hard_filter` (expand FilterExpr)
- [ ] Config: `tag_expansion_enabled`, `tag_expansion_threshold` (default: 0.8), `max_expanded_tags` (default: 5)

**Why Soft Boost Default**: Hard expansion can harm recall; soft boosting is safer.

### Phase 4: Document Relationships

**Goal**: Graph-based document linking for context expansion.

**Relationship Types**:
| Type | Direction | Description |
|------|-----------|-------------|
| `citation` | Directed | Document A references Document B |
| `version` | Directed | Document B supersedes Document A |
| `related` | Undirected | High semantic similarity |
| `parent_child` | Directed | Document B is part of Document A |

**Deliverables**:
- [ ] `Relationship` struct with `source`, `target`, `rel_type`, `confidence`
- [ ] `RelationIndex` per collection
- [ ] Manual linking via MCP tool: `link_documents(source, target, type)`
- [ ] Auto-detection: compute document centroids, link if similarity > 0.85
- [ ] Reranker that boosts chunks from related documents
- [ ] Config: `relation_types_enabled`, `similarity_threshold`, `relation_boost_factor`

**Design Decision**: Relationships are doc-level, not chunk-level. Forbid cross-collection edges initially.

### Phase 5: Multi-Resolution Indexing

**Goal**: Coarse-to-fine retrieval for better context.

**Resolution Levels**:
| Level | Content | Purpose |
|-------|---------|---------|
| Document | Title + first 5 sentences | "What is this about?" |
| Section | Heading + first 2 sentences | "Find the right chapter" |
| Chunk | 200 tokens (current) | Precise fact retrieval |

**Deliverables**:
- [ ] `Resolution` enum in types
- [ ] `parent_id` linking Chunk → Section → Document
- [ ] Extractive summary generation (no LLM, deterministic)
- [ ] Multi-index search with score merging (RRF or weighted linear)
- [ ] `ResolutionSpec` in `QuerySpec`: `{ strategy: CoarseToFine | Parallel, weights: [doc, section, chunk] }`
- [ ] ~20% storage overhead for summaries

**Why Last**: Most invasive to indexing/persistence; defer until core is stable.

---

## Part 6: Feature Interactions

### Collections × Tags
- Tags are namespaced per collection by default
- Global tag queries require explicit `SearchScope::All`
- Prevent surprise cross-collection results

### Tags × Semantic Expansion
- Start with soft boosting (add to BoostSpec), not hard filtering
- Make expansion opt-in via config
- Cap `max_expanded_tags` to prevent query dilution

### Collections × Relationships
- Forbid cross-collection edges initially
- Add explicit cross-collection links later if needed
- Keeps persistence and deletion clean

### Relationships × Multi-Resolution
- Relationships are doc-level
- Apply boosts at doc aggregation time
- Normalize per-doc to prevent double-counting chunks

### Tag Boost × Relationship Boost
- Treat as separate score components with caps
- Run relationship rerank after base scoring
- Prevent unbounded additive boosts

---

## Part 7: Rust Patterns

| Pattern | Use For |
|---------|---------|
| **Capability traits + composition** | IndexSet with optional components (tags, relations, summaries) |
| **Typed IDs / newtypes** | `DocId`, `ChunkId`, `CollectionId` prevent mixing |
| **Versioned persistence + migrations** | `PersistedIndex::V1(...)`, `V2(...)` enum |
| **AST + evaluator for FilterExpr** | Keep parsing separate from BitSet evaluation |
| **BitSet / roaring bitmaps** | Fast doc filtering with AND/OR/NOT |
| **Two-phase search pipeline** | Candidate generation → filter → score → rerank |
| **spawn_blocking** | PDF extraction, heavy parsing off async runtime |
| **Feature flags / runtime toggles** | Ship features incrementally |

---

## Part 8: Complexity Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Persistence explosion (collections × models × resolutions) | High | Manifest-driven layout, lazy loading, cleanup policies |
| Query planner ambiguity | Medium | Explicit QuerySpec validation, deterministic QueryPlan |
| Ranking instability from multiple boosts | Medium | Score component caps, A/B via eval harness |
| Lock contention in async | High | Fine-grained per-collection locks, spawn_blocking |
| DSL complexity | Medium | Small grammar, structured parse errors, precedence tests |
| Cross-collection relationships | Medium | Defer until core stable, require explicit opt-in |

---

## Part 9: Success Metrics

### Functional
- [ ] Collections isolate documents with separate configs
- [ ] Tag filters correctly apply AND/OR/NOT logic
- [ ] Semantic expansion boosts related tags
- [ ] Relationships boost connected documents
- [ ] Multi-resolution improves "big picture" queries

### Performance
- [ ] Tag filtering adds < 5ms overhead for 10K docs
- [ ] Collection scoping is O(1) not O(collections)
- [ ] Semantic expansion adds < 10ms for 100 unique tags
- [ ] Multi-resolution adds ~20% storage, not 3x

### Usability
- [ ] MCP tools intuitive for Claude Desktop users
- [ ] Backward compatible (existing calls work)
- [ ] Filter DSL parseable by AI assistants

---

## Appendix: Maverick Ideas (Future Exploration)

These ideas emerged from Gemini consultations but are deferred for simplicity:

1. **Semantic Virtual File System (SVFS)**: Path-based hierarchy where paths are embedded, enabling "semantic directory" searches.

2. **Document Entropy Scoring**: Calculate uniqueness score; link duplicates instead of re-indexing.

3. **Probabilistic Belonging**: Documents have 0.0-1.0 membership scores in multiple collections.

4. **Conversational File Injection**: RAG asks user where ambiguous documents should go.

5. **Dynamic Gravity Fields**: Collections defined by anchor vectors; documents auto-associate.

6. **Temporal Tag Decay**: Transient tags ("draft", "review-needed") lose weight over time.

---

## Next Steps

1. **Review & Approve**: Stakeholder review of this design
2. **Phase 0 Implementation**: QuerySpec contract (foundation)
3. **Phase 1 Implementation**: Collections (structural partition)
4. **Iterate**: Each phase builds on previous, evaluate before proceeding

**Estimated Effort**:
- Phase 0: 2-3 days
- Phase 1: 4-5 days
- Phase 2: 3-4 days
- Phase 3-5: 2-3 days each

**Total**: ~3-4 weeks for full implementation
