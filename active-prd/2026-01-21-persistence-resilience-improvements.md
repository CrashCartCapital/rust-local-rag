# PRD: Persistence, Resilience & Configuration Hardening

**Project**: rust-local-rag
**Created**: 2026-01-21
**Status**: Draft
**Authors**: Claude Code (Sonnet 4.5) with CRASH MCP + Gemini 3 Pro analysis
**Based On**: STAT-REPORT-172612 + STAT-REPORT-180809 comprehensive audits

---

## Executive Summary

This PRD addresses the three highest-impact architectural improvements identified by dual comprehensive audits of the rust-local-rag codebase. These improvements eliminate data corruption risks, enable crash recovery, and prepare the system for incremental operation—all critical for a privacy-preserving local RAG system running on solo developer laptops.

**Impact**: Eliminates P0 data corruption risks, reduces reindex time by 90%+, enables graceful degradation

**Prioritization Rationale**:
- Both audit reports identified persistence atomicity as the #1 scariest risk
- Gemini peer review validated unified SQLite persistence as highest ROI (impact: 10/10)
- All improvements are pragmatic for solo developers (no distributed systems complexity)

---

## Problem Statement

### P0 Problems (Data Corruption & Loss)

**Problem 1: Vector-Relational Desynchronization**
**Evidence**: STAT-REPORT-180809 E6-E8, STAT-REPORT-172612 E39-E40

The system maintains two independent persistence layers:
- SQLite (`jobs.db`): Job metadata, status, progress
- JSON (`chunks_{model}.json`): Document chunks, embeddings, hashes

**Failure Mode**:
If the process crashes between updating JSON (step 3) and marking job completed in SQLite (step 4):
1. Job status shows "in progress" indefinitely
2. Chunks JSON may be partially written (corrupted or missing documents)
3. Document hashes diverge from actual file state
4. No recovery mechanism—user must manually delete chunks file and reindex (20+ minutes)

**User Impact**:
Solo developer loses 20+ minutes of indexing work on every laptop sleep, Ctrl+C, or OOM crash. Silent quality degradation (search returns outdated results) with no error message.

---

**Problem 2: No Job Crash Recovery**
**Evidence**: STAT-REPORT-180809 E73, STAT-REPORT-172612 (no checkpoint logic)

Background indexing jobs restart from scratch after any interruption:
- Laptop sleep/wake
- Process killed (OOM, Ctrl+C)
- Ollama service restart

**User Impact**:
For 85K chunk corpus (70 PDFs), a full reindex takes 20+ minutes. Every interruption means starting over. This is unacceptable for local-first software.

---

**Problem 3: No Embedding Model Validation**
**Evidence**: STAT-REPORT-180809 E25-E26, STAT-REPORT-172612 E17

System detects model changes (`needs_reindex=true`) but doesn't validate dimension compatibility:
- User switches from `nomic-embed-text` (768D) to `mxbai-embed-large` (1024D)
- Old chunks with 768D embeddings remain in index
- Search operations panic on dimension mismatch OR silently fail

**User Impact**:
Silent index corruption after model upgrades. No fail-fast validation on startup.

---

### P1 Problems (Operational Friction)

**Problem 4: Full Reindex on Every Change**
**Evidence**: STAT-REPORT-180809 E70, STAT-REPORT-172612 (no incremental sync)

Adding or updating a single PDF requires re-embedding the entire corpus:
- No change detection granularity below document level
- SHA-256 hashes exist but only used for "unchanged" detection, not selective updates
- No way to update individual documents without full reindex

**User Impact**:
20+ minute reindex to add a single 10-page PDF. Discourages iterative use.

---

## Requirements

### R1: Unified Atomic Persistence (P0)

**Functional Requirements**:
- FR1.1: All document chunks, embeddings, hashes, and job metadata MUST be stored in a single SQLite database
- FR1.2: Job status updates and chunk writes MUST occur within the same SQLite transaction
- FR1.3: On crash/rollback, system MUST restore to last consistent state (no partial updates)
- FR1.4: Existing JSON-based indexes MUST be migrated automatically on first startup with new backend
- FR1.5: PersistenceBackend trait MUST remain in rag-core for future extensibility

**Non-Functional Requirements**:
- NFR1.1: Read performance MUST NOT degrade more than 10% vs current JSON approach
- NFR1.2: Migration from JSON to SQLite MUST complete within 5 minutes for 100K chunks
- NFR1.3: SQLite file size MUST NOT exceed 2x the current JSON file size

**Success Criteria**:
- Zero data corruption events in crash simulation tests (kill -9 during indexing)
- Job status always reflects actual index state (no orphaned "in progress" jobs)
- Backward compatibility: existing deployments upgrade seamlessly

---

### R2: Job Checkpointing & Resume (P0)

**Functional Requirements**:
- FR2.1: Background jobs MUST save progress checkpoints every N documents (configurable, default: 10)
- FR2.2: On startup, WorkerSupervisor MUST detect incomplete jobs and offer to resume
- FR2.3: Resumed jobs MUST skip already-processed documents (idempotent)
- FR2.4: Checkpoint data MUST be persisted in SQLite alongside job metadata

**Non-Functional Requirements**:
- NFR2.1: Checkpoint overhead MUST NOT add more than 5% to total indexing time
- NFR2.2: Resumed jobs MUST complete 90%+ faster than full reindex for 50% complete jobs

**Success Criteria**:
- Kill process at 50% completion → Resume → Completes in <10% of original time
- No duplicate chunks in index after resume (deterministic chunk IDs)

---

### R3: Embedding Configuration Validation (P0)

**Functional Requirements**:
- FR3.1: On startup, MUST verify Ollama is reachable at configured URL
- FR3.2: On startup, MUST verify embedding model exists in Ollama
- FR3.3: On startup, MUST verify embedding dimension matches existing index (if index exists)
- FR3.4: If validation fails, MUST prevent server start with clear error message
- FR3.5: MUST store a "canary embedding" (fixed input text) to detect upstream model changes

**Non-Functional Requirements**:
- NFR3.1: Startup validation MUST complete within 5 seconds
- NFR3.2: Error messages MUST include actionable remediation steps

**Success Criteria**:
- Starting with wrong model fails with error: "Dimension mismatch: index expects 768D, model provides 1024D. Run reindex or switch to compatible model."
- Canary embedding drift >5% triggers warning: "Embedding model weights may have changed. Reindex recommended."

---

### R4: Incremental Document Sync (P1)

**Functional Requirements**:
- FR4.1: MUST support updating individual documents without full reindex
- FR4.2: MUST detect added, modified, and deleted documents via SHA-256 hash comparison
- FR4.3: MUST remove chunks for deleted documents from index
- FR4.4: MUST rebuild AnnIndex and LexicalIndex incrementally (add/remove entries, not full rebuild)
- FR4.5: MUST provide MCP tool `sync_documents` as alternative to `start_reindex`

**Non-Functional Requirements**:
- NFR4.1: Syncing 5 changed documents (out of 70) MUST complete in <2 minutes
- NFR4.2: Index quality (Hit Rate@5) MUST NOT degrade vs full reindex

**Success Criteria**:
- Adding 1 new PDF: sync completes in <1 minute (vs 20+ min full reindex)
- Modifying 3 PDFs: only those 3 are re-embedded
- Deleting 2 PDFs: their chunks are removed, searches no longer return them

---

## Design

### D1: Unified SQLite Persistence Backend

**Architecture Decision**: Implement `SqlitePersistenceBackend` as alternative to `JsonFileBackend`, leveraging existing `PersistenceBackend` trait.

**Schema Design**:

```sql
-- Chunks table: stores document chunks with embeddings
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY NOT NULL,
    document_name TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- f32 array serialized via bincode
    page_number INTEGER,
    section TEXT,
    metadata TEXT,  -- JSON for extensibility
    created_at INTEGER NOT NULL,
    INDEX idx_chunks_document (document_name)
);

-- Document hashes table: for change detection
CREATE TABLE document_hashes (
    document_name TEXT PRIMARY KEY NOT NULL,
    hash TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

-- Engine metadata table: single row for model config
CREATE TABLE engine_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Enforce single row
    schema_version INTEGER NOT NULL,
    embedding_model_id TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    needs_reindex INTEGER NOT NULL DEFAULT 0,
    canary_embedding BLOB,  -- For drift detection
    updated_at INTEGER NOT NULL
);

-- Reuse existing jobs table from JobManager
-- (Already has: job_id, status, job_type, payload, progress, total, error, started_at, updated_at)
```

**Transaction Boundaries**:

```rust
// Atomic job completion with chunk persistence
async fn complete_job_with_chunks(
    tx: &mut SqliteTransaction,
    job_id: &str,
    chunks: Vec<DocumentChunk>,
) -> Result<()> {
    // 1. Insert/update chunks
    for chunk in chunks {
        sqlx::query("INSERT OR REPLACE INTO chunks (...) VALUES (...)")
            .execute(&mut *tx)
            .await?;
    }

    // 2. Update job status to completed
    sqlx::query("UPDATE jobs SET status = 'completed', updated_at = ? WHERE job_id = ?")
        .execute(&mut *tx)
        .await?;

    // 3. Commit transaction (both succeed or both rollback)
    tx.commit().await?;
    Ok(())
}
```

**Migration Strategy**:

```rust
// On first startup with SqlitePersistenceBackend:
// 1. Check if chunks_{model}.json exists
// 2. If yes, load JSON state
// 3. Insert all chunks into SQLite via transaction
// 4. Rename JSON to chunks_{model}.json.migrated (keep backup)
// 5. Log migration completion
```

**Implementation Files**:
- `crates/rag-core/src/persistence/sqlite.rs` (new file, ~400 LOC)
- `src/rag_engine.rs` (modify: inject SqlitePersistenceBackend)
- `crates/rag-core/src/persistence.rs` (add trait helper methods for transactions)

---

### D2: Job Checkpointing

**Checkpoint Data Structure**:

```rust
#[derive(Serialize, Deserialize)]
struct ReindexCheckpoint {
    job_id: String,
    processed_documents: HashSet<String>,  // SHA-256 hashes
    total_documents: usize,
    last_checkpoint_at: i64,
}
```

**Checkpoint Strategy**:
- Save checkpoint every 10 documents (configurable via `RAG_CHECKPOINT_INTERVAL`)
- Store as JSON in `jobs.payload` column (already exists)
- On resume: skip documents in `processed_documents` set

**Resume Logic**:

```rust
// WorkerSupervisor::run() startup sequence
async fn run(&self) -> Result<()> {
    // Find jobs with status = InProgress
    let resumable_jobs = self.job_manager.find_resumable_jobs().await?;

    for job in resumable_jobs {
        // Parse checkpoint from payload
        if let Some(checkpoint) = parse_checkpoint(&job.payload) {
            tracing::info!("Resuming job {} from checkpoint ({}/{} docs processed)",
                job.job_id, checkpoint.processed_documents.len(), checkpoint.total_documents);

            // Send resume request to worker with checkpoint context
            self.job_tx.send(JobRequest::Resume { job_id: job.job_id, checkpoint }).await?;
        }
    }

    // Continue with normal event loop
    // ...
}
```

**Implementation Files**:
- `src/worker.rs` (modify: add checkpoint save logic, resume handling)
- `src/job_manager.rs` (modify: add `find_resumable_jobs()` method)

---

### D3: Embedding Configuration Validation

**Validation Sequence**:

```rust
// In RagEngine::new() or server startup
async fn validate_embedding_config(backend: &impl EmbeddingBackend) -> Result<()> {
    // 1. Health check: Verify Ollama is reachable
    backend.health_check().await
        .map_err(|e| anyhow!("Ollama unreachable at {}: {}", backend.url(), e))?;

    // 2. Model availability: Verify model exists
    backend.verify_model_exists().await
        .map_err(|e| anyhow!("Embedding model '{}' not found in Ollama. Run: ollama pull {}",
            backend.model_id(), backend.model_id()))?;

    // 3. Dimension validation: Check against existing index
    if let Some(existing_state) = persistence_backend.load()? {
        let expected_dim = existing_state.embedding_dim;
        let actual_dim = backend.dimension();

        if expected_dim != actual_dim {
            return Err(anyhow!(
                "Embedding dimension mismatch:\n  \
                 Index expects: {}D (model: {})\n  \
                 Current model: {}D (model: {})\n\n  \
                 Options:\n  \
                 - Switch back to compatible model\n  \
                 - Run full reindex with new model",
                expected_dim, existing_state.embedding_model_id,
                actual_dim, backend.model_id()
            ));
        }
    }

    // 4. Canary embedding drift check (if index exists)
    if let Some(existing_state) = persistence_backend.load()? {
        const CANARY_TEXT: &str = "The quick brown fox jumps over the lazy dog";
        let current_embedding = backend.embed(CANARY_TEXT).await?;

        if let Some(stored_canary) = existing_state.canary_embedding {
            let similarity = cosine_similarity(&current_embedding, &stored_canary);
            if similarity < 0.95 {
                tracing::warn!(
                    "Canary embedding drift detected (similarity: {:.3}). \
                     Model weights may have changed. Reindex recommended.",
                    similarity
                );
            }
        } else {
            // First run: store canary for future drift detection
            // (save via persistence backend)
        }
    }

    Ok(())
}
```

**EmbeddingBackend Trait Extension**:

```rust
// Add to crates/rag-core/src/traits.rs
pub trait EmbeddingBackend: Send + Sync {
    // ... existing methods ...

    /// Health check: verify backend is reachable
    async fn health_check(&self) -> Result<(), EmbeddingError>;

    /// Verify model exists in backend
    async fn verify_model_exists(&self) -> Result<(), EmbeddingError>;

    /// Get backend URL for error messages
    fn url(&self) -> &str;
}
```

**Implementation Files**:
- `src/embeddings.rs` (add health_check, verify_model_exists methods)
- `crates/rag-core/src/traits.rs` (extend EmbeddingBackend trait)
- `src/main.rs` (add validation call in startup sequence)

---

### D4: Incremental Document Sync

**Change Detection Algorithm**:

```rust
async fn detect_document_changes(
    documents_dir: &Path,
    existing_hashes: &HashMap<String, String>,
) -> Result<DocumentChangeset> {
    let mut changeset = DocumentChangeset {
        added: vec![],
        modified: vec![],
        deleted: vec![],
    };

    // Scan filesystem for current documents
    let current_docs = discover_pdf_paths(documents_dir)?;
    let current_hashes: HashMap<String, String> = current_docs
        .iter()
        .map(|path| {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let hash = compute_file_hash(path)?;
            Ok((name, hash))
        })
        .collect::<Result<_>>()?;

    // Detect added and modified
    for (name, hash) in &current_hashes {
        match existing_hashes.get(name) {
            None => changeset.added.push(name.clone()),
            Some(old_hash) if old_hash != hash => changeset.modified.push(name.clone()),
            Some(_) => {} // Unchanged
        }
    }

    // Detect deleted
    for name in existing_hashes.keys() {
        if !current_hashes.contains_key(name) {
            changeset.deleted.push(name.clone());
        }
    }

    Ok(changeset)
}
```

**Incremental Index Update**:

```rust
async fn sync_documents(
    rag_engine: &mut RagEngine,
    changeset: DocumentChangeset,
) -> Result<SyncSummary> {
    let mut summary = SyncSummary::default();

    // Remove deleted documents
    for doc_name in &changeset.deleted {
        let removed_count = rag_engine.remove_document(doc_name).await?;
        summary.removed_chunks += removed_count;
    }

    // Update modified documents (remove old + add new)
    for doc_name in &changeset.modified {
        rag_engine.remove_document(doc_name).await?;
        let chunks = rag_engine.prepare_document(doc_name, /*...*/).await?;
        rag_engine.upsert_prepared_document(chunks).await?;
        summary.updated_docs += 1;
    }

    // Add new documents
    for doc_name in &changeset.added {
        let chunks = rag_engine.prepare_document(doc_name, /*...*/).await?;
        rag_engine.upsert_prepared_document(chunks).await?;
        summary.added_docs += 1;
    }

    // Rebuild indexes incrementally (if backend supports it)
    // For initial impl: full rebuild of AnnIndex/LexicalIndex (fast for small changes)
    rag_engine.rebuild_indexes().await?;

    Ok(summary)
}
```

**MCP Tool Addition**:

```rust
#[tool(description = "Incrementally sync document changes (faster than full reindex)")]
async fn sync_documents(&self) -> Result<CallToolResult, McpError> {
    // Similar to start_reindex, but creates JobType::Sync instead
    // Worker detects changes and applies incremental updates
    // ...
}
```

**Implementation Files**:
- `src/worker.rs` (add sync_documents_incremental function)
- `crates/rag-core/src/engine.rs` (add remove_document, rebuild_indexes methods)
- `src/mcp_server.rs` (add sync_documents MCP tool)
- `src/job_manager.rs` (add JobType::Sync variant)

---

## Implementation Tasks

### Phase 1: Unified SQLite Persistence (P0)

**Task 1.1: Design SQLite Schema**
- [ ] Create SQL migration file: `migrations/001_initial_schema.sql`
- [ ] Define chunks, document_hashes, engine_metadata tables
- [ ] Add indexes for performance (idx_chunks_document)
- [ ] Validate schema with SQLite CLI

**Task 1.2: Implement SqlitePersistenceBackend**
- [ ] Create `crates/rag-core/src/persistence/sqlite.rs`
- [ ] Implement PersistenceBackend trait methods (save, load)
- [ ] Add transaction helpers (save_chunks_atomic, update_metadata)
- [ ] Implement embedding serialization (f32 array → BLOB via bincode)
- [ ] Add connection pooling configuration (reuse JobManager pool)

**Task 1.3: JSON → SQLite Migration**
- [ ] Detect existing chunks_{model}.json on startup
- [ ] Load JSON state via JsonFileBackend
- [ ] Insert all data into SQLite via single transaction
- [ ] Rename JSON to .migrated (keep backup)
- [ ] Add migration logging + progress indication

**Task 1.4: Integrate with JobManager**
- [ ] Modify worker.rs to use transactions for job completion
- [ ] Ensure chunk inserts + job status update are atomic
- [ ] Add rollback handling on worker failure
- [ ] Update rag_engine.rs to inject SqlitePersistenceBackend

**Task 1.5: Testing**
- [ ] Unit test: SqlitePersistenceBackend save/load round-trip
- [ ] Integration test: crash simulation (kill -9 during save)
- [ ] Migration test: JSON → SQLite with 10K chunks
- [ ] Verify no orphaned jobs after crashes

**Acceptance Criteria**:
- [x] All chunks, hashes, metadata stored in SQLite
- [x] Job status and chunk writes are atomic (single transaction)
- [x] Crash recovery leaves system in consistent state
- [x] JSON migration completes successfully
- [x] Backward compatibility maintained

---

### Phase 2: Job Checkpointing (P0)

**Task 2.1: Checkpoint Data Model**
- [ ] Define ReindexCheckpoint struct in worker.rs
- [ ] Add serialization (serde JSON) for jobs.payload column
- [ ] Define checkpoint interval constant (default: 10 docs)
- [ ] Add configuration: RAG_CHECKPOINT_INTERVAL env var

**Task 2.2: Worker Checkpoint Logic**
- [ ] Modify reindex_documents() to save checkpoints every N docs
- [ ] Store checkpoint in jobs.payload via job_manager.update_payload()
- [ ] Add processed_documents HashSet to track completed work
- [ ] Ensure checkpoint saves are atomic (within job update transaction)

**Task 2.3: Resume Logic**
- [ ] Add WorkerSupervisor::find_resumable_jobs() call on startup
- [ ] Parse checkpoint from job.payload
- [ ] Skip processed documents during resume
- [ ] Log resume progress ("Resuming job X from 50/100 documents")

**Task 2.4: Testing**
- [ ] Test: kill job at 25%, 50%, 75% completion → resume → verify completion
- [ ] Test: no duplicate chunks after resume (idempotent)
- [ ] Test: checkpoint interval configuration (1, 10, 50 docs)
- [ ] Verify resume is faster than full reindex (90%+ speedup)

**Acceptance Criteria**:
- [x] Jobs save progress checkpoints every N documents
- [x] Resumed jobs skip already-processed documents
- [x] Resume completes in <10% of full reindex time
- [x] No data corruption or duplication after resume

---

### Phase 3: Embedding Validation (P0)

**Task 3.1: EmbeddingBackend Trait Extension**
- [ ] Add health_check() method to trait
- [ ] Add verify_model_exists() method to trait
- [ ] Add url() method to trait
- [ ] Implement methods in OllamaEmbeddingService

**Task 3.2: Validation Logic**
- [ ] Create validate_embedding_config() function
- [ ] Add dimension mismatch check vs existing index
- [ ] Add canary embedding drift detection (cosine similarity <0.95)
- [ ] Format error messages with actionable remediation

**Task 3.3: Integration + Testing**
- [ ] Call validation in server startup (main.rs)
- [ ] Add canary storage to engine_metadata table
- [ ] Test: wrong model fails with clear error
- [ ] Test: unreachable Ollama fails with clear error
- [ ] Test: dimension mismatch prevents startup

**Acceptance Criteria**:
- [x] Startup validates Ollama connectivity
- [x] Startup validates model exists and dimension matches
- [x] Canary drift detection warns on model weight changes
- [x] Error messages include remediation steps

---

### Phase 4: Incremental Sync (P1)

**Task 4.1: Change Detection**
- [ ] Implement detect_document_changes() in worker.rs
- [ ] Compare filesystem SHA-256 hashes vs existing document_hashes
- [ ] Classify documents as added, modified, deleted
- [ ] Return DocumentChangeset struct

**Task 4.2: Incremental Update Methods**
- [ ] Add RagEngine::remove_document() method
- [ ] Add RagEngine::rebuild_indexes() method (incremental if possible)
- [ ] Update AnnIndex to support remove operation
- [ ] Update LexicalIndex to support remove operation

**Task 4.3: Sync Job Implementation**
- [ ] Add JobType::Sync variant
- [ ] Implement sync_documents_incremental() in worker.rs
- [ ] Call detect_document_changes(), apply changeset
- [ ] Log sync summary (X added, Y modified, Z removed)

**Task 4.4: MCP Tool + Testing**
- [ ] Add sync_documents MCP tool in mcp_server.rs
- [ ] Test: add 1 PDF → sync completes in <1 min
- [ ] Test: modify 3 PDFs → only those 3 re-embedded
- [ ] Test: delete 2 PDFs → chunks removed, not in search results
- [ ] Verify Hit Rate@5 unchanged vs full reindex

**Acceptance Criteria**:
- [x] Syncing individual documents completes in <2 minutes
- [x] Deleted documents removed from index
- [x] Search quality (Hit Rate@5) unchanged vs full reindex
- [x] MCP tool sync_documents available to Claude Desktop

---

## Testing Strategy

### Unit Tests
- SqlitePersistenceBackend: save/load round-trip
- Checkpoint serialization/deserialization
- Change detection algorithm (added/modified/deleted)
- Embedding validation logic

### Integration Tests
- **Crash Recovery**: Kill process during indexing → restart → verify consistency
- **Resume**: Kill at 50% → resume → verify no duplicates, faster completion
- **Migration**: JSON → SQLite with 10K chunks → verify all data present
- **Incremental Sync**: Add/modify/delete PDFs → sync → verify correct chunks

### Acceptance Tests
- **End-to-End**: Full workflow with crashes, resumes, syncs
- **Eval Framework**: Run baseline eval after all changes → Hit Rate@5 ≥ 77.8%
- **Performance**: Measure indexing, search, sync latencies vs baseline

---

## Rollout Strategy

### Phase 1: Unified SQLite Persistence
**Rollout**: Feature flag `USE_SQLITE_PERSISTENCE` (default: false for first release)
**Dependencies**: None

### Phase 2: Job Checkpointing
**Rollout**: Enabled by default (automatically resumes jobs on startup)
**Dependencies**: Phase 1 (SQLite persistence required for atomic checkpoints)

### Phase 3: Embedding Validation
**Rollout**: Enabled by default (fail-fast on misconfiguration)
**Dependencies**: None (can be deployed independently)

### Phase 4: Incremental Sync
**Rollout**: New MCP tool available, manual invocation (gradual adoption)
**Dependencies**: Phase 1 (SQLite persistence required for efficient change detection)

---

## Success Metrics

### Pre-Implementation Baseline
- **Indexing Time**: 20+ minutes for 70 PDFs (85K chunks)
- **Crash Recovery**: Manual intervention required (delete chunks.json, reindex)
- **Model Switch**: Silent corruption risk
- **Document Update**: Full reindex (20+ minutes)

### Post-Implementation Targets
- **Indexing Time**: Unchanged (20+ minutes for full reindex)
- **Crash Recovery**: Automatic resume, <2 minutes to complete from 50% checkpoint
- **Model Switch**: Fail-fast with clear error message
- **Document Update**: Incremental sync, <1 minute for single PDF

### Quality Gates
- [ ] Zero data corruption in crash simulation tests (20 iterations)
- [ ] Job resume success rate: 100%
- [ ] Eval framework Hit Rate@5: ≥ 77.8% (no degradation)
- [ ] Incremental sync latency: <2 minutes for 5 changed documents

---

## Risk Mitigation

### Risk 1: SQLite Performance Degradation
**Mitigation**:
- Use BLOB for embeddings (efficient binary storage)
- Add indexes on document_name for fast lookups
- Enable WAL mode for concurrent reads during writes
- Benchmark read performance vs JSON (accept <10% degradation)

### Risk 2: Migration Failures
**Mitigation**:
- Keep JSON backups (.migrated files)
- Add rollback mechanism (delete SQLite, restore JSON)
- Test migration with production-scale data (100K chunks)

### Risk 3: Resume Logic Bugs (Duplicate Chunks)
**Mitigation**:
- Use deterministic chunk IDs (hash of document + chunk index)
- Add unique constraint on chunks.chunk_id
- Verify no duplicates in integration tests

### Risk 4: Incremental Sync Quality Degradation
**Mitigation**:
- Run full eval framework after sync operations
- Compare Hit Rate@5 vs full reindex baseline
- Add regression tests for known queries

---

## Open Questions

1. **SQLite File Location**: Should chunks.db be separate from jobs.db or unified?
   - **Recommendation**: Unified database for atomic transactions. Single file simplifies backup/restore.

2. **Embedding Compression**: Should we compress f32 arrays (e.g., zstd) for storage?
   - **Recommendation**: Defer to future optimization. Measure SQLite file size first.

3. **Canary Text**: Should canary be configurable or hardcoded?
   - **Recommendation**: Hardcoded for consistency. Document in CLAUDE.md.

4. **Incremental AnnIndex Rebuild**: Full rebuild or incremental add/remove?
   - **Recommendation**: Start with full rebuild (fast for <10K deltas). Optimize later if needed.

---

## Appendix

### A1: Related Audit Findings

**STAT-REPORT-172612**:
- E8, E9: In-memory state monolith (Arc<RwLock>)
- E39, E40: Corruption recovery missing
- E17: Embedding drift detection missing

**STAT-REPORT-180809**:
- E6-E8: Vector-relational desynchronization
- E70: No incremental sync
- E73: No job resume
- E25-E26: No model validation

**Gemini Peer Review**:
- Unified SQLite: Impact 10/10
- Job Checkpointing: Impact 8/10
- Ollama Validation: Impact 7/10
- Sharded Locking (DashMap): Impact 6/10 (deferred to future)

---

### A2: Alternative Approaches Considered

**Alternative 1: Distributed Locking (e.g., etcd, Redis)**
- **Rejected**: Over-engineered for solo dev use case. Adds external dependency.

**Alternative 2: Full Migration to Vector DB (Qdrant, Milvus)**
- **Rejected**: Breaks privacy-first goal (requires external service). Future consideration for scale.

**Alternative 3: DashMap for Sharded Locking**
- **Deferred**: Tactical fix for contention but doesn't solve OOM or atomicity. Consider in Phase 5.

**Alternative 4: Incremental AnnIndex (LSH sharding)**
- **Deferred**: Complex implementation. Full rebuild is acceptable for <10K changes.

---

## Conclusion

This PRD addresses the three highest-impact architectural improvements for rust-local-rag, validated by dual comprehensive audits and Gemini peer review. The proposed changes eliminate data corruption risks, enable graceful crash recovery, and prepare the system for incremental operation—all critical capabilities for a privacy-preserving local RAG system.

**Key Outcomes**:
1. **Data Safety**: Atomic transactions eliminate desynchronization risk
2. **Operational Resilience**: Job checkpointing enables crash recovery
3. **Configuration Hardening**: Fail-fast validation prevents silent corruption
4. **Incremental Efficiency**: Sync enables <2 minute updates vs 20+ minute reindexes

**Next Steps**:
1. Review PRD with stakeholder (solo dev = self-review)
2. Begin Phase 1 implementation (Unified SQLite Persistence)
3. Validate with crash recovery tests
4. Iterate based on lessons learned

---

**Document Status**: Ready for Implementation
**Review Date**: 2026-01-21
**Approver**: Self (Solo Developer)
