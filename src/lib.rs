pub mod config;
pub mod embeddings;
pub mod index_store;
pub mod job_manager;
pub mod job_payload;
mod mcp;
pub mod mcp_server;
pub mod pdf;
pub mod progress_logger;
pub mod rag_engine;
pub mod reranker;
pub mod worker;

// Re-export key structs for easier access
pub use config::Config;
pub use index_store::SqliteIndexStore;
pub use job_manager::JobManager;
pub use rag_engine::RagEngine;
pub use worker::{JobRequest, WorkerSupervisor};
