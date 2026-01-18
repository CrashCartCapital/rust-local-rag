pub mod config;
pub mod embeddings;
pub mod error;
pub mod job_manager;
mod mcp;
pub mod mcp_server;
pub mod progress_logger;
pub mod rag_engine;
pub mod reranker;
pub mod worker;

// Re-export key structs for easier access
pub use config::Config;
pub use error::RagError;
pub use job_manager::JobManager;
pub use rag_engine::RagEngine;
pub use worker::{JobRequest, WorkerSupervisor};
