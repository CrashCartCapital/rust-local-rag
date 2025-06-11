use anyhow::Result;
use rmcp::{
    Error as McpError, ServerHandler, ServiceExt, model::*, schemars, tool, transport::stdio,
};

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::rag_engine::RagEngine;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(description = "The search query")]
    pub query: String,
    #[schemars(description = "Number of results to return (default: 5)")]
    pub top_k: Option<usize>,
}

#[derive(Clone)]
pub struct RagMcpServer {
    rag_state: Arc<RwLock<RagEngine>>,
}

#[tool(tool_box)]
impl RagMcpServer {
    pub fn new(rag_state: Arc<RwLock<RagEngine>>) -> Self {
        Self { rag_state }
    }

    #[tool(description = "Search through uploaded documents using semantic similarity")]
    async fn search_documents(
        &self,
        #[tool(aggr)] SearchRequest { query, top_k }: SearchRequest,
    ) -> Result<CallToolResult, McpError> {
        let top_k = top_k.unwrap_or(5);
        let engine = self.rag_state.read().await;

        match engine.search(&query, top_k).await {
            Ok(results) => {
                let formatted_results = if results.is_empty() {
                    "No results found.".to_string()
                } else {
                    results
                        .iter()
                        .enumerate()
                        .map(|(i, result)| {
                            format!(
                                "**Result {}** (Score: {:.3}) [{}] (Chunk: {})\n{}\n",
                                i + 1,
                                result.score,
                                result.document,
                                result.chunk_id,
                                result.text
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n---\n\n")
                };

                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Found {} results for '{}':\n\n{}",
                    results.len(),
                    query,
                    formatted_results
                ))]))
            }
            Err(e) => Ok(CallToolResult {
                content: vec![Content::text(format!("Search error: {}", e))],
                is_error: Some(true),
            }),
        }
    }

    #[tool(description = "List all uploaded documents")]
    async fn list_documents(&self) -> Result<CallToolResult, McpError> {
        let engine = self.rag_state.read().await;
        let documents = engine.list_documents();

        let response = if documents.is_empty() {
            "No documents uploaded yet.".to_string()
        } else {
            format!(
                "Uploaded documents ({}):\n{}",
                documents.len(),
                documents
                    .iter()
                    .enumerate()
                    .map(|(i, doc)| format!("{}. {}", i + 1, doc))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        };

        Ok(CallToolResult::success(vec![Content::text(response)]))
    }

    #[tool(description = "Get RAG system statistics")]
    async fn get_stats(&self) -> Result<CallToolResult, McpError> {
        let engine = self.rag_state.read().await;
        let stats = engine.get_stats();

        Ok(CallToolResult::success(vec![Content::text(format!(
            "RAG System Stats:\n{}",
            serde_json::to_string_pretty(&stats).unwrap()
        ))]))
    }
}

#[tool(tool_box)]
impl ServerHandler for RagMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "rust-rag-server".to_string(),
                version: "0.1.0".to_string(),
            },
            instructions: Some(
                "A Rust-based RAG server for document search and analysis.".to_string(),
            ),
        }
    }
}

pub async fn start_mcp_server(rag_state: Arc<RwLock<RagEngine>>) -> Result<()> {
    tracing::info!("Starting MCP server");

    let server = RagMcpServer::new(rag_state);
    let service = server.serve(stdio()).await?;

    service.waiting().await?;

    Ok(())
}
