use std::process::ExitCode;

use anyhow::Context;

#[derive(Debug)]
struct CheckResult {
    id: &'static str,
    ok: bool,
    required: bool,
    message: String,
    remediation: Option<String>,
}

impl CheckResult {
    fn pass(id: &'static str, required: bool, message: impl Into<String>) -> Self {
        Self {
            id,
            ok: true,
            required,
            message: message.into(),
            remediation: None,
        }
    }

    fn fail(
        id: &'static str,
        required: bool,
        message: impl Into<String>,
        remediation: impl Into<String>,
    ) -> Self {
        Self {
            id,
            ok: false,
            required,
            message: message.into(),
            remediation: Some(remediation.into()),
        }
    }

    fn print(&self) {
        if self.ok {
            println!("[PASS] {}: {}", self.id, self.message);
        } else {
            println!("[FAIL] {}: {}", self.id, self.message);
            if let Some(remediation) = &self.remediation {
                println!("       remediation: {remediation}");
            }
        }
    }
}

fn get_data_dir() -> String {
    std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string())
}

fn get_documents_dir() -> String {
    std::env::var("DOCUMENTS_DIR").unwrap_or_else(|_| "./documents".to_string())
}

fn get_ollama_url() -> String {
    std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn get_embedding_model() -> String {
    std::env::var("OLLAMA_EMBEDDING_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string())
}

fn get_rerank_model_if_configured() -> Option<String> {
    std::env::var("OLLAMA_RERANK_MODEL").ok()
}

async fn check_data_dir_and_db(data_dir: &str) -> CheckResult {
    let data_dir_path = std::path::Path::new(data_dir);
    if let Err(e) = std::fs::create_dir_all(data_dir_path) {
        return CheckResult::fail(
            "DATA_DIR",
            true,
            format!("Cannot create data dir at {data_dir}: {e}"),
            "Set DATA_DIR to a writable folder and rerun.",
        );
    }

    let db_path = format!("sqlite:{data_dir}/jobs.db");
    match rust_local_rag::JobManager::new(&db_path).await {
        Ok(_) => CheckResult::pass(
            "DATA_DIR",
            true,
            format!("Writable and jobs.db openable ({db_path})"),
        ),
        Err(e) => CheckResult::fail(
            "DATA_DIR",
            true,
            format!("jobs.db is not openable ({db_path}): {e}"),
            "Check filesystem permissions; if the DB is corrupt, move it aside and rerun.",
        ),
    }
}

fn check_documents_dir(documents_dir: &str) -> CheckResult {
    let documents_dir_path = std::path::Path::new(documents_dir);
    if !documents_dir_path.exists() {
        return CheckResult::fail(
            "DOCUMENTS_DIR",
            true,
            format!("Directory does not exist: {documents_dir}"),
            "Create the directory or set DOCUMENTS_DIR to an existing folder containing PDFs.",
        );
    }

    match std::fs::read_dir(documents_dir_path) {
        Ok(_) => CheckResult::pass("DOCUMENTS_DIR", true, format!("Readable ({documents_dir})")),
        Err(e) => CheckResult::fail(
            "DOCUMENTS_DIR",
            true,
            format!("Directory is not readable: {documents_dir}: {e}"),
            "Fix permissions or set DOCUMENTS_DIR to a readable folder.",
        ),
    }
}

fn check_pdftotext() -> CheckResult {
    match std::process::Command::new("pdftotext").arg("-v").output() {
        Ok(_) => CheckResult::pass("PDFTOTEXT", false, "pdftotext found in PATH"),
        Err(e) => CheckResult::fail(
            "PDFTOTEXT",
            false,
            format!("pdftotext not found: {e}"),
            "Install poppler (e.g., `brew install poppler`) for higher-quality PDF text extraction.",
        ),
    }
}

async fn fetch_ollama_tags(ollama_url: &str) -> anyhow::Result<serde_json::Value> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(8))
        .build()
        .context("Failed to build HTTP client")?;

    let base = ollama_url.trim_end_matches('/');
    let response = client
        .get(format!("{base}/api/tags"))
        .send()
        .await
        .context("Failed to contact Ollama")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Ollama /api/tags returned {status}: {body}");
    }

    Ok(response.json().await.context("Failed to parse Ollama /api/tags JSON")?)
}

#[tokio::main]
async fn main() -> ExitCode {
    dotenv::dotenv().ok();

    let mut checks: Vec<CheckResult> = Vec::new();

    let data_dir = get_data_dir();
    let documents_dir = get_documents_dir();
    let ollama_url = get_ollama_url();
    let embedding_model = get_embedding_model();
    let rerank_model = get_rerank_model_if_configured();

    checks.push(check_data_dir_and_db(&data_dir).await);
    checks.push(check_documents_dir(&documents_dir));
    checks.push(check_pdftotext());

    let mut tags: Option<serde_json::Value> = None;
    match rust_local_rag::guardrails::check_ollama_url(&ollama_url) {
        Ok(()) => match fetch_ollama_tags(&ollama_url).await {
            Ok(value) => {
                checks.push(CheckResult::pass(
                    "OLLAMA_URL",
                    true,
                    format!("Reachable ({ollama_url})"),
                ));
                tags = Some(value);
            }
            Err(e) => {
                checks.push(CheckResult::fail(
                    "OLLAMA_URL",
                    true,
                    format!("Unreachable ({ollama_url}): {e}"),
                    "Make sure Ollama is running and OLLAMA_URL points to it (try `ollama serve`).",
                ));
            }
        },
        Err(e) => {
            checks.push(CheckResult::fail(
                "OLLAMA_URL",
                true,
                format!("{e}"),
                "Use a loopback OLLAMA_URL (localhost/127.0.0.1) or set RAG_ALLOW_REMOTE_OLLAMA=1 to override.",
            ));
        }
    }

    match &tags {
        Some(tags) => {
            if rust_local_rag::doctor::model_prefix_exists(tags, &embedding_model) {
                checks.push(CheckResult::pass(
                    "OLLAMA_EMBEDDING_MODEL",
                    true,
                    format!("Model found: {embedding_model}"),
                ));
            } else {
                checks.push(CheckResult::fail(
                    "OLLAMA_EMBEDDING_MODEL",
                    true,
                    format!("Model not found: {embedding_model}"),
                    format!("Run: ollama pull {embedding_model}"),
                ));
            }

            if let Some(rerank_model) = rerank_model {
                if rust_local_rag::doctor::model_prefix_exists(tags, &rerank_model) {
                    checks.push(CheckResult::pass(
                        "OLLAMA_RERANK_MODEL",
                        true,
                        format!("Model found: {rerank_model}"),
                    ));
                } else {
                    checks.push(CheckResult::fail(
                        "OLLAMA_RERANK_MODEL",
                        true,
                        format!("Model not found: {rerank_model}"),
                        format!("Run: ollama pull {rerank_model}"),
                    ));
                }
            } else {
                checks.push(CheckResult::pass(
                    "OLLAMA_RERANK_MODEL",
                    true,
                    "Not configured (set OLLAMA_RERANK_MODEL to enable reranking)",
                ));
            }
        }
        None => {
            checks.push(CheckResult::fail(
                "OLLAMA_EMBEDDING_MODEL",
                true,
                "Cannot verify embedding model because OLLAMA_URL check failed.",
                "Fix OLLAMA_URL reachability first, then rerun.",
            ));
            if let Some(rerank_model) = rerank_model {
                checks.push(CheckResult::fail(
                    "OLLAMA_RERANK_MODEL",
                    true,
                    format!("Cannot verify reranker model '{rerank_model}' because OLLAMA_URL check failed."),
                    "Fix OLLAMA_URL reachability first, then rerun.",
                ));
            } else {
                checks.push(CheckResult::pass(
                    "OLLAMA_RERANK_MODEL",
                    true,
                    "Not configured (set OLLAMA_RERANK_MODEL to enable reranking)",
                ));
            }
        }
    }

    let required_failed = checks.iter().any(|c| c.required && !c.ok);
    let optional_failed = checks.iter().any(|c| !c.required && !c.ok);

    for check in &checks {
        check.print();
    }

    if required_failed {
        println!("Overall: FAIL");
        ExitCode::from(1)
    } else {
        if optional_failed {
            println!("Overall: PASS (with optional warnings)");
        } else {
            println!("Overall: PASS");
        }
        ExitCode::SUCCESS
    }
}

