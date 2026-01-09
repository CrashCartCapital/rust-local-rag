use crate::job_manager::Job;

#[derive(Debug, serde::Serialize)]
pub(crate) struct ReindexResponse {
    pub(crate) job_id: String,
    pub(crate) status: String,
    pub(crate) documents_dir: String,
    pub(crate) message: String,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct JobStatusResponse {
    pub(crate) job_id: String,
    pub(crate) status: String,
    pub(crate) job_type: String,
    pub(crate) progress: i64,
    pub(crate) total: i64,
    pub(crate) error: Option<String>,
    pub(crate) started_at: i64,
    pub(crate) updated_at: i64,
}

impl From<&Job> for JobStatusResponse {
    fn from(job: &Job) -> Self {
        Self {
            job_id: job.job_id.clone(),
            status: job.status.as_str().to_string(),
            job_type: job.job_type.as_str().to_string(),
            progress: job.progress,
            total: job.total,
            error: job.error.clone(),
            started_at: job.started_at,
            updated_at: job.updated_at,
        }
    }
}

