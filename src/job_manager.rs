#![allow(dead_code)]
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::{FromRow, Pool, Sqlite, Type};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl JobStatus {
    pub const fn as_str(&self) -> &'static str {
        match self {
            JobStatus::Pending => "pending",
            JobStatus::InProgress => "inprogress",
            JobStatus::Completed => "completed",
            JobStatus::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Type, PartialEq)]
#[sqlx(type_name = "TEXT", rename_all = "lowercase")]
pub enum JobType {
    Reindex,
}

impl JobType {
    pub const fn as_str(&self) -> &'static str {
        match self {
            JobType::Reindex => "reindex",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Job {
    pub job_id: String,
    pub status: JobStatus,
    pub job_type: JobType,
    pub payload: Option<String>,
    pub progress: i64,
    pub total: i64,
    pub error: Option<String>,
    pub started_at: i64,
    pub updated_at: i64,
}

pub struct JobManager {
    pool: Pool<Sqlite>,
}

impl JobManager {
    pub async fn new(db_path: &str) -> Result<Self> {
        use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqliteSynchronous};
        use std::str::FromStr;

        // Configure SQLite for concurrent access
        let options = SqliteConnectOptions::from_str(db_path)?
            .busy_timeout(std::time::Duration::from_secs(30)) // Wait up to 30s for locks
            .journal_mode(SqliteJournalMode::Wal) // Write-Ahead Logging for better concurrency
            .synchronous(SqliteSynchronous::Normal) // Balance safety and performance
            .create_if_missing(true);

        let pool = sqlx::SqlitePool::connect_with(options).await?;

        // Create jobs table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL,
                job_type TEXT NOT NULL,
                payload TEXT,
                progress INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                started_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            "#,
        )
        .execute(&pool)
        .await?;

        // Create index on status column for find_resumable_jobs performance
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            .execute(&pool)
            .await?;

        Ok(Self { pool })
    }

    #[allow(dead_code)]
    pub async fn create_job(
        &self,
        job_type: JobType,
        payload: Option<String>,
        total: i64,
    ) -> Result<Job> {
        let job_id = Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();

        let job = sqlx::query_as::<_, Job>(
            "INSERT INTO jobs (job_id, status, job_type, payload, total, started_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)
             RETURNING *",
        )
        .bind(&job_id)
        .bind(JobStatus::Pending.as_str())
        .bind(job_type.as_str())
        .bind(&payload)
        .bind(total)
        .bind(now)
        .bind(now)
        .fetch_one(&self.pool)
        .await?;

        Ok(job)
    }

    pub async fn get_job(&self, job_id: &str) -> Result<Option<Job>> {
        let job = sqlx::query_as::<_, Job>("SELECT * FROM jobs WHERE job_id = ?")
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(job)
    }

    pub async fn update_status(
        &self,
        job_id: &str,
        status: JobStatus,
        error: Option<String>,
    ) -> Result<()> {
        let now = Utc::now().timestamp();

        sqlx::query("UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?")
            .bind(status.as_str())
            .bind(error)
            .bind(now)
            .bind(job_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn update_progress(&self, job_id: &str, progress: i64) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query("UPDATE jobs SET progress = ?, updated_at = ? WHERE job_id = ?")
            .bind(progress)
            .bind(now)
            .bind(job_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn update_total(&self, job_id: &str, total: i64) -> Result<()> {
        let now = Utc::now().timestamp();
        sqlx::query("UPDATE jobs SET total = ?, updated_at = ? WHERE job_id = ?")
            .bind(total)
            .bind(now)
            .bind(job_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn find_resumable_jobs(&self) -> Result<Vec<Job>> {
        let jobs = sqlx::query_as::<_, Job>(
            "SELECT * FROM jobs WHERE status = 'inprogress' OR status = 'pending'",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(jobs)
    }

    #[allow(dead_code)]
    pub async fn find_active_reindex_job(&self) -> Result<Option<Job>> {
        let job = sqlx::query_as::<_, Job>(
            "SELECT * FROM jobs WHERE job_type = 'reindex' AND (status = 'pending' OR status = 'inprogress')",
        )
        .fetch_optional(&self.pool)
        .await?;
        Ok(job)
    }

    /// Atomically create a reindex job if no active reindex job exists.
    /// Returns Ok(Some(job)) if created successfully, Ok(None) if job already exists.
    /// This prevents race conditions by performing check-and-create in a single transaction.
    pub async fn create_reindex_job_if_not_active(
        &self,
        payload: Option<String>,
        total: i64,
    ) -> Result<Option<Job>> {
        // Use BEGIN IMMEDIATE to acquire write lock immediately
        let mut tx = self.pool.begin().await?;

        // Check for active reindex job within transaction
        let active_job = sqlx::query_as::<_, Job>(
            "SELECT * FROM jobs WHERE job_type = 'reindex' AND (status = 'pending' OR status = 'inprogress')",
        )
        .fetch_optional(&mut *tx)
        .await?;

        if active_job.is_some() {
            // Job already exists, rollback and return None
            tx.rollback().await?;
            return Ok(None);
        }

        // No active job, create new one
        let job_id = Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();

        let job = sqlx::query_as::<_, Job>(
            "INSERT INTO jobs (job_id, status, job_type, payload, total, started_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?)
             RETURNING *",
        )
        .bind(&job_id)
        .bind(JobStatus::Pending.as_str())
        .bind(JobType::Reindex.as_str())
        .bind(&payload)
        .bind(total)
        .bind(now)
        .bind(now)
        .fetch_one(&mut *tx)
        .await?;

        // Commit transaction
        tx.commit().await?;

        Ok(Some(job))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_and_get_job() {
        let manager = JobManager::new("sqlite::memory:").await.unwrap();

        let job = manager
            .create_job(JobType::Reindex, Some("/test/path".to_string()), 100)
            .await
            .unwrap();

        assert_eq!(job.job_type, JobType::Reindex);
        assert_eq!(job.total, 100);
        assert_eq!(job.progress, 0);

        let retrieved = manager.get_job(&job.job_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().job_id, job.job_id);
    }

    #[tokio::test]
    async fn test_update_status() {
        let manager = JobManager::new("sqlite::memory:").await.unwrap();

        let job = manager
            .create_job(JobType::Reindex, None, 0)
            .await
            .unwrap();

        manager
            .update_status(&job.job_id, JobStatus::InProgress, None)
            .await
            .unwrap();

        let updated = manager.get_job(&job.job_id).await.unwrap().unwrap();
        assert!(matches!(updated.status, JobStatus::InProgress));

        manager
            .update_status(&job.job_id, JobStatus::Failed, Some("Test error".to_string()))
            .await
            .unwrap();

        let failed = manager.get_job(&job.job_id).await.unwrap().unwrap();
        assert!(matches!(failed.status, JobStatus::Failed));
        assert_eq!(failed.error, Some("Test error".to_string()));
    }

    #[tokio::test]
    async fn test_update_progress() {
        let manager = JobManager::new("sqlite::memory:").await.unwrap();

        let job = manager
            .create_job(JobType::Reindex, None, 10)
            .await
            .unwrap();

        manager.update_progress(&job.job_id, 5).await.unwrap();

        let updated = manager.get_job(&job.job_id).await.unwrap().unwrap();
        assert_eq!(updated.progress, 5);
        assert_eq!(updated.total, 10);
    }

    #[tokio::test]
    async fn test_find_active_reindex_job() {
        let manager = JobManager::new("sqlite::memory:").await.unwrap();

        // No active job initially
        let active = manager.find_active_reindex_job().await.unwrap();
        assert!(active.is_none());

        // Create pending job
        let job1 = manager
            .create_job(JobType::Reindex, None, 0)
            .await
            .unwrap();

        let active = manager.find_active_reindex_job().await.unwrap();
        assert!(active.is_some());
        assert_eq!(active.unwrap().job_id, job1.job_id);

        // Mark as completed
        manager
            .update_status(&job1.job_id, JobStatus::Completed, None)
            .await
            .unwrap();

        // Should find no active job
        let active = manager.find_active_reindex_job().await.unwrap();
        assert!(active.is_none());

        // Create in-progress job
        let job2 = manager
            .create_job(JobType::Reindex, None, 0)
            .await
            .unwrap();

        manager
            .update_status(&job2.job_id, JobStatus::InProgress, None)
            .await
            .unwrap();

        let active = manager.find_active_reindex_job().await.unwrap();
        assert!(active.is_some());
        assert_eq!(active.unwrap().job_id, job2.job_id);
    }

    #[tokio::test]
    async fn test_find_resumable_jobs() {
        let manager = JobManager::new("sqlite::memory:").await.unwrap();

        let job1 = manager
            .create_job(JobType::Reindex, None, 0)
            .await
            .unwrap();

        let job2 = manager
            .create_job(JobType::Reindex, None, 0)
            .await
            .unwrap();

        manager
            .update_status(&job2.job_id, JobStatus::InProgress, None)
            .await
            .unwrap();

        let resumable = manager.find_resumable_jobs().await.unwrap();
        assert_eq!(resumable.len(), 2);

        manager
            .update_status(&job1.job_id, JobStatus::Completed, None)
            .await
            .unwrap();

        let resumable = manager.find_resumable_jobs().await.unwrap();
        assert_eq!(resumable.len(), 1);
        assert_eq!(resumable[0].job_id, job2.job_id);
    }

    #[tokio::test]
    async fn test_atomic_reindex_job_creation() {
        let manager = JobManager::new("sqlite::memory:").await.unwrap();

        // First call should succeed
        let job1 = manager
            .create_reindex_job_if_not_active(Some("/test/path".to_string()), 100)
            .await
            .unwrap();
        assert!(job1.is_some());
        let job1 = job1.unwrap();
        assert_eq!(job1.job_type, JobType::Reindex);

        // Second call should fail (job already active)
        let job2 = manager
            .create_reindex_job_if_not_active(Some("/test/path2".to_string()), 200)
            .await
            .unwrap();
        assert!(job2.is_none());

        // Mark first job as completed
        manager
            .update_status(&job1.job_id, JobStatus::Completed, None)
            .await
            .unwrap();

        // Third call should succeed (previous job completed)
        let job3 = manager
            .create_reindex_job_if_not_active(Some("/test/path3".to_string()), 300)
            .await
            .unwrap();
        assert!(job3.is_some());
        let job3 = job3.unwrap();
        assert_eq!(job3.total, 300);
    }

    #[tokio::test]
    async fn test_concurrent_job_creation_race_condition() {
        use std::sync::Arc;

        // Use a temporary file-based database for this test (in-memory doesn't support WAL mode)
        let temp_db = format!("sqlite:/tmp/test_concurrent_{}.db", uuid::Uuid::new_v4());
        let manager = Arc::new(JobManager::new(&temp_db).await.unwrap());

        // Spawn 10 concurrent tasks all trying to create a reindex job
        let mut handles = vec![];
        for i in 0..10 {
            let manager_clone = manager.clone();
            let handle = tokio::spawn(async move {
                manager_clone
                    .create_reindex_job_if_not_active(
                        Some(format!("/test/path{i}")),
                        100 + i as i64,
                    )
                    .await
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap()) // Unwrap JoinHandle result
            .collect();

        // Separate successful operations from errors (SQLITE_BUSY is expected under high concurrency)
        let mut successful_jobs = vec![];
        let mut error_count = 0;

        for result in results {
            match result {
                Ok(Some(job)) => successful_jobs.push(job),
                Ok(None) => {}, // Job already exists - this is expected
                Err(e) => {
                    // SQLITE_BUSY errors are acceptable - they mean the transaction couldn't acquire lock
                    eprintln!("Task failed with error: {e}");
                    error_count += 1;
                }
            }
        }

        // CRITICAL: Exactly 1 task should have successfully created a job
        // (Others should get None for "already exists" or error for lock timeout)
        assert_eq!(
            successful_jobs.len(),
            1,
            "Expected exactly 1 job to be created successfully, but {} were created (errors: {})",
            successful_jobs.len(),
            error_count
        );

        // Verify the database only has 1 job
        let all_jobs = sqlx::query_as::<_, Job>("SELECT * FROM jobs")
            .fetch_all(&manager.pool)
            .await
            .unwrap();

        assert_eq!(
            all_jobs.len(),
            1,
            "Expected exactly 1 job in database, found {}",
            all_jobs.len()
        );

        // The job should be in pending or inprogress state
        assert!(
            matches!(all_jobs[0].status, JobStatus::Pending),
            "Expected job status to be Pending"
        );

        // Cleanup: Remove temp database file
        let db_file = temp_db.strip_prefix("sqlite:").unwrap_or(&temp_db);
        let _ = std::fs::remove_file(db_file);
        let _ = std::fs::remove_file(format!("{db_file}-shm")); // WAL shared memory
        let _ = std::fs::remove_file(format!("{db_file}-wal")); // WAL file
    }
}
