use anyhow::Result;
use rust_local_rag::job_manager::{JobManager, JobStatus, JobType};

#[tokio::test]
async fn test_repro_invalid_state_transition() -> Result<()> {
    let manager = JobManager::new("sqlite::memory:").await?;
    let job = manager.create_job(JobType::Reindex, None, 100).await?;

    // Transition to Completed
    manager.update_status(&job.job_id, JobStatus::Completed, None).await?;

    // Illegal transition: Completed -> Pending
    // This should fail.
    let result = manager.update_status(&job.job_id, JobStatus::Pending, None).await;
    assert!(result.is_err(), "Expected error for invalid transition");

    let err = result.unwrap_err();
    assert!(err.to_string().contains("Invalid job state transition"));

    // Verify state did not change
    let updated = manager.get_job(&job.job_id).await?.unwrap();
    assert_eq!(updated.status, JobStatus::Completed);

    Ok(())
}
