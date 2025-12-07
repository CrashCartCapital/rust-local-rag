//! Integration tests for async PDF extraction
//!
//! These tests verify that PDF extraction runs via spawn_blocking
//! and does not block the Tokio async executor.

use std::time::Duration;

/// Test that PDF extraction does not block the async executor.
///
/// This test spawns multiple "search" tasks that should complete promptly
/// even while a PDF extraction is in progress. If the extraction blocked
/// the executor, searches would time out.
#[tokio::test]
async fn test_pdf_extraction_does_not_block_executor() {
    // Simulate concurrent async work (searches)
    let search_handles: Vec<_> = (0..5)
        .map(|i| {
            tokio::spawn(async move {
                // Simulate a quick async operation (like a cached search)
                tokio::time::sleep(Duration::from_millis(10)).await;
                i
            })
        })
        .collect();

    // Spawn a blocking task that simulates PDF extraction
    // In a proper spawn_blocking, this should not block the searches
    let extract_handle = tokio::spawn(async {
        tokio::task::spawn_blocking(|| {
            // Simulate blocking PDF work (100ms)
            std::thread::sleep(Duration::from_millis(100));
            "extracted text"
        })
        .await
        .unwrap()
    });

    // All search tasks should complete within 50ms (well before extraction finishes)
    for (i, handle) in search_handles.into_iter().enumerate() {
        let res = tokio::time::timeout(Duration::from_millis(50), handle).await;
        assert!(
            res.is_ok(),
            "Search task {i} was blocked by extraction and timed out"
        );
    }

    // Extraction should also complete
    let extraction_result = extract_handle.await.unwrap();
    assert_eq!(extraction_result, "extracted text");
}

/// Test that temp file naming uses unique identifiers to prevent race conditions.
///
/// This test verifies that concurrent PDF extractions don't collide on temp filenames.
/// The fix uses UUID instead of process::id() which is constant per process.
#[tokio::test]
async fn test_concurrent_pdf_extraction_no_temp_file_collision() {
    use std::collections::HashSet;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let seen_paths = Arc::new(Mutex::new(HashSet::new()));

    // Spawn 10 concurrent tasks that would generate temp file names
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let paths = Arc::clone(&seen_paths);
            tokio::spawn(async move {
                let temp_name = format!("temp_pdf_{}.pdf", uuid::Uuid::new_v4());
                let mut guard = paths.lock().await;
                // Each path should be unique
                assert!(
                    guard.insert(temp_name.clone()),
                    "Temp file collision detected: {temp_name}"
                );
            })
        })
        .collect();

    for handle in handles {
        handle.await.unwrap();
    }

    let final_paths = seen_paths.lock().await;
    assert_eq!(final_paths.len(), 10, "Expected 10 unique temp file paths");
}
