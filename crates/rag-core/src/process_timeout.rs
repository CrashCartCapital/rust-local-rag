use std::time::Duration;

#[derive(Debug)]
pub enum TimeoutError {
    Expired,
}

impl std::fmt::Display for TimeoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeoutError::Expired => write!(f, "Process timed out"),
        }
    }
}

impl std::error::Error for TimeoutError {}

/// Checks if a deadline has been exceeded.
pub fn check_deadline(start: std::time::Instant, timeout: Duration) -> Result<(), TimeoutError> {
    if start.elapsed() > timeout {
        Err(TimeoutError::Expired)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_deadline() {
        let start = std::time::Instant::now();
        let timeout = Duration::from_millis(100);

        // Should be ok immediately
        assert!(check_deadline(start, timeout).is_ok());

        // Sleep past timeout
        std::thread::sleep(Duration::from_millis(150));
        assert!(matches!(
            check_deadline(start, timeout),
            Err(TimeoutError::Expired)
        ));
    }
}
