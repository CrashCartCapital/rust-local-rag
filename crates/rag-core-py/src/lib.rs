//! Python bindings for rag-core RAG engine.
//!
//! This crate provides PyO3-based Python bindings that expose rag-core's
//! RAG functionality to Python applications.

use pyo3::prelude::*;

mod adapters;
mod conversions;
mod engine;
mod errors;
mod mock_backend;
mod types;

/// Initialize the ragcore native module.
///
/// This is the entry point for the Python extension module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register exception types
    errors::register_exceptions(m)?;

    // Register classes
    m.add_class::<engine::PyRagEngine>()?;
    m.add_class::<types::PyQuerySpec>()?;
    m.add_class::<types::PySearchResult>()?;
    m.add_class::<types::PyScoreBreakdown>()?;
    m.add_class::<types::PyEngineStats>()?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
