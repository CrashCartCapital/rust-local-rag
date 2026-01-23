use anyhow::{Context, Result};
use std::sync::Arc;
use uuid::Uuid;

pub struct PdfExtractor;

impl PdfExtractor {
    /// Async wrapper for PDF text extraction using spawn_blocking.
    ///
    /// Uses a two-stage fallback strategy:
    /// 1. Try pure-Rust extraction (lopdf) first for deployment flexibility
    /// 2. Fall back to pdftotext binary if lopdf fails
    pub async fn extract_text(data: Vec<u8>) -> Result<String> {
        // Use Arc to share data between fallback tasks without cloning the underlying buffer
        let shared_data = Arc::new(data);
        let data_for_lopdf = Arc::clone(&shared_data);
        let data_for_fallback = Arc::clone(&shared_data);

        let lopdf_result =
            tokio::task::spawn_blocking(move || Self::lopdf_extract_sync(&data_for_lopdf))
                .await
                .context("lopdf extraction task failed")?;

        match lopdf_result {
            Ok(text) => {
                tracing::info!(
                    "✅ PDF extracted using pure-Rust backend (lopdf): {} chars",
                    text.chars().count()
                );
                Ok(text)
            }
            Err(lopdf_err) => {
                tracing::warn!(
                    error = %lopdf_err,
                    "Pure-Rust PDF extraction failed, falling back to pdftotext"
                );

                let lopdf_empty = lopdf_err
                    .downcast_ref::<rag_core::EngineError>()
                    .map(|e| {
                        matches!(
                            e,
                            rag_core::EngineError::Validation {
                                kind: rag_core::ValidationKind::EmptyText,
                                ..
                            }
                        )
                    })
                    .unwrap_or(false);

                let pdftotext_result = tokio::task::spawn_blocking(move || {
                    Self::pdftotext_extract_sync(&data_for_fallback)
                })
                .await
                .context("pdftotext extraction task failed")?;

                match pdftotext_result {
                    Ok(text) => {
                        tracing::info!(
                            "✅ PDF extracted using pdftotext fallback: {} chars",
                            text.chars().count()
                        );
                        Ok(text)
                    }
                    Err(pdftotext_err) => {
                        // Check if pdftotext also returned EmptyText
                        let pdftotext_empty = pdftotext_err
                            .downcast_ref::<rag_core::EngineError>()
                            .map(|e| {
                                matches!(
                                    e,
                                    rag_core::EngineError::Validation {
                                        kind: rag_core::ValidationKind::EmptyText,
                                        ..
                                    }
                                )
                            })
                            .unwrap_or(false);

                        if pdftotext_empty {
                            return Err(pdftotext_err);
                        }

                        if lopdf_empty {
                            return Err(rag_core::EngineError::validation_no_chunk(
                                rag_core::ValidationKind::EmptyText,
                            )
                            .into());
                        }

                        tracing::error!(
                            lopdf_error = %lopdf_err,
                            pdftotext_error = %pdftotext_err,
                            "Both PDF extraction backends failed"
                        );
                        Err(anyhow::anyhow!(
                            "PDF extraction failed: lopdf error: {}, pdftotext error: {}",
                            lopdf_err,
                            pdftotext_err
                        ))
                    }
                }
            }
        }
    }

    fn lopdf_extract_sync(data: &[u8]) -> Result<String> {
        use lopdf::Document;

        let doc = Document::load_mem(data)
            .map_err(|e| anyhow::anyhow!("lopdf failed to parse PDF: {}", e))?;

        let pages = doc.get_pages();
        let mut all_text = String::with_capacity(pages.len() * 500);

        for (page_num, _page_id) in pages {
            match doc.extract_text(&[page_num]) {
                Ok(page_text) => {
                    if !all_text.is_empty() && !page_text.is_empty() {
                        all_text.push('\n');
                    }
                    all_text.push_str(&page_text);
                }
                Err(e) => {
                    tracing::debug!(
                        "lopdf: failed to extract text from page {}: {}",
                        page_num,
                        e
                    );
                }
            }
        }

        if all_text.trim().is_empty() {
            return Err(rag_core::EngineError::validation_no_chunk(
                rag_core::ValidationKind::EmptyText,
            )
            .into());
        }

        Ok(all_text)
    }

    /// Synchronous PDF extraction using pdftotext binary.
    /// Uses UUID for temp filename to prevent race conditions in concurrent calls.
    fn pdftotext_extract_sync(data: &[u8]) -> Result<String> {
        use std::process::Command;

        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!("temp_pdf_{}.pdf", Uuid::new_v4()));

        std::fs::write(&temp_file, data)
            .map_err(|e| anyhow::anyhow!("Failed to write temp PDF: {}", e))?;

        let output = Command::new("pdftotext")
            .arg("-layout")
            .arg("-enc")
            .arg("UTF-8")
            .arg(&temp_file)
            .arg("-")
            .output();
        if let Err(e) = std::fs::remove_file(&temp_file) {
            tracing::debug!(
                error = %e,
                path = %temp_file.display(),
                "Failed to remove temp file after pdftotext"
            );
        }

        match output {
            Ok(output) if output.status.success() => {
                let text = String::from_utf8(output.stdout)
                    .unwrap_or_else(|e| String::from_utf8_lossy(&e.into_bytes()).to_string());

                if text.trim().is_empty() {
                    tracing::warn!("pdftotext extracted 0 characters");
                    Err(rag_core::EngineError::validation_no_chunk(
                        rag_core::ValidationKind::EmptyText,
                    )
                    .into())
                } else {
                    Ok(text)
                }
            }
            Ok(output) => {
                let error_msg = String::from_utf8_lossy(&output.stderr);
                tracing::warn!("pdftotext failed with error: {}", error_msg);
                Err(anyhow::anyhow!("pdftotext failed: {}", error_msg))
            }
            Err(e) => {
                tracing::warn!("Failed to run pdftotext command: {}", e);
                Err(anyhow::anyhow!(
                    "pdftotext command failed: {} (is poppler installed?)",
                    e
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::content::{Content, Operation};
    use lopdf::{Dictionary, Document, Object, Stream, StringFormat};

    // Helper to create a PDF with text
    fn create_pdf_with_text(text: &str) -> Vec<u8> {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", "Font".into()),
            ("Subtype", "Type1".into()),
            ("BaseFont", "Helvetica".into()),
        ]));
        let resources_id = doc.add_object(Dictionary::from_iter(vec![(
            "Font",
            Dictionary::from_iter(vec![("F1", font_id.into())]).into(),
        )]));

        let content = Content {
            operations: vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 48.into()]),
                Operation::new("Td", vec![100.into(), 600.into()]),
                Operation::new(
                    "Tj",
                    vec![Object::String(
                        text.as_bytes().to_vec(),
                        StringFormat::Literal,
                    )],
                ),
                Operation::new("ET", vec![]),
            ],
        };
        let stream = Stream::new(Dictionary::new(), content.encode().unwrap());
        let stream_id = doc.add_object(stream);

        let page_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", "Page".into()),
            ("Parent", pages_id.into()),
            ("Contents", stream_id.into()),
            ("Resources", resources_id.into()),
        ]));

        let pages = Dictionary::from_iter(vec![
            ("Type", "Pages".into()),
            ("Kids", vec![page_id.into()].into()),
            ("Count", 1.into()),
        ]);
        doc.objects.insert(pages_id, Object::Dictionary(pages));

        let catalog_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", "Catalog".into()),
            ("Pages", pages_id.into()),
        ]));

        doc.trailer.set("Root", catalog_id);

        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();
        buffer
    }

    // Helper to create a PDF with no text
    fn create_empty_pdf() -> Vec<u8> {
        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();

        // Create an empty page content stream
        let content = Content { operations: vec![] };
        let stream = Stream::new(Dictionary::new(), content.encode().unwrap());
        let stream_id = doc.add_object(stream);

        // Create page dictionary
        let page_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", "Page".into()),
            ("Parent", pages_id.into()),
            ("Contents", stream_id.into()),
        ]));

        // Create pages dictionary
        let pages = Dictionary::from_iter(vec![
            ("Type", "Pages".into()),
            ("Kids", vec![page_id.into()].into()),
            ("Count", 1.into()),
        ]);
        doc.objects.insert(pages_id, Object::Dictionary(pages));

        // Create catalog
        let catalog_id = doc.add_object(Dictionary::from_iter(vec![
            ("Type", "Catalog".into()),
            ("Pages", pages_id.into()),
        ]));

        doc.trailer.set("Root", catalog_id);

        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();
        buffer
    }

    #[tokio::test]
    async fn test_extract_text_success() {
        let pdf_data = create_pdf_with_text("Hello World");
        let text = PdfExtractor::extract_text(pdf_data)
            .await
            .expect("Extraction failed");
        assert!(text.contains("Hello World"));
    }

    #[tokio::test]
    async fn test_extract_text_empty_returns_validation_error() {
        let pdf_data = create_empty_pdf();
        let result = PdfExtractor::extract_text(pdf_data).await;

        match result {
            Ok(_) => panic!("Should have returned error"),
            Err(e) => {
                if let Some(engine_error) = e.downcast_ref::<rag_core::EngineError>() {
                    match engine_error {
                        rag_core::EngineError::Validation {
                            kind: rag_core::ValidationKind::EmptyText,
                            ..
                        } => {
                            // Success
                        }
                        _ => panic!("Expected Validation(EmptyText), got {engine_error:?}"),
                    }
                } else {
                    panic!("Got unexpected error type: {e:?}");
                }
            }
        }
    }

    #[tokio::test]
    async fn test_extract_text_corrupt() {
        let pdf_data = vec![0u8; 100]; // Random junk
        let result = PdfExtractor::extract_text(pdf_data).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        // Should be a parsing error, not validation empty text
        assert!(err.to_string().contains("lopdf failed to parse PDF"));
    }
}
