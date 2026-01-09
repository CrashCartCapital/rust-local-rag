use crate::types::ChunkMetadata;
use regex::Regex;
use std::str::FromStr;
use std::sync::OnceLock;
#[cfg(feature = "tracing")]
use tracing::instrument;

#[derive(Debug, Clone)]
struct SentenceInfo {
    text: String,
    tokens: usize,
    page: usize,
    heading: Option<String>,
    index: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ChunkFragment {
    pub(crate) text: String,
    pub(crate) page_number: usize,
    pub(crate) section: Option<String>,
    pub(crate) metadata: ChunkMetadata,
}

impl ChunkFragment {
    fn from_metadata(text: String, metadata: ChunkMetadata) -> Self {
        Self {
            text,
            page_number: metadata.page_range.map(|(start, _)| start).unwrap_or(1),
            section: metadata.section_title.clone(),
            metadata,
        }
    }
}

#[cfg_attr(feature = "tracing", instrument(skip(text)))]
pub(crate) fn chunk_text(
    text: &str,
    chunk_tokens: usize,
    sentence_overlap: usize,
) -> Vec<ChunkFragment> {
    let sentences = extract_sentences(text);
    if sentences.is_empty() {
        #[cfg(feature = "tracing")]
        tracing::debug!("No sentences extracted from text");
        return Vec::new();
    }

    let mut window: Vec<usize> = Vec::new();
    let mut token_sum = 0usize;
    let mut fragments = Vec::new();

    for (idx, sentence) in sentences.iter().enumerate() {
        window.push(idx);
        token_sum += sentence.tokens;

        if token_sum >= chunk_tokens {
            if let Some((chunk_text, metadata)) =
                finalize_chunk(&window, &sentences, sentence_overlap)
            {
                fragments.push(ChunkFragment::from_metadata(chunk_text, metadata));
            }

            let overlap_start = window.len().saturating_sub(sentence_overlap);
            window = window.split_off(overlap_start);
            token_sum = window.iter().map(|&i| sentences[i].tokens).sum();
        }
    }

    if !window.is_empty()
        && let Some((chunk_text, metadata)) = finalize_chunk(&window, &sentences, 0)
    {
        fragments.push(ChunkFragment::from_metadata(chunk_text, metadata));
    }

    #[cfg(feature = "tracing")]
    tracing::debug!(
        "Chunking complete. Created {} fragments from {} sentences.",
        fragments.len(),
        sentences.len()
    );

    fragments
}

fn finalize_chunk(
    sentence_indices: &[usize],
    sentences: &[SentenceInfo],
    overlap_with_previous: usize,
) -> Option<(String, ChunkMetadata)> {
    if sentence_indices.is_empty() {
        return None;
    }

    let mut text_parts: Vec<&str> = Vec::with_capacity(sentence_indices.len());
    let mut min_page: Option<usize> = None;
    let mut max_page: Option<usize> = None;
    let mut section_title: Option<String> = None;
    let mut token_sum = 0usize;

    for &idx in sentence_indices {
        let sentence = sentences.get(idx)?;
        text_parts.push(&sentence.text);
        token_sum += sentence.tokens;

        min_page = Some(match min_page {
            Some(current_min) => current_min.min(sentence.page),
            None => sentence.page,
        });

        max_page = Some(match max_page {
            Some(current_max) => current_max.max(sentence.page),
            None => sentence.page,
        });

        if section_title.is_none()
            && let Some(title) = &sentence.heading
        {
            section_title = Some(title.clone());
        }
    }

    let start_index = sentences
        .get(*sentence_indices.first()?)
        .map(|s| s.index)
        .unwrap_or(0);
    let end_index = sentences
        .get(*sentence_indices.last()?)
        .map(|s| s.index)
        .unwrap_or(start_index);

    let chunk_text = normalize_from_parts(&text_parts);

    let mut metadata = ChunkMetadata {
        page_range: min_page.zip(max_page),
        sentence_range: Some((start_index, end_index)),
        section_title,
        token_count: token_sum,
        overlap_with_previous,
    };

    if let Some(title) = metadata.section_title.as_mut() {
        const MAX_TITLE_LEN: usize = 160;
        if title.chars().count() > MAX_TITLE_LEN {
            *title = title.chars().take(MAX_TITLE_LEN).collect();
        }
    }

    if chunk_text.is_empty() {
        return None;
    }

    Some((chunk_text, metadata))
}

#[cfg_attr(feature = "tracing", instrument(skip(text)))]
fn extract_sentences(text: &str) -> Vec<SentenceInfo> {
    let splitter = sentence_splitter();
    let mut sentences: Vec<SentenceInfo> = Vec::new();
    let mut sentence_index = 0usize;

    for (page_idx, page_text) in text.split('\u{0c}').enumerate() {
        let page_number = page_idx + 1;
        let mut last_heading: Option<String> = None;

        for block in page_text.split("\n\n") {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }

            let lines: Vec<&str> = block.lines().collect();
            if lines.len() == 1 && is_heading(lines[0]) {
                last_heading = Some(lines[0].trim().to_string());
                continue;
            }

            let mut paragraph_lines = Vec::new();
            for line in lines {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if paragraph_lines.is_empty() && is_heading(trimmed) {
                    last_heading = Some(trimmed.to_string());
                    continue;
                }
                paragraph_lines.push(trimmed);
            }

            if paragraph_lines.is_empty() {
                continue;
            }

            let normalized = normalize_from_parts(&paragraph_lines);
            if normalized.is_empty() {
                continue;
            }

            let splits: Vec<&str> = splitter
                .split(&normalized)
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .collect();

            let parts = if splits.is_empty() {
                vec![normalized.as_str()]
            } else {
                splits
            };

            for part in parts {
                let tokens = approximate_token_count(part);
                if tokens == 0 {
                    continue;
                }
                sentences.push(SentenceInfo {
                    text: part.to_string(),
                    tokens,
                    page: page_number,
                    heading: last_heading.clone(),
                    index: sentence_index,
                });
                sentence_index += 1;
            }
        }
    }

    if sentences.is_empty() {
        let normalized = normalize_whitespace(text);
        if !normalized.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::debug!("Fallback: treating entire text as single sentence");
            sentences.push(SentenceInfo {
                text: normalized.clone(),
                tokens: approximate_token_count(&normalized),
                page: 1,
                heading: None,
                index: 0,
            });
        }
    }

    #[cfg(feature = "tracing")]
    tracing::trace!("Extracted {} sentences", sentences.len());

    sentences
}

fn normalize_whitespace(value: &str) -> String {
    normalize_from_parts(&[value])
}

fn normalize_from_parts<S: AsRef<str>>(parts: &[S]) -> String {
    // Optimization: Pre-calculate total length to avoid reallocations.
    // We add parts.len() to account for spaces between parts, although split_whitespace
    // might result in fewer chars, it's a safe upper bound for initial capacity.
    let total_len: usize = parts.iter().map(|s| s.as_ref().len()).sum();
    let estimated_len = total_len + parts.len();

    let mut result = String::with_capacity(estimated_len);
    let mut first = true;
    for part in parts {
        for word in part.as_ref().split_whitespace() {
            if !first {
                result.push(' ');
            }
            result.push_str(word);
            first = false;
        }
    }
    result
}

fn is_heading(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.len() > 120 {
        return false;
    }

    let word_count = trimmed.split_whitespace().count();
    if word_count == 0 || word_count > 12 {
        return false;
    }

    let uppercase_letters = trimmed.chars().filter(|c| c.is_uppercase()).count();
    let lowercase_letters = trimmed.chars().filter(|c| c.is_lowercase()).count();

    if lowercase_letters == 0 && uppercase_letters > 0 {
        return true;
    }

    if trimmed.ends_with(':') {
        return true;
    }

    if word_count <= 4 && uppercase_letters >= lowercase_letters {
        return true;
    }

    heading_regex().is_match(trimmed)
}

fn heading_regex() -> &'static Regex {
    static HEADING_REGEX: OnceLock<Regex> = OnceLock::new();
    HEADING_REGEX.get_or_init(|| Regex::new(r"^\d+\.\s").expect("valid heading regex pattern"))
}

fn approximate_token_count(value: &str) -> usize {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return 0;
    }

    let mut char_count: usize = 0;
    let mut word_count = 0;
    let mut in_word = false;

    if trimmed.is_ascii() {
        let bytes = trimmed.as_bytes();
        char_count = bytes.len();
        for &b in bytes {
            if b.is_ascii_whitespace() {
                in_word = false;
            } else if !in_word {
                in_word = true;
                word_count += 1;
            }
        }
    } else {
        for c in trimmed.chars() {
            char_count += 1;
            if c.is_whitespace() {
                in_word = false;
            } else if !in_word {
                in_word = true;
                word_count += 1;
            }
        }
    }

    let char_estimate = char_count.div_ceil(4);
    // (word_count * 0.9).ceil() is equivalent to (word_count * 9 + 9) / 10 in integer arithmetic
    #[allow(clippy::manual_div_ceil)]
    let word_estimate = (word_count * 9 + 9) / 10;
    char_estimate.max(word_estimate).max(1)
}

fn sentence_splitter() -> &'static srx::Rules {
    static SPLITTER: OnceLock<srx::Rules> = OnceLock::new();
    SPLITTER.get_or_init(|| {
        const SRX_XML: &str = include_str!("../../../data/segment.srx");
        let srx = srx::SRX::from_str(SRX_XML).expect("valid SRX rules from embedded segment.srx");
        srx.language_rules("en")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_info_creation() {
        let test_text = "Dr. Smith presented findings.\u{c}This is page two. Results show success.";

        let sentences = extract_sentences(test_text);

        assert!(!sentences.is_empty(), "Should extract sentences");

        for sentence in &sentences {
            assert!(sentence.tokens > 0, "Each sentence should have token count");
            assert!(sentence.page > 0, "Each sentence should have page number");
        }
    }

    #[test]
    fn test_finalize_chunk_creates_metadata() {
        let test_text = "Sentence one. Sentence two.\u{c}Page two sentence.";
        let sentences = extract_sentences(test_text);

        assert!(!sentences.is_empty(), "Should have sentences");

        let indices: Vec<usize> = vec![0, 1];
        let result = finalize_chunk(&indices, &sentences, 0);

        assert!(result.is_some(), "Should create chunk");

        let (text, metadata) = result.unwrap();
        assert!(!text.is_empty(), "Chunk text should not be empty");
        assert!(
            metadata.sentence_range.is_some(),
            "sentence_range should be populated"
        );
        assert!(metadata.token_count > 0, "token_count should be positive");
        assert!(
            metadata.page_range.is_some(),
            "page_range should be populated"
        );
    }

    #[test]
    fn test_chunk_boundaries_align_to_sentences() {
        let text = "Sentence one. Sentence two.";
        let chunks = chunk_text(text, 1, 0);
        let chunk_texts: Vec<String> = chunks.into_iter().map(|c| c.text).collect();
        assert_eq!(
            chunk_texts,
            vec!["Sentence one.".to_string(), "Sentence two.".to_string()]
        );
    }

    #[test]
    fn test_perf_baseline_correctness() {
        // Validation of behavior before optimization
        assert_eq!(normalize_whitespace("  foo   bar  "), "foo bar");
        assert_eq!(normalize_whitespace("foo\nbar"), "foo bar");

        let parts = vec!["foo  ", "  bar"];
        assert_eq!(normalize_whitespace(&parts.join(" ")), "foo bar");

        // approximate_token_count
        // "hello world" -> chars=11/4=3. words=2*0.9=1.8->2. max=3.
        assert_eq!(approximate_token_count("hello world"), 3);
        // "a b c d e" -> chars=9/4=3. words=5*0.9=4.5->5. max=5.
        // Wait: chars=9 (5 chars + 4 spaces). 9/4 = 2.25 -> 3.
        // words=5. 5*0.9 = 4.5 -> 5.
        // max(3, 5) = 5.
        assert_eq!(approximate_token_count("a b c d e"), 5);
    }

}
