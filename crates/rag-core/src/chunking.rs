use crate::error::{EngineError, Result};
use crate::types::ChunkMetadata;
use regex::Regex;
use std::str::FromStr;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
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
    timeout: Option<Duration>,
) -> Result<Vec<ChunkFragment>> {
    let start_time = Instant::now();
    let sentences = extract_sentences(text, Some(chunk_tokens), timeout)?;
    if sentences.is_empty() {
        #[cfg(feature = "tracing")]
        tracing::debug!("No sentences extracted from text");
        return Ok(Vec::new());
    }

    let mut window: Vec<usize> = Vec::new();
    let mut token_sum = 0usize;
    let mut fragments = Vec::new();
    let mut max_emitted_index: Option<usize> = None;

    for (idx, sentence) in sentences.iter().enumerate() {
        if let Some(t) = timeout {
            if start_time.elapsed() > t {
                return Err(EngineError::Timeout(t));
            }
        }

        window.push(idx);
        token_sum += sentence.tokens;

        if token_sum >= chunk_tokens {
            if let Some((chunk_text, metadata)) =
                finalize_chunk(&window, &sentences, sentence_overlap)
            {
                if let Some((_, end)) = metadata.sentence_range {
                    max_emitted_index = Some(max_emitted_index.map_or(end, |m| m.max(end)));
                }
                fragments.push(ChunkFragment::from_metadata(chunk_text, metadata));
            }

            let overlap_start = window.len().saturating_sub(sentence_overlap).max(1);
            window = window.split_off(overlap_start);
            token_sum = window.iter().map(|&i| sentences[i].tokens).sum();
        }
    }

    if !window.is_empty() {
        let last_idx = window.last().copied().unwrap_or(0);
        let last_sentence_index = sentences[last_idx].index;

        let is_redundant = if let Some(max_idx) = max_emitted_index {
            last_sentence_index <= max_idx
        } else {
            false
        };

        if !is_redundant {
            if let Some((chunk_text, metadata)) = finalize_chunk(&window, &sentences, 0) {
                fragments.push(ChunkFragment::from_metadata(chunk_text, metadata));
            }
        }
    }

    #[cfg(feature = "tracing")]
    tracing::debug!(
        "Chunking complete. Created {} fragments from {} sentences.",
        fragments.len(),
        sentences.len()
    );

    Ok(fragments)
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
fn extract_sentences(
    text: &str,
    hard_token_limit: Option<usize>,
    timeout: Option<Duration>,
) -> Result<Vec<SentenceInfo>> {
    let start_time = Instant::now();
    let splitter = sentence_splitter();
    let mut sentences: Vec<SentenceInfo> = Vec::new();
    let mut sentence_index = 0usize;

    for (page_idx, page_text) in text.split('\u{0c}').enumerate() {
        if let Some(t) = timeout {
            if start_time.elapsed() > t {
                return Err(EngineError::Timeout(t));
            }
        }

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

                if let Some(limit) = hard_token_limit {
                    if tokens > limit {
                        let sub_parts = split_part_hard(part, limit);
                        for sub in sub_parts {
                            let sub_tokens = approximate_token_count(&sub);
                            if sub_tokens == 0 {
                                continue;
                            }
                            sentences.push(SentenceInfo {
                                text: sub,
                                tokens: sub_tokens,
                                page: page_number,
                                heading: last_heading.clone(),
                                index: sentence_index,
                            });
                            sentence_index += 1;
                        }
                        continue;
                    }
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

    Ok(sentences)
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

    match heading_regex() {
        Ok(re) => re.is_match(trimmed),
        Err(_e) => {
            #[cfg(feature = "tracing")]
            tracing::error!("Regex invalid: {}", _e);
            false
        }
    }
}

fn heading_regex() -> std::result::Result<&'static Regex, &'static str> {
    static HEADING_REGEX: OnceLock<std::result::Result<Regex, String>> = OnceLock::new();
    HEADING_REGEX
        .get_or_init(|| {
            Regex::new(r"^\d+\.\s").map_err(|e| format!("valid heading regex pattern: {}", e))
        })
        .as_ref()
        .map_err(|e| e.as_str())
}

fn approximate_token_count(value: &str) -> usize {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return 0;
    }

    let bytes = trimmed.as_bytes();
    let mut space_count = 0;
    let mut continuation_count = 0;

    // Optimization: Single pass byte scan avoids state machine overhead.
    // 1. We know the input is normalized (from normalize_from_parts) so words are separated
    //    by single spaces. Thus word_count = space_count + 1.
    // 2. We can detect multi-byte chars by counting continuation bytes (10xxxxxx).
    //    char_count = total_bytes - continuation_bytes.
    for &b in bytes {
        if b == b' ' {
            space_count += 1;
        } else if (b & 0xC0) == 0x80 {
            // Count UTF-8 continuation bytes to subtract from total length
            continuation_count += 1;
        }
    }

    let char_count = bytes.len() - continuation_count;
    let word_count = space_count + 1;

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

fn split_part_hard(text: &str, limit: usize) -> Vec<String> {
    if limit == 0 {
        return vec![text.to_string()];
    }
    // First try splitting by whitespace
    let mut result = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    // If it's a single word (no whitespace) or just one word
    if words.len() <= 1 {
        return split_massive_word(text, limit);
    }

    let mut current_chunk = String::new();
    for word in words {
        let word_tokens = approximate_token_count(word);
        if word_tokens > limit {
            if !current_chunk.is_empty() {
                result.push(current_chunk);
                current_chunk = String::new();
            }
            result.extend(split_massive_word(word, limit));
            continue;
        }

        let candidate = if current_chunk.is_empty() {
            word.to_string()
        } else {
            format!("{current_chunk} {word}")
        };

        if approximate_token_count(&candidate) > limit {
            if !current_chunk.is_empty() {
                result.push(current_chunk);
            }
            current_chunk = word.to_string();
        } else {
            current_chunk = candidate;
        }
    }
    if !current_chunk.is_empty() {
        result.push(current_chunk);
    }
    result
}

fn split_massive_word(text: &str, limit: usize) -> Vec<String> {
    if limit == 0 {
        return vec![text.to_string()];
    }
    let mut result = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut start = 0;

    while start < chars.len() {
        let mut end = (start + limit.saturating_mul(4)).min(chars.len());

        loop {
            let slice: String = chars[start..end].iter().collect();
            let tokens = approximate_token_count(&slice);

            if tokens <= limit {
                result.push(slice);
                start = end;
                break;
            } else {
                if end - start <= 1 {
                    result.push(slice);
                    start = end;
                    break;
                }
                end -= 1;
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_info_creation() {
        let test_text = "Dr. Smith presented findings.\u{c}This is page two. Results show success.";

        let sentences = extract_sentences(test_text, None, None).unwrap();

        assert!(!sentences.is_empty(), "Should extract sentences");

        for sentence in &sentences {
            assert!(sentence.tokens > 0, "Each sentence should have token count");
            assert!(sentence.page > 0, "Each sentence should have page number");
        }
    }

    #[test]
    fn test_finalize_chunk_creates_metadata() {
        let test_text = "Sentence one. Sentence two.\u{c}Page two sentence.";
        let sentences = extract_sentences(test_text, None, None).unwrap();

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
        // Limit 4 is exactly enough for one sentence (~4 tokens).
        let chunks = chunk_text(text, 4, 0, None).unwrap();
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

        let parts = ["foo  ", "  bar"];
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

    #[test]
    fn test_chunking_overlap_exceeds_window() {
        // Use full words to ensure SRX splits them correctly.
        let text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.";
        // Approx tokens per sentence:
        // "Sentence one." -> 13 chars (4 tok) / 2 words (2 tok). -> 4 tokens.
        // limit 5.
        // Chunk 1: "Sentence one." (4) < 5.
        // Chunk 1: "Sentence one. Sentence two." (8) >= 5.
        // Overlap 10.
        let chunks = chunk_text(text, 5, 10, None).unwrap();
        let chunk_texts: Vec<String> = chunks.into_iter().map(|c| c.text).collect();

        // If bug exists (overlap=0 calc):
        // 1. "one. two." (window keeps both)
        // 2. "one. two. three."
        // 3. "one. two. three. four."
        // 4. "one. two. three. four. five."
        // 5. "one. two. three. four. five." (after loop)

        // If fixed (overlap_start=max(1)):
        // 1. "one. two." (drop one) -> window [two]
        // 2. "two. three." (drop two) -> window [three]
        // 3. "three. four."
        // 4. "four. five."
        // 5. "five." (remaining)

        assert!(
            chunk_texts.len() <= 6,
            "Should not produce infinitely growing chunks. Got: {:?}",
            chunk_texts
        );
        if !chunk_texts.is_empty() {
            assert_eq!(chunk_texts[0], "Sentence one. Sentence two.");
            // If the bug exists, the second chunk will be longer
            assert_eq!(chunk_texts[1], "Sentence two. Sentence three.");
        }
    }

    #[test]
    fn test_chunking_massive_sentence() {
        // Single sentence larger than chunk limit
        let text = "This is a very long sentence that exceeds the token limit all by itself.";
        // Approx tokens: ~60 chars -> 15 tokens.
        // Limit: 5.
        let chunks = chunk_text(text, 5, 0, None).unwrap();
        // Should be split into multiple chunks
        assert!(chunks.len() > 1);
        for chunk in chunks {
            // Chunks can exceed limit by one sentence size. Since max sentence size is limit,
            // max chunk size is roughly 2 * limit.
            assert!(
                chunk.metadata.token_count <= 10,
                "Chunk exceeded relaxed limit (2x): {}",
                chunk.metadata.token_count
            );
        }
    }

    #[test]
    fn test_chunking_tiny_limit() {
        let text = "First. Second. Third.";
        // Limit 2. Each sentence is ~1-2 tokens.
        // Should produce 3 chunks.
        let chunks = chunk_text(text, 2, 0, None).unwrap();
        let texts: Vec<String> = chunks.into_iter().map(|c| c.text).collect();
        assert_eq!(texts, vec!["First.", "Second.", "Third."]);
    }

    #[test]
    fn test_chunking_massive_token_splitting() {
        // Create a string that is effectively one giant word
        let limit = 10;
        // Each char is 0.25 tokens. We want > 10 tokens.
        // 100 chars -> 25 tokens.
        let text = "a".repeat(100);

        let chunks = chunk_text(&text, limit, 0, None).unwrap();

        // Currently this returns 1 chunk of size 25.
        // We want it to split.
        // With limit 10, we expect roughly 3 chunks.
        assert!(
            chunks.len() > 1,
            "Should split massive token. Got: {}",
            chunks.len()
        );
        for chunk in chunks {
            assert!(
                chunk.metadata.token_count <= limit,
                "Chunk token count {} exceeds limit {}",
                chunk.metadata.token_count,
                limit
            );
        }
    }

    #[test]
    fn test_split_massive_word_overflow() {
        // `split_massive_word` uses `limit * 4` as an upper bound on chars-per-token.
        // Make sure this calculation can't overflow.
        let limit = (usize::MAX / 4) + 1;

        let text = "a".repeat(20);
        // Should produce 1 chunk because 20 tokens (approx) < massive limit
        let chunks = split_massive_word(&text, limit);

        assert_eq!(chunks.len(), 1, "Should have 1 chunk for massive limit");
        assert_eq!(chunks[0], text, "Should preserve text");
    }

    #[test]
    fn test_split_massive_word_unicode_boundaries() {
        let limit = 2;
        let text = "😀😀😀😀";

        let chunks = split_massive_word(text, limit);
        assert_eq!(chunks.len(), 1, "Should fit in one chunk");

        // Let's use something that generates more tokens.
        // "a".repeat(20).
        // 20 chars -> 5 tokens.
        // Limit 2.
        // Should split into ~3 chunks (2 chunks of 8 chars=2tok, 1 chunk of 4 chars=1tok).

        let text = "a".repeat(20);
        let chunks = split_massive_word(&text, limit);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_split_massive_word_zero_limit_guard() {
        let text = "test";
        let chunks = split_massive_word(text, 0);
        // Existing code guard: returns vec![text]
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "test");
    }

    #[test]
    fn test_redundant_tail_chunk_suppression() {
        let s1 = "First sentence.";
        let s2 = "Second sentence.";
        let text = format!("{} {}", s1, s2);

        // Debug sentences
        let sentences = extract_sentences(&text, None, None).unwrap();
        for s in &sentences {
            println!("S: '{}' ({})", s.text, s.tokens);
        }
        // "First sentence." -> ~3 tokens
        // "Second sentence." -> ~4 tokens

        // Limit 6.
        // S1 (3) < 6.
        // S1 + S2 (7) >= 6. Trigger.
        // Chunk [S1, S2].
        // Overlap 1 (S2). Window becomes [S2].
        // Loop ends.

        // Final window [S2]. Last index is S2.
        // S2 was part of Chunk 1. Redundant.

        let chunks = chunk_text(&text, 6, 1, None).unwrap();
        for (i, c) in chunks.iter().enumerate() {
            println!("Chunk {}: {}", i, c.text);
        }

        assert_eq!(chunks.len(), 1, "Should not emit redundant tail chunk");
        assert_eq!(chunks[0].text, text, "First chunk should be full text");
    }

    #[test]
    fn test_tail_chunk_with_new_content_is_kept() {
        // S1 (3), S2 (4), S3 (3).
        let s1 = "First sentence.";
        let s2 = "Second sentence.";
        let s3 = "Third sentence.";
        let text = format!("{} {} {}", s1, s2, s3);

        // Limit 6.
        // [S1, S2] -> 7 >= 6. Trigger Chunk 1.
        // Overlap 1 -> [S2].
        // Add S3 -> [S2, S3] -> 7 >= 6. Trigger Chunk 2.
        // Overlap 1 -> [S3].
        // Loop ends.
        // Final window [S3]. Last is S3.
        // S3 was included in Chunk 2. Redundant.

        // This setup produces [S1, S2], [S2, S3].
        // Both are kept. The final residue [S3] is dropped.

        let chunks = chunk_text(&text, 6, 1, None).unwrap();

        assert_eq!(chunks.len(), 2, "Should have 2 chunks");
        assert_eq!(chunks[0].text, format!("{} {}", s1, s2));
        assert_eq!(chunks[1].text, format!("{} {}", s2, s3));

        // Now try a case where the final chunk is NOT triggered in loop but added at end.
        // "Longer first sentence here." (4 words, 15+chars -> ~5 tokens)
        // "Short one." (~2 tokens)
        // "Tiny." (1 token)

        let s1_long = "Longer first sentence here.";
        let s2_short = "Short one.";
        let s3_tiny = "Tiny.";
        let _text2 = format!("{} {} {}", s1_long, s2_short, s3_tiny);

        // Limit 8.
        // [S1] (5) < 8.
        // [S1, S2] (7) < 8.
        // [S1, S2, S3] (8) >= 8. Trigger Chunk 1.
        // Overlap 1 -> [S3].
        // Loop ends.
        // Final window [S3]. Last is S3.
        // S3 was emitted? Yes, in Chunk 1.
        // S3 index (2). Max emitted (2).
        // Redundant.

        // Wait, we need a case where S3 was NOT emitted fully?
        // Or where we didn't trigger?

        // Limit 10.
        // [S1, S2, S3] (8) < 10.
        // Loop ends.
        // Final [S1, S2, S3].
        // Max emitted None.
        // Not redundant. Emit.

        // We need: Loop triggers chunk, leaves residue that is NEW.
        // [S1, S2] trigger. [S3] remains and is new.
        // S1(5), S2(5), S3(1). Limit 8.
        // [S1, S2] (10) >= 8. Trigger.
        // Overlap 1 -> [S2].
        // Add S3 -> [S2, S3] (6) < 8.
        // Loop ends.
        // Final [S2, S3].
        // S3 index > S2 (max emitted).
        // Emit.

        let s1_big = "This is a significantly longer sentence to occupy space.";
        let s2_big = "Another significantly long sentence to trigger the limit.";
        let s3_small = "Small.";

        let text3 = format!("{} {} {}", s1_big, s2_big, s3_small);
        // Approx: S1 ~10, S2 ~10, S3 ~1.
        // Limit 15.
        // [S1, S2] -> 20. Trigger.
        // Overlap 1 -> [S2].
        // [S2, S3] -> 11 < 15.
        // Final emit [S2, S3].

        let chunks3 = chunk_text(&text3, 15, 1, None).unwrap();
        assert_eq!(chunks3.len(), 2, "Should emit tail chunk with new content");
        assert_eq!(chunks3[0].text, format!("{} {}", s1_big, s2_big));
        assert_eq!(chunks3[1].text, format!("{} {}", s2_big, s3_small));
    }

    #[test]
    fn test_chunking_timeout() {
        // Massive text to force some processing time, though "0" timeout is the key.
        let text = "a".repeat(10000);

        // Timeout of 0 nanoseconds (effectively immediate).
        // Since we check elapsed() > timeout, this might be tricky if it runs too fast.
        // But usually elapsed() will be non-zero after some ops.
        // To be safe, we can use a small timeout that is guaranteed to be exceeded by processing
        // if we loop enough, but here we can just pass Duration::ZERO.

        let result = chunk_text(&text, 100, 0, Some(Duration::from_nanos(0)));
        assert!(result.is_err(), "Should timeout with 0 duration");

        if let Err(EngineError::Timeout(_)) = result {
            // Success
        } else {
            panic!("Expected EngineError::Timeout, got {:?}", result);
        }
    }

    #[test]
    fn test_heading_regex_compilation() {
        assert!(
            heading_regex().is_ok(),
            "Heading regex should compile successfully"
        );
    }
}
