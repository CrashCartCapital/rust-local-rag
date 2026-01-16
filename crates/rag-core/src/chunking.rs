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

            let overlap_start = window.len().saturating_sub(sentence_overlap).max(1);
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

struct SentenceExtractor<'a> {
    splitter: &'a srx::Rules,
    sentences: Vec<SentenceInfo>,
    sentence_index: usize,
    last_heading: Option<String>,
}

impl<'a> SentenceExtractor<'a> {
    fn new() -> Self {
        Self {
            splitter: sentence_splitter(),
            sentences: Vec::new(),
            sentence_index: 0,
            last_heading: None,
        }
    }

    fn extract(&mut self, text: &str) {
        for (page_idx, page_text) in text.split('\u{0c}').enumerate() {
            let page_number = page_idx + 1;
            // Heading context resets per page
            self.last_heading = None;
            self.process_page(page_text, page_number);
        }

        // Fallback: if no sentences were found but text exists, treat as one blob
        if self.sentences.is_empty() {
            let normalized = normalize_whitespace(text);
            if !normalized.is_empty() {
                #[cfg(feature = "tracing")]
                tracing::debug!("Fallback: treating entire text as single sentence");
                self.push_sentence(normalized, 1, None);
            }
        }
    }

    fn process_page(&mut self, page_text: &str, page_number: usize) {
        for block in page_text.split("\n\n") {
            let block = block.trim();
            if block.is_empty() {
                continue;
            }
            self.process_block(block, page_number);
        }
    }

    fn process_block(&mut self, block: &str, page_number: usize) {
        let lines: Vec<&str> = block.lines().collect();

        // Case 1: Block is just a heading
        if lines.len() == 1 && is_heading(lines[0]) {
            self.last_heading = Some(lines[0].trim().to_string());
            return;
        }

        // Case 2: Block contains paragraph text, possibly starting with heading
        let mut paragraph_lines = Vec::new();
        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // If the first line of what we thought was a paragraph is actually a heading
            if paragraph_lines.is_empty() && is_heading(trimmed) {
                self.last_heading = Some(trimmed.to_string());
                continue;
            }
            paragraph_lines.push(trimmed);
        }

        if paragraph_lines.is_empty() {
            return;
        }

        self.process_paragraph(&paragraph_lines, page_number);
    }

    fn process_paragraph(&mut self, lines: &[&str], page_number: usize) {
        let normalized = normalize_from_parts(lines);
        if normalized.is_empty() {
            return;
        }

        let splits: Vec<&str> = self
            .splitter
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
            let heading = self.last_heading.clone();
            self.push_sentence(part.to_string(), page_number, heading);
        }
    }

    fn push_sentence(&mut self, text: String, page: usize, heading: Option<String>) {
        let tokens = approximate_token_count(&text);
        if tokens == 0 {
            return;
        }

        self.sentences.push(SentenceInfo {
            text,
            tokens,
            page,
            heading,
            index: self.sentence_index,
        });
        self.sentence_index += 1;
    }
}

#[cfg_attr(feature = "tracing", instrument(skip(text)))]
fn extract_sentences(text: &str) -> Vec<SentenceInfo> {
    let mut extractor = SentenceExtractor::new();
    extractor.extract(text);

    #[cfg(feature = "tracing")]
    tracing::trace!("Extracted {} sentences", extractor.sentences.len());

    extractor.sentences
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
        let chunks = chunk_text(text, 5, 10);
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
        let chunks = chunk_text(text, 5, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
    }

    #[test]
    fn test_chunking_tiny_limit() {
        let text = "First. Second. Third.";
        // Limit 1. Each sentence is ~1-2 tokens.
        // Should produce 3 chunks.
        let chunks = chunk_text(text, 1, 0);
        let texts: Vec<String> = chunks.into_iter().map(|c| c.text).collect();
        assert_eq!(texts, vec!["First.", "Second.", "Third."]);
    }
}

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_heading_behavior() {
        let text = "Heading 1\n\nBody text.\n\nHeading 2\nBody text 2.";
        let sentences = extract_sentences(text);

        // Debug output to see what we actually get
        for (i, s) in sentences.iter().enumerate() {
            println!("Sentence {}: '{}', heading: {:?}", i, s.text, s.heading);
        }

        // Based on current logic:
        // "Heading 1" is a block. split("\n\n").
        // lines.len() == 1. is_heading("Heading 1") -> true (short, mixed case/numeric?).
        // Actually "Heading 1" has uppercase H, lowercase eading.
        // is_heading checks:
        //  - word_count (2) <= 12.
        //  - uppercase > 0.
        //  - lowercase > 0.
        //  - word_count <= 4 && uppercase >= lowercase? "Heading 1": 2 words. "H" "1"? No.
        //  - heading_regex?

        // Wait, "Heading 1" might not match is_heading logic if not careful.
        // "Heading 1" has 1 uppercase ('H'). 6 lowercase.
        // uppercase >= lowercase? 1 >= 6 False.
        // heading_regex? "^\d+\.\s". No.

        // Let's use a clearer heading: "1. Introduction"

        let text_clear = "1. Introduction\n\nBody text.\n\n2. Conclusion\nBody text 2.";
        let sentences = extract_sentences(text_clear);

        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].text, "Body text.");
        assert_eq!(sentences[0].heading, Some("1. Introduction".to_string()));

        assert_eq!(sentences[1].text, "Body text 2.");
        // "2. Conclusion" is followed by "\nBody text 2." in the SAME block?
        // No, I used "\n\n" in the string literal.
        // "2. Conclusion\nBody text 2." -> This is ONE block if split by "\n\n".
        // Inside block:
        // Line 1: "2. Conclusion". is_heading -> true.
        // last_heading updated. continue.
        // Line 2: "Body text 2.". Added to paragraph_lines.

        assert_eq!(sentences[1].heading, Some("2. Conclusion".to_string()));
    }
}
