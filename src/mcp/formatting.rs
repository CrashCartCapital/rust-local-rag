pub(crate) fn get_highlight_regex(query: &str) -> Option<regex::Regex> {
    let terms: Vec<&str> = query
        .split_whitespace()
        .filter(|t| {
            if t.len() > 2 {
                true
            } else if t.len() == 2 {
                !matches!(
                    *t,
                    "as" | "at" | "by" | "if" | "in" | "is" | "it" | "of" | "on" | "or" | "to"
                )
            } else {
                false
            }
        })
        .collect();

    if !terms.is_empty() {
        let pattern = terms
            .iter()
            .map(|t| {
                let escaped = regex::escape(t);
                if t.chars().next().is_some_and(|c| c.is_alphanumeric()) {
                    format!("\\b{escaped}")
                } else {
                    escaped
                }
            })
            .collect::<Vec<_>>()
            .join("|");
        regex::RegexBuilder::new(&format!("(?i)({pattern})"))
            .build()
            .ok()
    } else {
        None
    }
}

pub(crate) fn format_search_results(
    results: &[crate::rag_engine::SearchResult],
    query: &str,
) -> String {
    if results.is_empty() {
        return "No results found. Try broader keywords or check if documents are uploaded with `list_documents`."
            .to_string();
    }

    let highlight_re = get_highlight_regex(query);

    results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let provenance = if result.page_number > 0 {
                format!("{} (page {})", result.document, result.page_number)
            } else {
                result.document.clone()
            };
            let section = result
                .section
                .as_ref()
                .map(|s| format!("*Section: {s}*\n"))
                .unwrap_or_default();

            let percentage = (result.score * 100.0).round() as i32;

            let mut score_parts = Vec::new();
            if let Some(s) = result.embedding_score {
                score_parts.push(format!("Semantic: {s:.2}"));
            }
            if let Some(s) = result.lexical_score {
                #[allow(clippy::collapsible_if)]
                if s > 0.0 {
                    score_parts.push(format!("Keyword: {s:.2}"));
                }
            }
            if let Some(s) = result.reranker_score {
                score_parts.push(format!("Reranker: {s:.2}"));
            }

            let score_breakdown = if !score_parts.is_empty() {
                format!("*Scores: {}*\n", score_parts.join(" | "))
            } else {
                String::new()
            };

            let text = if let Some(re) = &highlight_re {
                re.replace_all(&result.text, "**$1**").to_string()
            } else {
                result.text.clone()
            };

            format!(
                "**{}. [{}%] {}**\n{}{}{}\n",
                i + 1,
                percentage,
                provenance,
                section,
                score_breakdown,
                text
            )
        })
        .collect::<Vec<_>>()
        .join("\n---\n\n")
}
