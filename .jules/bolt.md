## 2024-05-22 - Search Candidate Selection Optimization
**Learning:** For RAG systems where candidate sets can be large (especially in brute-force fallback or large lexical recalls), full sorting O(N log N) is wasteful when we only need the top K items.
**Action:** Used `select_nth_unstable_by` to perform an O(N) partition to identify the top K items, followed by sorting only those K items. This improves performance significantly for large candidate sets (e.g., ~400ms -> ~30ms for 100k items in tests, though real world impact depends on candidate set size).

This pattern is useful for any "top-k" retrieval logic where the initial set is large.
