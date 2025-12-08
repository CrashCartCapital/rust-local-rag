# RAG Evaluation Framework Specification

**Version**: 1.1 (Post-Ensemble Review)
**Date**: 2025-12-07
**Status**: Approved - Ready for Implementation
**Reviewed by**: CRASH + Gemini-3-Pro-Preview + Codex

---

## Executive Summary

This document specifies a rigorous evaluation framework for rust-local-rag's retrieval and reranking pipeline. The framework enables:

1. **Objective quality measurement** via ground truth datasets and IR metrics
2. **A/B testing** of embedding models, reranker models, and hybrid search weights
3. **Prompt engineering** experiments for reranker optimization
4. **Performance profiling** with latency tracking

**Current State**: 70 documents, 85,492 chunks, zero search quality tests.
**Goal**: Statistically valid evaluation with actionable optimization insights.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Ground Truth Dataset](#2-ground-truth-dataset)
3. [Metrics Suite](#3-metrics-suite)
4. [Evaluation Harness](#4-evaluation-harness)
5. [Experiment Configurations](#5-experiment-configurations)
6. [A/B Testing Framework](#6-ab-testing-framework)
7. [Prompt Engineering Harness](#7-prompt-engineering-harness)
8. [Model Comparison Matrix](#8-model-comparison-matrix)
9. [Implementation Phases](#9-implementation-phases)
10. [File Structure](#10-file-structure)
11. [Open Questions](#11-open-questions)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Evaluation Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Ground Truth │───▶│  Eval Runner │───▶│   Reports    │       │
│  │   Dataset    │    │   (Python)   │    │  (MD + CSV)  │       │
│  └──────────────┘    └──────┬───────┘    └──────────────┘       │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    rust-local-rag                         │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │   │
│  │  │  Embedding  │──▶│   Hybrid    │──▶│  Reranker   │     │   │
│  │  │   Model     │   │   Search    │   │   (LLM)     │     │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Configs    │    │   Prompts    │    │   Models     │       │
│  │   (YAML)     │    │   (TXT)      │    │  (Ollama)    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **External Harness**: Evaluation runs as Python scripts, not embedded in Rust binary
2. **Config-Driven**: All experiments defined in YAML files for reproducibility
3. **MCP-Compatible**: Can query via MCP tools or direct HTTP for flexibility
4. **Incremental**: Start with MVP metrics, add complexity over time

---

## 2. Ground Truth Dataset

### 2.1 Dataset Size & Statistical Power

| Queries | Margin of Error (95% CI) | Use Case |
|---------|--------------------------|----------|
| 30 | ±18% | Quick sanity checks |
| 50 | ±14% | MVP - detect major regressions |
| 100 | ±10% | Standard evaluation |
| 200+ | ±7% | Fine-grained model selection |

**Recommendation**: Start with 50 queries (20 manual "golden" + 30 synthetic)

### 2.2 Query Categories

Based on the corpus (70 documents on quant finance/trading/ML):

| Category | Description | Target Queries |
|----------|-------------|----------------|
| `options_derivatives` | Black-Scholes, Greeks, volatility, pricing | 8-10 |
| `portfolio_management` | Optimization, risk, allocation, Markowitz | 8-10 |
| `algorithmic_trading` | Strategies, backtesting, execution, signals | 8-10 |
| `machine_learning` | Feature engineering, time series ML, regime detection | 8-10 |
| `risk_management` | VaR, CVaR, stress testing, hedging | 6-8 |
| `time_series` | Forecasting, ARIMA, neural networks, seasonality | 6-8 |
| `data_engineering` | Python, Polars, data cleaning, APIs | 4-6 |
| `rejection` | Queries with NO answer in corpus (negative testing) | 5-10 |

### 2.3 Query Difficulty Levels

| Level | Description | Example |
|-------|-------------|---------|
| `easy` | Single-source factual answer | "What is the formula for delta?" |
| `medium` | Requires synthesis from 2-3 chunks | "How do managed futures strategies work?" |
| `hard` | Cross-domain or nuanced answer | "Compare ML approaches to options pricing" |
| `adversarial` | Keyword-heavy but wrong domain | "What is the best Python time to execute?" |

### 2.4 Ground Truth Schema

**MVP Schema** (simplified per ensemble review - reduces annotation burden ~60%):

```jsonl
{
  "query_id": "Q001",
  "query": "How do I calculate implied volatility using Black-Scholes?",
  "category": "options_derivatives",
  "difficulty": "medium",
  "is_rejection": false,
  "gold_references": [
    {
      "document": "Black Scholes with Python (Van Der Post 2024).pdf",
      "page": 45,
      "relevance": 3
    },
    {
      "document": "Option Volatility and Pricing (Natenberg 2015).pdf",
      "page": 112,
      "relevance": 3
    }
  ],
  "expected_answer_gist": "IV calculation via Newton-Raphson or bisection on Black-Scholes formula",
  "notes": "Multiple valid approaches exist"
}
```

**Schema Design Rationale:**
- **Page-level granularity**: Sufficient for MVP retrieval debugging; avoids fuzzy text matching issues
- **No partial/negative chunks**: Reduces annotation complexity; negatives are implicit (anything not gold)
- **`is_rejection` flag**: Marks queries with NO expected answer in corpus
- **`expected_answer_gist`**: Brief summary for future LLM-as-Judge Answer Quality testing (Phase 3)

### 2.5 Fuzzy Matching Strategy

Since chunk IDs change on reindex, evaluation uses fuzzy matching:

```python
def matches_gold_reference(retrieved_chunk, gold_ref) -> bool:
    """Match retrieved chunk to gold reference by document + page"""
    # 1. Normalize document names (case-insensitive, strip extensions)
    doc_match = normalize_doc_name(retrieved_chunk["document"]) == \
                normalize_doc_name(gold_ref["document"])

    # 2. Page number match (allow ±1 for boundary chunks)
    page_match = abs(retrieved_chunk["page"] - gold_ref["page"]) <= 1

    return doc_match and page_match

def normalize_doc_name(name: str) -> str:
    """Normalize document names for matching"""
    return name.lower().replace(".pdf", "").strip()
```

**Implementation Notes:**
- Index by (doc, page) first to avoid O(n²) comparisons
- Page ±1 tolerance handles chunks spanning page boundaries
- Log ambiguous matches for manual review
- Each gold reference matched at most once (first-match wins)

### 2.6 Relevance Scale

| Score | Label | Definition |
|-------|-------|------------|
| 3 | `highly_relevant` | Directly answers the query with specific, actionable information |
| 2 | `relevant` | Contains useful information but incomplete or requires inference |
| 1 | `marginally_relevant` | Same topic but doesn't answer the specific query |
| 0 | `irrelevant` | Unrelated or actively misleading |

### 2.7 Ground Truth Generation Strategy

**Hybrid Approach (Recommended):**

1. **Synthetic Generation (30 queries)**
   - Use Gemini/Claude to read random chunks from corpus
   - Generate questions that the chunk would answer
   - Auto-label with source chunk as gold standard
   - Filter for quality and diversity

2. **Manual Golden Set (20 queries)**
   - Hand-craft high-value queries you'd actually ask
   - Manually identify gold chunks by searching corpus
   - Include edge cases and adversarial queries

3. **Rejection Queries (5-10 queries)**
   - Questions about topics NOT in the 70 documents
   - Expected behavior: low-confidence results or explicit "not found"
   - Examples: "What is the Fed's current interest rate policy?" (no current events)

---

## 3. Metrics Suite

### 3.1 Primary Metrics (MVP)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Hit Rate@k** | `1 if any(gold_chunk in top_k) else 0` | Did we retrieve ANY relevant chunk? |
| **MRR@k** | `1/rank_of_first_gold_chunk` | How high was the first relevant result? |
| **Latency p95** | `percentile(latencies, 95)` | Speed for 95% of queries |

### 3.2 Secondary Metrics (Phase 2)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **NDCG@k** | `DCG@k / IDCG@k` | Ranking quality with graded relevance |
| **Precision@k** | `relevant_in_top_k / k` | What fraction of top-k is relevant? |
| **Recall@k** | `relevant_in_top_k / total_relevant` | What fraction of all relevant did we find? |
| **Context Precision** | `relevant_chunks / retrieved_chunks` | How much noise in the context? |

### 3.3 Advanced Metrics (Phase 3)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Score Discrimination** | `std(scores) / mean(scores)` | Are scores spread out or clustered? (CV > 0.3 is good) |
| **Rank Correlation** | `spearman(predicted_rank, gold_rank)` | Does ranking match human judgment? |
| **Reranker Lift** | `NDCG_reranked - NDCG_embedding_only` | How much does reranker help? |
| **Rejection Accuracy** | `correct_rejections / rejection_queries` | Do we avoid false positives? |
| **Answer Faithfulness** | LLM-as-Judge score (0-1) | Does retrieved context support the expected answer? |

**Answer Faithfulness (RAGAS-style):**
For Phase 3, add LLM-as-Judge evaluation to measure whether retrieved context actually contains information needed to answer the query. Uses the `expected_answer_gist` from ground truth:

```python
def answer_faithfulness(query: str, retrieved_context: str, expected_gist: str) -> float:
    """LLM-as-Judge: Does context contain info to answer the query?

    Prompt an LLM to score 0-1 whether the retrieved context
    contains sufficient information to produce the expected answer.
    """
    prompt = f"""Given this query and retrieved context, rate 0-1 whether
the context contains enough information to answer the query.

Query: {query}
Expected answer should include: {expected_gist}
Retrieved context: {retrieved_context}

Score (0=no relevant info, 1=fully answers): """
    # Call evaluation LLM (e.g., GPT-4, Claude)
    return parse_score(llm_call(prompt))
```

### 3.4 Metric Interpretation Guide

```
Hit Rate@5:
  > 0.90: Excellent - almost always finding relevant content
  > 0.75: Good - usually finding relevant content
  > 0.50: Needs work - missing relevant content half the time
  < 0.50: Poor - fundamental retrieval issues

MRR@5:
  > 0.80: Excellent - relevant content usually in top 2
  > 0.50: Good - relevant content usually in top 3-4
  > 0.30: Needs work - relevant content often buried
  < 0.30: Poor - ranking not working

NDCG@5:
  > 0.80: Excellent ranking
  > 0.60: Good ranking
  > 0.40: Fair ranking
  < 0.40: Poor ranking
```

---

## 4. Evaluation Harness

### 4.1 Core Components

```python
# eval_runner.py - Main orchestrator
class EvalRunner:
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
        self.ground_truth = load_jsonl(self.config["ground_truth_path"])
        self.rag_client = RAGClient(self.config["rag_endpoint"])

    def run_evaluation(self) -> EvalResults:
        results = []
        for query in self.ground_truth:
            search_results = self.rag_client.search(
                query=query["query"],
                top_k=self.config["top_k"]
            )
            results.append(self.evaluate_query(query, search_results))
        return self.aggregate_metrics(results)

    def evaluate_query(self, query: dict, results: list) -> QueryResult:
        gold_ids = {c["chunk_id"] for c in query["gold_chunks"]}
        retrieved_ids = [r["chunk_id"] for r in results]

        return QueryResult(
            query_id=query["query_id"],
            hit_rate=self.calc_hit_rate(gold_ids, retrieved_ids),
            mrr=self.calc_mrr(gold_ids, retrieved_ids),
            ndcg=self.calc_ndcg(query, results),
            latency_ms=results.latency_ms
        )
```

### 4.2 RAG Client Interface

```python
# rag_client.py - Interface to rust-local-rag
class RAGClient:
    """Abstracts communication with rust-local-rag.

    Recommendation: Use HTTP mode for evaluation (simpler, faster).
    MCP mode available for testing actual production path.
    """

    def __init__(self, endpoint: str, mode: str = "http"):
        self.endpoint = endpoint
        self.mode = mode  # "http" (recommended) or "mcp"

    def search(self, query: str, top_k: int = 5) -> SearchResults:
        start = time.perf_counter()

        if self.mode == "mcp":
            results = self._search_via_mcp(query, top_k)
        else:
            results = self._search_via_http(query, top_k)

        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResults(results=results, latency_ms=latency_ms)

    def _search_via_http(self, query: str, top_k: int) -> list:
        """Direct HTTP call - recommended for evaluation"""
        import requests
        response = requests.post(
            f"{self.endpoint}/search",
            json={"query": query, "top_k": top_k},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["results"]

    def _search_via_mcp(self, query: str, top_k: int) -> list:
        """MCP via mcpjungle - use for production path testing"""
        # Subprocess to mcp-remote is viable but adds complexity
        # Prefer HTTP for evaluation; MCP for integration testing
        raise NotImplementedError("Use HTTP mode for MVP evaluation")
```

**Note**: HTTP mode is recommended for evaluation (simpler, no subprocess management).
MCP mode can be added later for integration testing if needed.

### 4.3 Metrics Calculator

```python
# metrics.py - IR metric implementations
from typing import List, Set
import math

def hit_rate_at_k(gold_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """Binary hit indicator: 1 if ANY gold chunk in top-k, else 0.

    Note: This is a per-query hit indicator, not recall@k (which would be
    the proportion of gold_ids retrieved). Use for "did we find anything relevant?"
    """
    assert k >= 1, f"k must be >= 1, got {k}"
    if not retrieved_ids:
        return 0.0
    return 1.0 if gold_ids & set(retrieved_ids[:k]) else 0.0

def mrr_at_k(gold_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """Mean Reciprocal Rank: 1/position of first gold chunk.

    Returns 0.0 if no gold chunk found in top-k.
    """
    assert k >= 1, f"k must be >= 1, got {k}"
    if not retrieved_ids:
        return 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gold_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevances: List[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain.

    Uses LINEAR gain variant: rel / log2(rank + 1)
    (Not exponential: (2^rel - 1) / log2(rank + 1))

    This is the standard TREC/academic formulation.
    """
    assert k >= 1, f"k must be >= 1, got {k}"
    if not relevances:
        return 0.0

    # Use min(k, len) to handle short result lists
    actual_k = min(k, len(relevances))
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:actual_k]))
    ideal = sorted(relevances, reverse=True)[:actual_k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(gold_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """Fraction of top-k that are relevant."""
    assert k >= 1, f"k must be >= 1, got {k}"
    if not retrieved_ids:
        return 0.0
    actual_k = min(k, len(retrieved_ids))
    relevant_count = len(gold_ids & set(retrieved_ids[:actual_k]))
    return relevant_count / actual_k

def context_precision(relevances: List[int]) -> float:
    """Fraction of retrieved chunks that are relevant (relevance > 0).

    Measures "noise" in retrieved context - lower noise = better for LLM consumption.
    """
    if not relevances:
        return 0.0
    return sum(1 for r in relevances if r > 0) / len(relevances)
```

**Implementation Notes:**
- Uses `math.log2` instead of `numpy.log2` to minimize dependencies
- All functions guard against `k < 1` and empty inputs
- NDCG uses linear gain (standard TREC formulation), not exponential

---

## 5. Experiment Configurations

### 5.1 Configuration Schema

```yaml
# configs/baseline.yaml
name: "baseline"
description: "Current production configuration"

# RAG Server Settings
rag_endpoint: "http://localhost:8080"
connection_mode: "mcp"  # or "http"

# Retrieval Settings
top_k: 5
retrieval_top_k: 15  # Initial candidates before reranking

# Model Settings
embedding_model: "embed-light"  # Quantized qwen3-embed
reranker_model: "phi4-mini"
reranker_enabled: true

# Hybrid Search Weights
dense_weight: 0.7   # Vector similarity weight
sparse_weight: 0.3  # Lexical/BM25 weight

# Reranker Settings
reranker_prompt: "prompts/reranker_v1.txt"
reranker_temperature: 0.1
reranker_timeout_seconds: 60

# Evaluation Settings
ground_truth_path: "eval/ground_truth.jsonl"
metrics: ["hit_rate@5", "mrr@5", "ndcg@5", "latency_p95"]
```

### 5.2 Experiment Variants

```yaml
# configs/embedding_only.yaml
name: "embedding_only"
description: "No reranking - pure vector similarity"
reranker_enabled: false
# ... rest same as baseline

# configs/high_recall.yaml
name: "high_recall"
description: "More candidates for reranking"
retrieval_top_k: 30
top_k: 10
# ... rest same as baseline

# configs/lexical_heavy.yaml
name: "lexical_heavy"
description: "More weight on keyword matching"
dense_weight: 0.4
sparse_weight: 0.6
# ... rest same as baseline
```

---

## 6. A/B Testing Framework

### 6.1 Comparison Runner

```python
# ab_runner.py - Statistical A/B comparison
class ABRunner:
    def __init__(self, config_a: str, config_b: str):
        self.runner_a = EvalRunner(config_a)
        self.runner_b = EvalRunner(config_b)

    def run_comparison(self) -> ABResults:
        results_a = self.runner_a.run_evaluation()
        results_b = self.runner_b.run_evaluation()

        return ABResults(
            config_a=results_a,
            config_b=results_b,
            statistical_tests=self.run_statistical_tests(results_a, results_b)
        )

    def run_statistical_tests(self, a: EvalResults, b: EvalResults) -> dict:
        """Paired tests since same queries evaluated"""
        return {
            "ndcg_paired_ttest": paired_ttest(a.ndcg_scores, b.ndcg_scores),
            "mrr_paired_ttest": paired_ttest(a.mrr_scores, b.mrr_scores),
            "bootstrap_ci": bootstrap_confidence_interval(
                a.ndcg_scores, b.ndcg_scores,
                n_bootstrap=1000
            )
        }
```

### 6.2 Statistical Significance

```python
def paired_ttest(scores_a: List[float], scores_b: List[float]) -> dict:
    """Paired t-test for same queries across two configs"""
    from scipy import stats

    differences = [a - b for a, b in zip(scores_a, scores_b)]
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
        "mean_difference": np.mean(differences),
        "std_difference": np.std(differences)
    }

def bootstrap_confidence_interval(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> dict:
    """Bootstrap CI for difference in means"""
    differences = []
    n = len(scores_a)

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        diff = np.mean([scores_a[i] for i in indices]) - \
               np.mean([scores_b[i] for i in indices])
        differences.append(diff)

    alpha = (1 - confidence) / 2
    lower = np.percentile(differences, alpha * 100)
    upper = np.percentile(differences, (1 - alpha) * 100)

    return {
        "mean_difference": np.mean(differences),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "significant": lower > 0 or upper < 0  # CI doesn't include 0
    }
```

### 6.3 A/B Report Format

```markdown
# A/B Test Report: baseline vs embedding_only

## Summary
- **Winner**: baseline (with reranking)
- **Confidence**: 95% CI, p < 0.01

## Metrics Comparison

| Metric | baseline | embedding_only | Difference | p-value |
|--------|----------|----------------|------------|---------|
| Hit Rate@5 | 0.88 | 0.82 | +0.06 | 0.023* |
| MRR@5 | 0.76 | 0.65 | +0.11 | 0.008** |
| NDCG@5 | 0.81 | 0.72 | +0.09 | 0.004** |
| Latency p95 | 2.3s | 0.4s | +1.9s | - |

## Per-Category Breakdown
[... category-level metrics ...]

## Recommendation
Reranking improves quality significantly (+11% MRR) at 6x latency cost.
For latency-sensitive applications, consider embedding_only.
```

---

## 7. Prompt Engineering Harness

### 7.1 Prompt Template System

```
prompts/
├── reranker_v1.txt      # Current production prompt
├── reranker_v2_minimal.txt  # Minimal instructions
├── reranker_v3_domain.txt   # Domain-specific instructions
├── reranker_v4_cot.txt      # Chain-of-thought format
└── variations/
    ├── scale_0_100.txt      # 0-100 scoring
    ├── scale_1_5.txt        # 1-5 Likert scale
    └── binary.txt           # Yes/No with confidence
```

### 7.2 Prompt Experiment Config

```yaml
# configs/prompt_experiment.yaml
name: "prompt_cot"
description: "Chain-of-thought scoring format"
reranker_prompt: "prompts/reranker_v4_cot.txt"
# ... rest same as baseline
```

### 7.3 Prompt Variations to Test

| Variation | Hypothesis | Metrics to Watch |
|-----------|------------|------------------|
| **Minimal** | Less instruction = faster, maybe less consistent | Latency, score variance |
| **Domain-specific** | "You are a quant finance expert" = better domain judgment | NDCG on finance queries |
| **Chain-of-thought** | Reasoning before score = better discrimination | Score distribution entropy |
| **Few-shot (3 examples)** | More examples = better calibration | Score correlation with gold |
| **Binary + confidence** | Simpler task = faster, more consistent | Latency, binary accuracy |

### 7.4 Prompt Consistency Test

```python
def test_prompt_consistency(prompt_path: str, n_repeats: int = 5) -> dict:
    """Test if same query+doc gets similar scores across runs"""
    scores_per_query = defaultdict(list)

    for _ in range(n_repeats):
        for query, doc in test_pairs:
            score = reranker.score(query, doc, prompt_path)
            scores_per_query[(query, doc)].append(score)

    variances = [np.var(scores) for scores in scores_per_query.values()]

    return {
        "mean_variance": np.mean(variances),
        "max_variance": max(variances),
        "consistent": np.mean(variances) < 0.05  # threshold
    }
```

---

## 8. Model Comparison Matrix

### 8.1 Embedding Models to Test

| Model | Dimensions | Context | Speed | Notes |
|-------|-----------|---------|-------|-------|
| `embed-light` | 1024 | 8192 | Fast | **Current baseline** - quantized qwen3-embed |
| `embed-heavy` | 1024 | 8192 | Medium | Full qwen3-embed (2.9 GB) - already installed |
| `mxbai-embed-large` | 1024 | 512 | Medium | Higher dim, popular |
| `snowflake-arctic-embed` | 1024 | 512 | Medium | Strong benchmarks |
| `nomic-embed-text` | 768 | 8192 | Fast | Code default fallback |

### 8.2 Reranker Models to Test

**Current Approach: LLM-based Reranking**

| Model | Size | Est. Speed | Notes |
|-------|------|------------|-------|
| `phi4-mini` | 3.8B | ~3s/chunk | Current, fast |
| `llama3.1:8b` | 8B | ~8s/chunk | More capable |
| `mistral:7b` | 7B | ~6s/chunk | Good reasoning |
| `gemma2:9b` | 9B | ~10s/chunk | Google quality |
| `qwen2.5:7b` | 7B | ~7s/chunk | Strong coding/reasoning |

**Alternative: Dedicated Cross-Encoders** (Future consideration)

Per Gemini review: dedicated cross-encoder models are typically 10x faster and often more accurate than LLM-based reranking. Consider benchmarking if latency becomes an issue:

| Model | Type | Speed | Notes |
|-------|------|-------|-------|
| `bge-reranker-v2-m3` | Cross-encoder | ~0.1s/chunk | BAAI, multilingual |
| `mxbai-rerank-xsmall-v1` | Cross-encoder | ~0.05s/chunk | mixedbread.ai, tiny |
| `ms-marco-MiniLM-L-6-v2` | Cross-encoder | ~0.03s/chunk | Classic, fast |

**Trade-off**: LLM rerankers offer flexibility (custom prompts, domain adaptation) at higher latency cost. Cross-encoders are faster but less adaptable.

### 8.3 Full Experiment Matrix

**Phase 2 Target**: Test top combinations, not full 5×5 matrix

```
Priority 1 (MVP):
- baseline: embed-light + phi4-mini (current production)
- embedding_only: embed-light + none

Priority 2 (Embedding comparison):
- embed-light vs embed-heavy (both already installed!)
- Optional: mxbai-embed-large if results warrant
- All with phi4-mini reranker

Priority 3 (Reranker comparison):
- embed-light embedding (keep constant)
- phi4-mini vs qwen2.5:7b vs llama3.1

Priority 4 (Hybrid weights):
- Dense:Sparse = 0.8:0.2, 0.7:0.3, 0.6:0.4, 0.5:0.5
- Best embedding + best reranker
```

### 8.4 Pareto Frontier Visualization

```python
def plot_pareto_frontier(results: List[ConfigResult]):
    """Plot quality vs latency tradeoff"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in results:
        ax.scatter(r.latency_p95, r.ndcg_at_5, label=r.config_name)
        ax.annotate(r.config_name, (r.latency_p95, r.ndcg_at_5))

    # Draw Pareto frontier
    pareto_points = compute_pareto_frontier(results)
    ax.plot([p.latency_p95 for p in pareto_points],
            [p.ndcg_at_5 for p in pareto_points],
            'r--', label='Pareto Frontier')

    ax.set_xlabel('Latency p95 (seconds)')
    ax.set_ylabel('NDCG@5')
    ax.set_title('Quality vs Latency Tradeoff')
    ax.legend()

    plt.savefig('reports/pareto_frontier.png')
```

---

## 9. Implementation Phases

### Phase 1: MVP (Week 1)

**Deliverables:**
- [ ] Ground truth dataset: 50 queries (20 golden + 30 synthetic)
- [ ] Basic evaluation harness: `eval_runner.py`, `metrics.py`
- [ ] MVP metrics: Hit Rate@5, MRR@5, Latency p95
- [ ] Single config: baseline only
- [ ] Simple report: markdown output

**Success Criteria:**
- Can run `python eval/run.py --config baseline` and get metrics
- Metrics are reproducible (same results on same data)

### Phase 2: A/B Testing (Week 2)

**Deliverables:**
- [ ] Multiple configs: baseline, embedding_only, lexical_heavy
- [ ] A/B comparison runner with statistical tests
- [ ] NDCG metric implementation
- [ ] Per-category breakdown
- [ ] Comparison report format

**Success Criteria:**
- Can answer "Is reranking worth the latency cost?"
- Statistical significance testing works

### Phase 3: Model Comparison (Week 3)

**Deliverables:**
- [ ] Script to switch embedding/reranker models
- [ ] Test 2-3 embedding models
- [ ] Test 2-3 reranker models
- [ ] Pareto frontier visualization
- [ ] Hybrid weight tuning

**Success Criteria:**
- Can identify optimal model combination for quality/latency tradeoff

### Phase 4: Prompt Engineering (Week 4)

**Deliverables:**
- [ ] Prompt template system
- [ ] 3-4 prompt variations
- [ ] Consistency testing
- [ ] Score distribution analysis

**Success Criteria:**
- Have data-driven prompt selection

---

## 10. File Structure

```
rust-local-rag/
├── eval/                           # Evaluation framework
│   ├── __init__.py
│   ├── run.py                      # CLI entry point
│   ├── eval_runner.py              # Main orchestrator
│   ├── ab_runner.py                # A/B comparison
│   ├── metrics.py                  # Metric calculations
│   ├── rag_client.py               # Interface to RAG server
│   ├── report_generator.py         # Output reports
│   │
│   ├── configs/                    # Experiment configurations
│   │   ├── baseline.yaml
│   │   ├── embedding_only.yaml
│   │   ├── lexical_heavy.yaml
│   │   └── ...
│   │
│   ├── ground_truth/               # Labeled evaluation data
│   │   ├── golden_queries.jsonl    # Manually curated
│   │   ├── synthetic_queries.jsonl # LLM-generated
│   │   ├── rejection_queries.jsonl # Negative testing
│   │   └── README.md               # Dataset documentation
│   │
│   └── reports/                    # Generated reports
│       ├── baseline_2024-01-15.md
│       ├── ab_reranking_2024-01-16.md
│       └── pareto_frontier.png
│
├── prompts/                        # (existing) Reranker prompts
│   ├── reranker.txt                # Current production
│   └── variations/                 # Experimental prompts
│
└── docs/
    └── RAG_EVALUATION_FRAMEWORK_SPEC.md  # This document
```

---

## 11. Design Decisions (Resolved)

All design questions have been resolved. Documenting decisions for reference:

### User Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Ground Truth Creation** | Claude + Ensemble with PDF context, manual curation | Avoid "poisoning the well" by not using RAG tools for query generation |
| **Evaluation Frequency** | Manual only | Run on major changes (model swap, prompt change), not CI |
| **Model Testing Scope** | `embed-light` vs `embed-heavy` first, keep `phi4-mini` reranker | Both already installed; start with what we have |
| **Success Thresholds** | Hit Rate@5 > 0.80, Latency p95 < 5s | Personal research use case, not production SLA |
| **Rejection Testing** | Include 5-10 rejection queries | Finance context: false confidence worse than "I don't know" |

### Technical Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| **MCP vs HTTP Access** | HTTP primary, MCP optional | Per Codex review: HTTP is simpler for evaluation; MCP adds subprocess complexity. Use HTTP for eval, MCP for integration testing. |
| **Chunk ID Matching** | Fuzzy match (document + page) | Per ensemble review: page-level granularity sufficient for MVP; avoids PDF text artifact issues |
| **Multi-run Averaging** | Single run | Temperature 0.1 = low variance; add averaging later if needed |
| **Ground Truth Schema** | Simplified MVP schema | Per Gemini review: removed partial/negative chunks, reduces annotation burden ~60% |
| **NDCG Variant** | Linear gain | Per Codex review: standard TREC formulation, not exponential |

---

## Appendix A: Sample Ground Truth Queries

```jsonl
{"query_id": "Q001", "query": "How do I calculate implied volatility using Black-Scholes?", "category": "options_derivatives", "difficulty": "medium"}
{"query_id": "Q002", "query": "What is the difference between delta and gamma in options trading?", "category": "options_derivatives", "difficulty": "easy"}
{"query_id": "Q003", "query": "How does the Kelly criterion apply to position sizing?", "category": "portfolio_management", "difficulty": "medium"}
{"query_id": "Q004", "query": "What are the main components of a trend-following strategy?", "category": "algorithmic_trading", "difficulty": "easy"}
{"query_id": "Q005", "query": "How do I detect regime changes in financial time series?", "category": "machine_learning", "difficulty": "hard"}
{"query_id": "Q006", "query": "What is Value at Risk and how is it calculated?", "category": "risk_management", "difficulty": "easy"}
{"query_id": "Q007", "query": "How do LSTM networks compare to transformers for time series forecasting?", "category": "time_series", "difficulty": "hard"}
{"query_id": "Q008", "query": "What is the best way to clean financial tick data in Python?", "category": "data_engineering", "difficulty": "medium"}
{"query_id": "Q009", "query": "What is the current Federal Reserve interest rate?", "category": "rejection", "difficulty": "adversarial"}
{"query_id": "Q010", "query": "How do I apply reinforcement learning to portfolio optimization?", "category": "machine_learning", "difficulty": "hard"}
```

---

## Appendix B: Dependencies

```toml
# pyproject.toml (uv)
[project]
name = "rust-local-rag-eval"
version = "0.1.0"
requires-python = ">=3.11"

# MVP dependencies (Phase 1)
dependencies = [
    "pyyaml>=6.0",       # Config loading
    "requests>=2.31",    # HTTP client for RAG API
    "rich>=13.0",        # Pretty CLI output
    "typer>=0.9",        # CLI framework
]

[project.optional-dependencies]
# Phase 2: A/B testing with statistics
phase2 = [
    "numpy>=1.24",       # Bootstrap CI, array ops
    "scipy>=1.11",       # Statistical tests (t-test)
]

# Phase 3: Visualization and advanced metrics
phase3 = [
    "pandas>=2.0",       # Data manipulation
    "matplotlib>=3.7",   # Plotting
    "seaborn>=0.12",     # Statistical visualization
]

# Optional: TREC-style evaluation
advanced = [
    "pytrec_eval>=0.5",  # TREC evaluation metrics
    "ranx>=0.3",         # Advanced IR metrics
]
```

**Dependency Philosophy:**
- MVP uses only `pyyaml`, `requests`, `rich`, `typer` (no numpy!)
- Metrics use `math.log2` instead of `numpy.log2` to minimize deps
- Add numpy/scipy only when A/B testing is needed (Phase 2)
- Visualization deps only for Phase 3

---

**Document Status**: Approved (2025-12-07)

**Implementation Plan**:
1. Create `eval/` directory structure
2. Generate ground truth dataset:
   - Claude + Ensemble reads actual PDFs (not via RAG) to create queries
   - Cross-reference with web for domain accuracy
   - Manual curation of 20 golden + 30 synthetic + 5-10 rejection queries
3. Implement MVP harness (Hit Rate@5, MRR@5, Latency p95)
4. Run first baseline evaluation with `embed-light` + `phi4-mini`
5. Iterate based on results
