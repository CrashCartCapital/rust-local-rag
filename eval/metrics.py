"""IR metric implementations for RAG evaluation.

All metrics follow standard TREC/academic formulations.
Uses math.log2 instead of numpy to minimize dependencies.

Metrics:
- hit_rate_at_k: Binary indicator if any gold in top-k
- mrr_at_k: Mean Reciprocal Rank
- ndcg_at_k: Normalized Discounted Cumulative Gain (linear gain variant)
- precision_at_k: Fraction of top-k that are relevant
- context_precision: Fraction of retrieved that are relevant (noise measure)
"""

from typing import List, Set
import math


def hit_rate_at_k(gold_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """Binary hit indicator: 1 if ANY gold chunk in top-k, else 0.

    Note: This is a per-query hit indicator, not recall@k (which would be
    the proportion of gold_ids retrieved). Use for "did we find anything relevant?"

    Args:
        gold_ids: Set of gold chunk identifiers (document + page combos)
        retrieved_ids: Ordered list of retrieved chunk identifiers
        k: Number of top results to consider

    Returns:
        1.0 if any gold in top-k, 0.0 otherwise
    """
    assert k >= 1, f"k must be >= 1, got {k}"
    if not retrieved_ids:
        return 0.0
    return 1.0 if gold_ids & set(retrieved_ids[:k]) else 0.0


def mrr_at_k(gold_ids: Set[str], retrieved_ids: List[str], k: int) -> float:
    """Mean Reciprocal Rank: 1/position of first gold chunk.

    Returns 0.0 if no gold chunk found in top-k.

    Args:
        gold_ids: Set of gold chunk identifiers
        retrieved_ids: Ordered list of retrieved chunk identifiers
        k: Number of top results to consider

    Returns:
        1/rank of first gold hit, or 0.0 if no hit
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

    Args:
        relevances: List of relevance scores (0-3) in retrieval order
        k: Number of top results to consider

    Returns:
        NDCG score in [0.0, 1.0]
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
    """Fraction of top-k that are relevant.

    Args:
        gold_ids: Set of gold chunk identifiers
        retrieved_ids: Ordered list of retrieved chunk identifiers
        k: Number of top results to consider

    Returns:
        Precision score in [0.0, 1.0]
    """
    assert k >= 1, f"k must be >= 1, got {k}"
    if not retrieved_ids:
        return 0.0
    actual_k = min(k, len(retrieved_ids))
    relevant_count = len(gold_ids & set(retrieved_ids[:actual_k]))
    return relevant_count / actual_k


def context_precision(relevances: List[int]) -> float:
    """Fraction of retrieved chunks that are relevant (relevance > 0).

    Measures "noise" in retrieved context - lower noise = better for LLM consumption.

    Args:
        relevances: List of relevance scores for all retrieved chunks

    Returns:
        Fraction of chunks with relevance > 0
    """
    if not relevances:
        return 0.0
    return sum(1 for r in relevances if r > 0) / len(relevances)


# Convenience function for aggregating metrics across queries
def aggregate_metrics(query_results: List[dict]) -> dict:
    """Aggregate per-query metrics into summary statistics.

    Args:
        query_results: List of dicts with 'hit_rate', 'mrr', 'ndcg', 'latency_ms'

    Returns:
        Dict with mean values and percentiles for latency
    """
    if not query_results:
        return {}

    n = len(query_results)

    # Calculate means
    hit_rates = [r.get("hit_rate", 0.0) for r in query_results]
    mrrs = [r.get("mrr", 0.0) for r in query_results]
    ndcgs = [r.get("ndcg", 0.0) for r in query_results]
    latencies = [r.get("latency_ms", 0.0) for r in query_results]

    # Sort latencies for percentiles
    sorted_latencies = sorted(latencies)

    def percentile(data: List[float], p: float) -> float:
        """Calculate percentile without numpy."""
        if not data:
            return 0.0
        idx = int(len(data) * p / 100)
        idx = min(idx, len(data) - 1)
        return data[idx]

    return {
        "hit_rate_mean": sum(hit_rates) / n,
        "mrr_mean": sum(mrrs) / n,
        "ndcg_mean": sum(ndcgs) / n,
        "latency_mean_ms": sum(latencies) / n,
        "latency_p50_ms": percentile(sorted_latencies, 50),
        "latency_p95_ms": percentile(sorted_latencies, 95),
        "latency_p99_ms": percentile(sorted_latencies, 99),
        "n_queries": n,
    }
