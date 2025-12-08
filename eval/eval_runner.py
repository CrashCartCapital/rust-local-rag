"""Evaluation runner - main orchestrator for RAG evaluation.

Loads ground truth, runs queries through RAG, calculates metrics,
and generates reports.
"""

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime

import yaml

from .metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    context_precision,
    aggregate_metrics,
)
from .rag_client import (
    RAGClient,
    SearchResponse,
    matches_gold_reference,
    make_chunk_key,
)


@dataclass
class QueryResult:
    """Result of evaluating a single query."""

    query_id: str
    query: str
    category: str
    hit_rate: float
    mrr: float
    ndcg: float
    precision: float
    latency_ms: float
    retrieved_docs: List[str] = field(default_factory=list)
    gold_docs: List[str] = field(default_factory=list)
    is_rejection: bool = False
    notes: str = ""


@dataclass
class EvalConfig:
    """Evaluation configuration loaded from YAML."""

    name: str
    description: str
    rag_endpoint: str
    connection_mode: str
    top_k: int
    ground_truth_path: str
    metrics: List[str]

    # Optional settings with defaults
    retrieval_top_k: int = 15
    embedding_model: str = "embed-light"
    reranker_model: str = "phi4-mini"
    reranker_enabled: bool = True
    page_tolerance: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            rag_endpoint=data.get("rag_endpoint", "http://localhost:8080"),
            connection_mode=data.get("connection_mode", "http"),
            top_k=data.get("top_k", 5),
            ground_truth_path=data.get("ground_truth_path", "eval/ground_truth/queries.jsonl"),
            metrics=data.get("metrics", ["hit_rate@5", "mrr@5", "latency_p95"]),
            retrieval_top_k=data.get("retrieval_top_k", 15),
            embedding_model=data.get("embedding_model", "embed-light"),
            reranker_model=data.get("reranker_model", "phi4-mini"),
            reranker_enabled=data.get("reranker_enabled", True),
            page_tolerance=data.get("page_tolerance", 1),
        )


@dataclass
class GroundTruthQuery:
    """A single ground truth query with expected results."""

    query_id: str
    query: str
    category: str
    difficulty: str
    is_rejection: bool
    gold_references: List[Dict]  # [{document, page, relevance}]
    expected_answer_gist: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruthQuery":
        """Parse from JSONL dict with validation."""
        query_id = data.get("query_id", "")
        is_rejection = data.get("is_rejection", False)
        gold_references = data.get("gold_references", [])

        # Validate: non-rejection queries must have gold references
        if not is_rejection and not gold_references:
            raise ValueError(
                f"Query {query_id}: non-rejection queries must have at least one gold_reference"
            )

        # Validate: rejection queries should have empty gold references
        if is_rejection and gold_references:
            warnings.warn(
                f"Query {query_id}: rejection query has gold_references (expected empty)"
            )

        return cls(
            query_id=query_id,
            query=data.get("query", ""),
            category=data.get("category", "unknown"),
            difficulty=data.get("difficulty", "medium"),
            is_rejection=is_rejection,
            gold_references=gold_references,
            expected_answer_gist=data.get("expected_answer_gist", ""),
            notes=data.get("notes", ""),
        )


class EvalRunner:
    """Main evaluation runner."""

    def __init__(self, config: EvalConfig):
        """Initialize evaluation runner.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.client = RAGClient(
            endpoint=config.rag_endpoint,
            mode=config.connection_mode,
        )
        self.ground_truth: List[GroundTruthQuery] = []
        self._load_ground_truth()

    def _load_ground_truth(self) -> None:
        """Load ground truth queries from JSONL file."""
        gt_path = Path(self.config.ground_truth_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        self.ground_truth = []
        with open(gt_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    data = json.loads(line)
                    self.ground_truth.append(GroundTruthQuery.from_dict(data))

    def _build_gold_keys(self, query: GroundTruthQuery) -> Set[str]:
        """Build set of gold keys for matching."""
        keys = set()
        for ref in query.gold_references:
            # Add keys for page ± tolerance, clamping to valid page numbers (>= 1)
            for page_offset in range(-self.config.page_tolerance, self.config.page_tolerance + 1):
                page = ref["page"] + page_offset
                if page >= 1:  # Only valid page numbers
                    key = make_chunk_key(ref["document"], page)
                    keys.add(key)
        return keys

    def _get_relevance_for_result(
        self, result, gold_refs: List[Dict], tolerance: int
    ) -> int:
        """Get relevance score for a retrieved result.

        Returns the highest relevance across all matching gold references.
        """
        max_relevance = 0
        for ref in gold_refs:
            if matches_gold_reference(
                result, ref["document"], ref["page"], tolerance
            ):
                relevance = ref.get("relevance", 3)
                max_relevance = max(max_relevance, relevance)
        return max_relevance

    def evaluate_query(self, query: GroundTruthQuery) -> QueryResult:
        """Evaluate a single query against RAG.

        Args:
            query: Ground truth query

        Returns:
            QueryResult with metrics
        """
        # Execute search
        response: SearchResponse = self.client.search(
            query=query.query, top_k=self.config.top_k
        )

        # Build keys for retrieved results
        retrieved_keys = [
            make_chunk_key(r.document, r.page) for r in response.results
        ]
        retrieved_docs = [r.document for r in response.results]

        # Build gold keys
        gold_keys = self._build_gold_keys(query)
        gold_docs = [ref["document"] for ref in query.gold_references]

        # Get relevance scores for retrieved results
        relevances = [
            self._get_relevance_for_result(
                r, query.gold_references, self.config.page_tolerance
            )
            for r in response.results
        ]

        # Handle rejection queries
        if query.is_rejection:
            # For rejection queries, success is NOT finding ANY results
            # Since gold_references is empty, we can't use gold_keys matching
            # Instead, success = no results returned (or very low scores)
            has_results = len(retrieved_keys[:self.config.top_k]) > 0
            rejection_success = 0.0 if has_results else 1.0
            return QueryResult(
                query_id=query.query_id,
                query=query.query,
                category=query.category,
                hit_rate=rejection_success,  # 1.0 if no results, 0.0 if results found
                mrr=0.0,  # N/A for rejection
                ndcg=0.0,  # N/A for rejection
                precision=0.0,  # N/A for rejection
                latency_ms=response.latency_ms,
                retrieved_docs=retrieved_docs,
                gold_docs=gold_docs,
                is_rejection=True,
                notes="Rejection query - success if no results returned",
            )

        # Calculate metrics for normal queries
        k = self.config.top_k
        return QueryResult(
            query_id=query.query_id,
            query=query.query,
            category=query.category,
            hit_rate=hit_rate_at_k(gold_keys, retrieved_keys, k),
            mrr=mrr_at_k(gold_keys, retrieved_keys, k),
            ndcg=ndcg_at_k(relevances, k),
            precision=precision_at_k(gold_keys, retrieved_keys, k),
            latency_ms=response.latency_ms,
            retrieved_docs=retrieved_docs,
            gold_docs=gold_docs,
            is_rejection=False,
        )

    def run_evaluation(self, verbose: bool = False) -> Dict:
        """Run full evaluation on all ground truth queries.

        Args:
            verbose: Print progress during evaluation

        Returns:
            Dict with results and aggregated metrics
        """
        if not self.ground_truth:
            raise ValueError("No ground truth queries loaded")

        # Check server health
        if not self.client.health_check():
            raise ConnectionError(
                f"RAG server not reachable at {self.config.rag_endpoint}"
            )

        results: List[QueryResult] = []
        start_time = time.perf_counter()

        for i, query in enumerate(self.ground_truth):
            if verbose:
                print(f"[{i+1}/{len(self.ground_truth)}] {query.query_id}: {query.query[:50]}...")

            result = self.evaluate_query(query)
            results.append(result)

            if verbose:
                status = "HIT" if result.hit_rate > 0 else "MISS"
                print(f"  -> {status} (MRR={result.mrr:.2f}, latency={result.latency_ms:.0f}ms)")

        total_time = time.perf_counter() - start_time

        # Separate rejection and normal results for aggregation
        normal_results = [r for r in results if not r.is_rejection]
        rejection_results = [r for r in results if r.is_rejection]

        # Convert to dicts for aggregation
        normal_dicts = [
            {
                "hit_rate": r.hit_rate,
                "mrr": r.mrr,
                "ndcg": r.ndcg,
                "latency_ms": r.latency_ms,
            }
            for r in normal_results
        ]

        aggregated = aggregate_metrics(normal_dicts)

        # Add rejection accuracy if we have rejection queries
        if rejection_results:
            rejection_accuracy = sum(r.hit_rate for r in rejection_results) / len(rejection_results)
            aggregated["rejection_accuracy"] = rejection_accuracy
            aggregated["n_rejection_queries"] = len(rejection_results)

        return {
            "config": self.config.name,
            "timestamp": datetime.now().isoformat(),
            "total_time_seconds": total_time,
            "metrics": aggregated,
            "per_query_results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "category": r.category,
                    "hit_rate": r.hit_rate,
                    "mrr": r.mrr,
                    "ndcg": r.ndcg,
                    "precision": r.precision,
                    "latency_ms": r.latency_ms,
                    "is_rejection": r.is_rejection,
                }
                for r in results
            ],
            "per_category_metrics": self._aggregate_by_category(results),
        }

    def _aggregate_by_category(self, results: List[QueryResult]) -> Dict:
        """Aggregate metrics by query category."""
        categories: Dict[str, List[QueryResult]] = {}
        for r in results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        category_metrics = {}
        for cat, cat_results in categories.items():
            # Skip rejection queries for category metrics
            normal = [r for r in cat_results if not r.is_rejection]
            if normal:
                category_metrics[cat] = {
                    "n_queries": len(normal),
                    "hit_rate_mean": sum(r.hit_rate for r in normal) / len(normal),
                    "mrr_mean": sum(r.mrr for r in normal) / len(normal),
                    "latency_mean_ms": sum(r.latency_ms for r in normal) / len(normal),
                }

        return category_metrics
