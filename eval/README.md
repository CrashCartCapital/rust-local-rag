# RAG Evaluation Framework

A Python-based evaluation harness for measuring rust-local-rag retrieval quality and performance.

## Quick Start

```bash
# Ensure RAG server is running
cd /path/to/rust-local-rag
make run

# Run baseline evaluation
cd eval
python -m eval.run evaluate --config baseline -v
```

## Architecture

```
eval/
├── __init__.py              # Package marker
├── run.py                   # CLI entry point (typer-based)
├── eval_runner.py           # Main evaluation orchestrator
├── rag_client.py            # HTTP client for RAG server
├── metrics.py               # IR metric implementations
├── pyproject.toml           # Dependencies (uv/pip)
│
├── configs/                 # Experiment configurations
│   ├── baseline.yaml        # Production config (embed-light + phi4-mini)
│   └── embedding_only.yaml  # No reranking baseline
│
├── ground_truth/            # Labeled evaluation data
│   ├── queries.jsonl        # 50 queries (45 retrieval + 5 rejection)
│   └── README.md            # Dataset documentation
│
└── reports/                 # Generated reports
    ├── BASELINE_EVALUATION_SUMMARY.md  # Latest evaluation results
    └── baseline_run.log     # Raw execution logs
```

## Usage

### Running Evaluations

```bash
# Basic evaluation
python -m eval.run evaluate --config baseline

# Verbose output (per-query results)
python -m eval.run evaluate --config baseline -v

# With specific output file
python -m eval.run evaluate --config baseline -o reports/my_run.json
```

### Available Configs

| Config | Description | Use Case |
|--------|-------------|----------|
| `baseline` | embed-light + phi4-mini reranker | Production quality testing |
| `embedding_only` | No reranking | Latency testing, ablation |

## Metrics

### Primary Metrics (MVP)

| Metric | Description | Target |
|--------|-------------|--------|
| **Hit Rate@k** | Did we find ANY relevant document in top-k? | >80% |
| **MRR@k** | Reciprocal rank of first relevant result | >0.50 |
| **Latency p95** | 95th percentile response time | <5s |

### Secondary Metrics (Phase 2)

- NDCG@k - Ranking quality with graded relevance
- Precision@k - Fraction of top-k that are relevant
- Context Precision - Noise ratio in retrieved chunks

## Ground Truth Dataset

50 queries covering 8 categories of quant finance topics:

| Category | Queries | Description |
|----------|---------|-------------|
| options_derivatives | 10 | Black-Scholes, Greeks, volatility |
| algorithmic_trading | 10 | Strategies, backtesting, signals |
| portfolio_management | 8 | Optimization, risk, allocation |
| risk_management | 5 | VaR, CVaR, stress testing |
| machine_learning | 6 | ML in finance, time series |
| time_series | 4 | Forecasting, ARIMA |
| data_engineering | 2 | Python, data cleaning |
| rejection | 5 | Queries with NO answer in corpus |

### Query Schema

```json
{
  "query_id": "Q001",
  "query": "How do I implement the triple barrier labeling method?",
  "category": "algorithmic_trading",
  "difficulty": "medium",
  "is_rejection": false,
  "gold_references": [
    {"document": "Advances in Financial ML (Lopez de Prado 2018).pdf", "page": 45, "relevance": 3}
  ],
  "expected_answer_gist": "Triple barrier method for labeling trades",
  "notes": ""
}
```

## Configuration Schema

```yaml
name: "config_name"
description: "Human-readable description"

# Server Connection
rag_endpoint: "http://localhost:3046"
connection_mode: "http"

# Retrieval Settings
top_k: 5                    # Final results to return
retrieval_top_k: 15         # Candidates before reranking

# Model Settings
embedding_model: "embed-light"
reranker_model: "phi4-mini"
reranker_enabled: true

# Hybrid Search Weights
dense_weight: 0.7           # Vector similarity
sparse_weight: 0.3          # Lexical/BM25

# Evaluation Settings
ground_truth_path: "eval/ground_truth/queries.jsonl"
metrics: ["hit_rate@5", "mrr@5", "ndcg@5", "latency_p95"]
page_tolerance: 15          # Allow page variance in matching
```

## Latest Results

**Baseline Evaluation (2025-12-08)**:

| Metric | Value |
|--------|-------|
| Document Hit Rate@5 | 77.8% (35/45) |
| Latency p50 | ~31s |
| Latency p95 | ~42s |

**Key Finding**: 9 of 10 "misses" are valid alternative sources. True miss rate is ~2%.

See `reports/BASELINE_EVALUATION_SUMMARY.md` for detailed analysis.

## Dependencies

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

Core dependencies:
- `pyyaml` - Config loading
- `requests` - HTTP client
- `rich` - Pretty CLI output
- `typer` - CLI framework

## Development

### Adding New Queries

1. Edit `ground_truth/queries.jsonl`
2. Follow the schema documented in `ground_truth/README.md`
3. Avoid using RAG search to create ground truth (prevents bias)

### Adding New Configs

1. Create `configs/my_config.yaml`
2. Copy from `baseline.yaml` as template
3. Modify settings as needed
4. Run: `python -m eval.run evaluate --config my_config`

### Adding New Metrics

1. Implement in `metrics.py`
2. Add to config `metrics` list
3. Update `eval_runner.py` to calculate and report

## Roadmap

- [x] Phase 1: MVP evaluation framework
- [x] Phase 1: Ground truth dataset (50 queries)
- [x] Phase 1: Baseline evaluation
- [ ] Phase 2: A/B testing framework
- [ ] Phase 2: Statistical significance tests
- [ ] Phase 3: Model comparison matrix
- [ ] Phase 3: Prompt engineering experiments

## Reference

See `docs/RAG_EVALUATION_FRAMEWORK_SPEC.md` for full specification and design decisions.
