# Ground Truth Dataset

This directory contains labeled evaluation queries for the RAG system.

## Files

- `queries.jsonl` - Main evaluation dataset (50 queries)
  - 20 golden queries (manually curated, high-value)
  - 25 synthetic queries (LLM-generated from corpus)
  - 5 rejection queries (no answer in corpus)

## Schema

Each line is a JSON object with:

```json
{
  "query_id": "Q001",
  "query": "How do I calculate implied volatility using Black-Scholes?",
  "category": "options_derivatives",
  "difficulty": "medium",
  "is_rejection": false,
  "gold_references": [
    {"document": "Black Scholes with Python (Van Der Post 2024).pdf", "page": 45, "relevance": 3}
  ],
  "expected_answer_gist": "IV calculation via Newton-Raphson or bisection",
  "notes": ""
}
```

## Categories

- `options_derivatives` - Black-Scholes, Greeks, volatility, pricing
- `portfolio_management` - Optimization, risk, allocation
- `algorithmic_trading` - Strategies, backtesting, execution
- `machine_learning` - ML in finance, time series ML
- `risk_management` - VaR, CVaR, stress testing
- `time_series` - Forecasting, ARIMA, neural networks
- `data_engineering` - Python, data cleaning
- `rejection` - Queries with NO answer in corpus

## Relevance Scale

- 3: Directly answers with specific, actionable information
- 2: Contains useful information but incomplete
- 1: Same topic but doesn't answer the query
- 0: Irrelevant (implicit for non-gold)

## Generating New Queries

Ground truth was created by:
1. Reading actual PDF content (not via RAG to avoid bias)
2. Crafting queries that span the corpus
3. Manually identifying which documents/pages answer each query
4. Validating with ensemble AI consultation

Do NOT use RAG search results to create ground truth - this "poisons the well".
