#!/usr/bin/env python3
"""CLI entry point for RAG evaluation framework.

Usage:
    python -m eval.run --config baseline
    python -m eval.run --config baseline --verbose
    python -m eval.run --config baseline --output reports/baseline_2025-01-01.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import yaml
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .eval_runner import EvalRunner, EvalConfig

app = typer.Typer(help="RAG Evaluation Framework for rust-local-rag")
console = Console()


def find_config(config_name: str) -> Path:
    """Find config file by name or path."""
    # Try as direct path first
    if Path(config_name).exists():
        return Path(config_name)

    # Try in configs directory
    config_dir = Path(__file__).parent / "configs"
    candidates = [
        config_dir / config_name,
        config_dir / f"{config_name}.yaml",
        config_dir / f"{config_name}.yml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Config not found: {config_name}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def print_results_table(results: dict) -> None:
    """Print results as a rich table."""
    metrics = results.get("metrics", {})

    # Summary table
    table = Table(title=f"Evaluation Results: {results.get('config', 'unknown')}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Hit Rate@5", f"{metrics.get('hit_rate_mean', 0):.3f}")
    table.add_row("MRR@5", f"{metrics.get('mrr_mean', 0):.3f}")
    table.add_row("NDCG@5", f"{metrics.get('ndcg_mean', 0):.3f}")
    table.add_row("Latency p50", f"{metrics.get('latency_p50_ms', 0):.0f}ms")
    table.add_row("Latency p95", f"{metrics.get('latency_p95_ms', 0):.0f}ms")
    table.add_row("Queries", str(metrics.get("n_queries", 0)))

    if "rejection_accuracy" in metrics:
        table.add_row("Rejection Accuracy", f"{metrics.get('rejection_accuracy', 0):.3f}")

    console.print(table)

    # Per-category breakdown
    per_category = results.get("per_category_metrics", {})
    if per_category:
        cat_table = Table(title="Per-Category Breakdown")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("N", style="dim")
        cat_table.add_column("Hit Rate", style="green")
        cat_table.add_column("MRR", style="green")

        for cat, cat_metrics in sorted(per_category.items()):
            cat_table.add_row(
                cat,
                str(cat_metrics.get("n_queries", 0)),
                f"{cat_metrics.get('hit_rate_mean', 0):.3f}",
                f"{cat_metrics.get('mrr_mean', 0):.3f}",
            )

        console.print(cat_table)


def generate_markdown_report(results: dict, output_path: Path) -> None:
    """Generate markdown report."""
    metrics = results.get("metrics", {})
    md_path = output_path.with_suffix(".md")

    lines = [
        f"# Evaluation Report: {results.get('config', 'unknown')}",
        "",
        f"**Timestamp**: {results.get('timestamp', 'unknown')}",
        f"**Total Time**: {results.get('total_time_seconds', 0):.1f}s",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Hit Rate@5 | {metrics.get('hit_rate_mean', 0):.3f} |",
        f"| MRR@5 | {metrics.get('mrr_mean', 0):.3f} |",
        f"| NDCG@5 | {metrics.get('ndcg_mean', 0):.3f} |",
        f"| Latency p50 | {metrics.get('latency_p50_ms', 0):.0f}ms |",
        f"| Latency p95 | {metrics.get('latency_p95_ms', 0):.0f}ms |",
        f"| Queries | {metrics.get('n_queries', 0)} |",
        "",
    ]

    # Per-category
    per_category = results.get("per_category_metrics", {})
    if per_category:
        lines.extend([
            "## Per-Category Breakdown",
            "",
            "| Category | N | Hit Rate | MRR |",
            "|----------|---|----------|-----|",
        ])
        for cat, cat_metrics in sorted(per_category.items()):
            lines.append(
                f"| {cat} | {cat_metrics.get('n_queries', 0)} | "
                f"{cat_metrics.get('hit_rate_mean', 0):.3f} | "
                f"{cat_metrics.get('mrr_mean', 0):.3f} |"
            )
        lines.append("")

    # Failed queries
    per_query = results.get("per_query_results", [])
    misses = [q for q in per_query if q.get("hit_rate", 1) == 0 and not q.get("is_rejection")]
    if misses:
        lines.extend([
            "## Missed Queries (Hit Rate = 0)",
            "",
        ])
        for q in misses[:10]:  # Limit to 10
            lines.append(f"- **{q.get('query_id')}**: {q.get('query', '')[:80]}...")
        if len(misses) > 10:
            lines.append(f"- ... and {len(misses) - 10} more")
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"[dim]Markdown report: {md_path}[/dim]")


@app.command()
def evaluate(
    config: str = typer.Option("baseline", "--config", "-c", help="Config name or path"),
    output: str = typer.Option(None, "--output", "-o", help="Output JSON path"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Generate markdown report"),
):
    """Run evaluation with specified configuration."""
    try:
        # Find and load config
        config_path = find_config(config)
        console.print(f"[cyan]Loading config:[/cyan] {config_path}")
        eval_config = EvalConfig.from_yaml(str(config_path))

        # Create runner
        runner = EvalRunner(eval_config)
        console.print(f"[cyan]Loaded {len(runner.ground_truth)} ground truth queries[/cyan]")

        # Run evaluation
        console.print(Panel(f"Running evaluation: {eval_config.name}", style="bold blue"))
        results = runner.run_evaluation(verbose=verbose)

        # Print results
        print_results_table(results)

        # Save output
        if output:
            output_path = Path(output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"eval/reports/{eval_config.name}_{timestamp}.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved:[/green] {output_path}")

        # Generate markdown
        if markdown:
            generate_markdown_report(results, output_path)

        # Success threshold check
        hit_rate = results.get("metrics", {}).get("hit_rate_mean", 0)
        if hit_rate >= 0.80:
            console.print(Panel("[green]SUCCESS: Hit Rate >= 0.80 threshold[/green]", style="green"))
        else:
            console.print(Panel(f"[yellow]Below threshold: Hit Rate {hit_rate:.3f} < 0.80[/yellow]", style="yellow"))

    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ConnectionError as e:
        console.print(f"[red]Connection Error:[/red] {e}")
        console.print("[dim]Make sure the RAG server is running.[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def check(
    config: str = typer.Option("baseline", "--config", "-c", help="Config name or path"),
):
    """Check RAG server health and ground truth file."""
    from .rag_client import RAGClient

    # Load config to get correct endpoint
    try:
        config_path = find_config(config)
        eval_config = EvalConfig.from_yaml(str(config_path))
        console.print(f"[cyan]Using config:[/cyan] {config_path}")
    except FileNotFoundError:
        console.print(f"[yellow]Config '{config}' not found, using defaults[/yellow]")
        eval_config = None

    # Create client with config endpoint
    if eval_config:
        client = RAGClient(
            endpoint=eval_config.rag_endpoint,
            mode=eval_config.connection_mode,
        )
        console.print(f"[cyan]Checking RAG server at:[/cyan] {eval_config.rag_endpoint} ({eval_config.connection_mode} mode)")
    else:
        client = RAGClient()
        console.print("[cyan]Checking RAG server at:[/cyan] default endpoint")

    if client.health_check():
        console.print("[green]RAG server is healthy[/green]")
        try:
            stats = client.get_stats()
            console.print(f"  Documents: {stats.get('documents', 'unknown')}")
            console.print(f"  Chunks: {stats.get('chunks', 'unknown')}")
        except Exception as e:
            console.print(f"[yellow]Could not get stats: {e}[/yellow]")
    else:
        console.print("[red]RAG server not reachable[/red]")

    # Check ground truth
    gt_path = Path("eval/ground_truth/queries.jsonl")
    if gt_path.exists():
        with open(gt_path) as f:
            n_queries = sum(1 for line in f if line.strip() and not line.startswith("#"))
        console.print(f"[green]Ground truth:[/green] {n_queries} queries")
    else:
        console.print(f"[yellow]Ground truth not found:[/yellow] {gt_path}")


@app.command()
def list_configs():
    """List available configurations."""
    config_dir = Path(__file__).parent / "configs"
    if not config_dir.exists():
        console.print("[yellow]No configs directory found[/yellow]")
        return

    configs = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    if not configs:
        console.print("[yellow]No config files found[/yellow]")
        return

    table = Table(title="Available Configurations")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for config_path in sorted(configs):
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            name = config_path.stem
            desc = data.get("description", "")[:60]
            table.add_row(name, desc)
        except Exception:
            table.add_row(config_path.stem, "[error loading]")

    console.print(table)


if __name__ == "__main__":
    app()
