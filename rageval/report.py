"""Scorecard rendering — rich (terminal) and Markdown.

Both renderers consume an :class:`~rageval.harness.EvalReport`. The Markdown
output is what the README's Usage section embeds, so it is kept plain and
copy-paste friendly.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from .harness import EvalReport

# Human-readable labels for the aggregate metric keys, in display order.
_METRIC_LABELS: list[tuple[str, str]] = [
    ("hit_rate", "Hit-rate@k"),
    ("recall_at_k", "Recall@k"),
    ("mrr", "MRR"),
    ("ndcg", "nDCG@k"),
    ("answer_f1", "Answer F1 (ROUGE-lite)"),
    ("grounding", "Citation grounding"),
]


def render_rich(report: EvalReport, console: Console | None = None) -> None:
    """Print a colored scorecard to the terminal via rich."""
    console = console or Console()

    summary = Table(title=f"RAG Eval Scorecard (k={report.k})", title_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Score", justify="right", style="green")
    for key, label in _METRIC_LABELS:
        summary.add_row(label, f"{report.aggregate[key]:.3f}")
    console.print(summary)

    detail = Table(title="Per-query results", title_style="bold cyan")
    detail.add_column("Q", style="bold")
    detail.add_column("Hit", justify="right")
    detail.add_column("Recall", justify="right")
    detail.add_column("MRR", justify="right")
    detail.add_column("nDCG", justify="right")
    detail.add_column("F1", justify="right")
    detail.add_column("Ground", justify="right")
    for row in report.per_query:
        detail.add_row(
            row.question_id,
            f"{row.hit_rate:.0f}",
            f"{row.recall_at_k:.2f}",
            f"{row.mrr:.2f}",
            f"{row.ndcg:.2f}",
            f"{row.answer_f1:.2f}",
            f"{row.grounding:.2f}",
        )
    console.print(detail)


def render_markdown(report: EvalReport) -> str:
    """Return the scorecard as a Markdown string."""
    lines: list[str] = []
    lines.append(f"### RAG Eval Scorecard (k={report.k})")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("| --- | --- |")
    for key, label in _METRIC_LABELS:
        lines.append(f"| {label} | {report.aggregate[key]:.3f} |")
    lines.append("")
    lines.append("| Q | Hit | Recall | MRR | nDCG | F1 | Ground |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in report.per_query:
        lines.append(
            f"| {row.question_id} | {row.hit_rate:.0f} | {row.recall_at_k:.2f} | "
            f"{row.mrr:.2f} | {row.ndcg:.2f} | {row.answer_f1:.2f} | {row.grounding:.2f} |"
        )
    return "\n".join(lines)
