"""Console entry point: ``rag-eval``.

Wires the bundled corpus + eval set through the TF-IDF retriever and the
extractive demo answerer, then renders a scorecard. Everything runs offline.
"""

from __future__ import annotations

import random

import click
import numpy as np

from . import datasets
from .harness import ExtractiveAnswerer, evaluate
from .report import render_markdown, render_rich
from .retriever import TfidfRetriever

# Fixed seed so any incidental randomness stays reproducible across runs.
_SEED = 42


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("-k", "--top-k", "k", default=3, show_default=True, help="Retrieval cutoff k.")
@click.option(
    "--corpus",
    "corpus_dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Corpus directory of .md docs (defaults to bundled data/corpus).",
)
@click.option(
    "--evalset",
    "evalset_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Eval set JSON file (defaults to bundled data/evalset.json).",
)
@click.option(
    "--markdown",
    "as_markdown",
    is_flag=True,
    help="Emit a Markdown scorecard instead of the rich terminal view.",
)
def main(k: int, corpus_dir: str | None, evalset_path: str | None, as_markdown: bool) -> None:
    """Evaluate a RAG pipeline over a labeled eval set and print a scorecard."""
    random.seed(_SEED)
    np.random.seed(_SEED)

    documents = datasets.load_corpus(corpus_dir or datasets.DEFAULT_CORPUS_DIR)
    evalset = datasets.load_evalset(evalset_path or datasets.DEFAULT_EVALSET_PATH)

    retriever = TfidfRetriever(documents)
    answerer = ExtractiveAnswerer(documents)
    report = evaluate(retriever, answerer, evalset, k=k)

    if as_markdown:
        click.echo(render_markdown(report))
    else:
        render_rich(report)


if __name__ == "__main__":
    main()
