"""Loading of the corpus and the labeled eval set.

The bundled data lives under ``data/`` at the repo root. Both loaders default to
that location so the tool runs fully offline out of the box, but any path may be
supplied to point at your own eval set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Repo root is two levels up from this file (rageval/datasets.py -> repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "corpus"
DEFAULT_EVALSET_PATH = _REPO_ROOT / "data" / "evalset.json"


@dataclass(frozen=True)
class Document:
    """A single corpus document identified by its filename stem."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class EvalExample:
    """One labeled query: relevant doc ids plus a reference answer."""

    question_id: str
    question: str
    relevant_ids: list[str]
    reference_answer: str


def load_corpus(corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> list[Document]:
    """Load every ``*.md`` file in ``corpus_dir`` as a :class:`Document`.

    The document id is the filename without extension (e.g. ``vpc.md`` -> ``vpc``).
    Results are sorted by id for deterministic ordering.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    documents = [
        Document(doc_id=path.stem, text=path.read_text(encoding="utf-8"))
        for path in sorted(corpus_path.glob("*.md"))
    ]
    if not documents:
        raise ValueError(f"No .md documents found in corpus directory: {corpus_path}")
    return documents


def load_evalset(evalset_path: str | Path = DEFAULT_EVALSET_PATH) -> list[EvalExample]:
    """Load the labeled eval set from a JSON file.

    Expected shape: a JSON array of objects with ``id``, ``question``,
    ``relevant_ids`` and ``reference_answer`` fields.
    """
    path = Path(evalset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Eval set not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalExample(
            question_id=item["id"],
            question=item["question"],
            relevant_ids=list(item["relevant_ids"]),
            reference_answer=item["reference_answer"],
        )
        for item in raw
    ]
