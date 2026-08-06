"""Evaluation harness.

Runs a retriever and an answerer over the labeled eval set and collects both
retrieval metrics and answer-quality heuristics into a structured result that
:mod:`rageval.report` renders.

The harness depends only on the :class:`~rageval.retriever.Retriever` and
:class:`Answerer` protocols, so any pipeline can be plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Protocol

from . import answer_metrics, metrics
from .datasets import Document, EvalExample
from .retriever import Retriever


class Answerer(Protocol):
    """Generates an answer from a query and the retrieved document ids.

    Answers may cite docs inline as ``[doc_id]`` so grounding can be checked.
    """

    def answer(self, query: str, retrieved_ids: list[str]) -> str:
        ...


class ExtractiveAnswerer:
    """Deterministic demo answerer.

    Returns the opening line of the top retrieved document and cites it. This is
    a stand-in for a real generator; it exists so the harness has something to
    score end-to-end offline. Because it only ever cites retrieved docs, it is
    grounded by construction — the grounding *check* itself is exercised with
    adversarial cases in the test suite.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._text_by_id = {doc.doc_id: doc.text for doc in documents}

    def answer(self, query: str, retrieved_ids: list[str]) -> str:
        if not retrieved_ids:
            return "No documents were retrieved for this query."
        top_id = retrieved_ids[0]
        snippet = self._first_sentence(self._text_by_id.get(top_id, ""))
        return f"{snippet} [{top_id}]"

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Best-effort first sentence: skip markdown headings, split on '. '."""
        for line in text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped.split(". ")[0].rstrip(".") + "."
        return ""


@dataclass(frozen=True)
class QueryResult:
    """Per-query scores produced by the harness."""

    question_id: str
    question: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    answer: str
    hit_rate: float
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    mrr: float
    ndcg: float
    answer_f1: float
    grounding: float
    ungrounded_citations: list[str]


@dataclass(frozen=True)
class EvalReport:
    """Aggregate report over the whole eval set."""

    k: int
    per_query: list[QueryResult]
    aggregate: dict[str, float] = field(default_factory=dict)


def evaluate(
    retriever: Retriever,
    answerer: Answerer,
    evalset: list[EvalExample],
    k: int = 3,
) -> EvalReport:
    """Score ``retriever`` + ``answerer`` over ``evalset`` at cutoff ``k``.

    Returns an :class:`EvalReport` with per-query rows and macro-averaged
    aggregates (each query weighted equally).
    """
    if not evalset:
        raise ValueError("Eval set is empty; nothing to evaluate")

    per_query: list[QueryResult] = []
    for example in evalset:
        retrieved = retriever.retrieve(example.question, k)
        answer = answerer.answer(example.question, retrieved)
        per_query.append(
            QueryResult(
                question_id=example.question_id,
                question=example.question,
                retrieved_ids=retrieved,
                relevant_ids=example.relevant_ids,
                answer=answer,
                hit_rate=metrics.hit_rate(retrieved, example.relevant_ids, k),
                precision_at_k=metrics.precision_at_k(retrieved, example.relevant_ids, k),
                recall_at_k=metrics.recall_at_k(retrieved, example.relevant_ids, k),
                f1_at_k=metrics.f1_at_k(retrieved, example.relevant_ids, k),
                mrr=metrics.mrr(retrieved, example.relevant_ids, k),
                ndcg=metrics.ndcg(retrieved, example.relevant_ids, k),
                answer_f1=answer_metrics.overlap_f1(answer, example.reference_answer),
                grounding=answer_metrics.grounding_score(answer, retrieved),
                ungrounded_citations=answer_metrics.ungrounded_citations(answer, retrieved),
            )
        )

    aggregate = {
        "hit_rate": mean(r.hit_rate for r in per_query),
        "precision_at_k": mean(r.precision_at_k for r in per_query),
        "recall_at_k": mean(r.recall_at_k for r in per_query),
        "f1_at_k": mean(r.f1_at_k for r in per_query),
        "mrr": mean(r.mrr for r in per_query),
        "ndcg": mean(r.ndcg for r in per_query),
        "answer_f1": mean(r.answer_f1 for r in per_query),
        "grounding": mean(r.grounding for r in per_query),
    }
    return EvalReport(k=k, per_query=per_query, aggregate=aggregate)
