"""Pure retrieval metrics.

Every function here is deterministic and side-effect free: it takes a ranked
list of retrieved document ids plus the set of relevant ids for one query and
returns a float. Aggregation across a query set lives in :mod:`rageval.harness`.

Relevance is treated as **binary** (a document is either relevant or not),
which is the standard setup for a labeled RAG eval set.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _top_k(retrieved: Sequence[str], k: int) -> list[str]:
    """Return the first ``k`` retrieved ids, guarding against a non-positive k."""
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    return list(retrieved[:k])


def hit_rate(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if at least one relevant doc appears in the top-k, else 0.0.

    Also known as success@k. Answers "did retrieval surface *anything*
    useful in the window the answerer actually sees?".
    """
    relevant_set = set(relevant)
    return 1.0 if any(doc in relevant_set for doc in _top_k(retrieved, k)) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant docs that appear in the top-k.

    Returns 0.0 when there are no relevant docs (nothing to recall).
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    found = sum(1 for doc in set(_top_k(retrieved, k)) if doc in relevant_set)
    return found / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k slots occupied by a (distinct) relevant doc.

    Precision@k divides the number of distinct relevant docs found in the top-k
    by the window size ``k`` -- it answers "how much of what the answerer sees
    is actually useful?". Duplicate ids in ``retrieved`` are collapsed, and the
    denominator is always ``k`` (a short retrieval that leaves slots empty is
    penalized, which is the standard convention). Returns 0.0 when there are no
    relevant docs.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    found = sum(1 for doc in set(_top_k(retrieved, k)) if doc in relevant_set)
    return found / k


def f1_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Harmonic mean of :func:`precision_at_k` and :func:`recall_at_k`.

    Balances "is the top-k mostly relevant?" (precision) against "did we find
    most of the relevant docs?" (recall). Returns 0.0 when both are 0 (i.e. no
    relevant doc was retrieved), avoiding a divide-by-zero.
    """
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mrr(retrieved: Sequence[str], relevant: Iterable[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant doc (1/rank), else 0.0.

    Rewards ranking a relevant doc high. If ``k`` is given, only the top-k are
    considered; otherwise the full ranking is scanned.
    """
    relevant_set = set(relevant)
    ranked = retrieved if k is None else _top_k(retrieved, k)
    for index, doc in enumerate(ranked):
        if doc in relevant_set:
            return 1.0 / (index + 1)
    return 0.0


def dcg(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Discounted cumulative gain over the top-k with binary gains.

    gain_i = 1 for a relevant doc, discounted by log2(rank + 1).

    A relevant doc that appears more than once in the ranking is credited only
    at its best (first) rank -- like :func:`precision_at_k` and
    :func:`recall_at_k`, duplicate retrieved ids are collapsed. Counting a
    repeat again would double its gain and let :func:`ndcg` exceed 1.0.
    """
    relevant_set = set(relevant)
    seen: set[str] = set()
    total = 0.0
    for index, doc in enumerate(_top_k(retrieved, k)):
        if doc in relevant_set and doc not in seen:
            seen.add(doc)
            total += 1.0 / math.log2(index + 2)
    return total


def ndcg(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized DCG@k with binary relevance in [0, 1].

    Divides the achieved DCG by the ideal DCG (all relevant docs ranked first).
    Returns 0.0 when there are no relevant docs.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ideal_hits = min(len(relevant_set), k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    if ideal == 0.0:
        return 0.0
    return dcg(retrieved, relevant_set, k) / ideal
