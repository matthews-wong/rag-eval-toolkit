"""Unit tests for the pure retrieval metrics with hand-computed expected values."""

import math

import pytest

from rageval import metrics


# --- hit_rate --------------------------------------------------------------

def test_hit_rate_hit_within_k():
    assert metrics.hit_rate(["a", "b", "c"], {"c"}, k=3) == 1.0


def test_hit_rate_miss_when_relevant_below_cutoff():
    # 'c' is relevant but sits at rank 3, outside the top-2 window.
    assert metrics.hit_rate(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_hit_rate_no_relevant():
    assert metrics.hit_rate(["a", "b"], set(), k=2) == 0.0


# --- recall_at_k -----------------------------------------------------------

def test_recall_partial():
    # 2 of 3 relevant docs land in the top-3.
    assert metrics.recall_at_k(["a", "b", "x"], {"a", "b", "c"}, k=3) == pytest.approx(2 / 3)


def test_recall_full():
    assert metrics.recall_at_k(["a", "b"], {"a", "b"}, k=3) == 1.0


def test_recall_no_relevant_is_zero():
    assert metrics.recall_at_k(["a"], set(), k=1) == 0.0


# --- mrr -------------------------------------------------------------------

def test_mrr_first_relevant_at_rank_two():
    assert metrics.mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_first_relevant_at_rank_one():
    assert metrics.mrr(["a", "b"], {"a", "b"}) == 1.0


def test_mrr_no_relevant_found():
    assert metrics.mrr(["x", "y"], {"a"}) == 0.0


def test_mrr_respects_k_cutoff():
    # Relevant doc is at rank 3; with k=2 it is out of scope -> 0.
    assert metrics.mrr(["x", "y", "a"], {"a"}, k=2) == 0.0


# --- ndcg ------------------------------------------------------------------

def test_ndcg_perfect_ranking_is_one():
    assert metrics.ndcg(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_ndcg_known_value_single_relevant_at_rank_two():
    # One relevant doc at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    expected = (1 / math.log2(3)) / 1.0
    assert metrics.ndcg(["x", "a", "y"], {"a"}, k=3) == pytest.approx(expected)


def test_ndcg_known_value_two_relevant_ranks_one_and_three():
    # Relevant at ranks 1 and 3.
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # = 1 + 0.5
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal: ranks 1 and 2
    assert metrics.ndcg(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(dcg / idcg)


def test_ndcg_no_relevant_is_zero():
    assert metrics.ndcg(["a", "b"], set(), k=2) == 0.0


# --- guards ----------------------------------------------------------------

def test_non_positive_k_raises():
    with pytest.raises(ValueError):
        metrics.hit_rate(["a"], {"a"}, k=0)
