"""Unit tests for the answer-quality heuristics."""

import pytest

from rageval import answer_metrics


# --- overlap_f1 ------------------------------------------------------------

def test_overlap_f1_identical_is_one():
    assert answer_metrics.overlap_f1("the cat sat", "the cat sat") == pytest.approx(1.0)


def test_overlap_f1_disjoint_is_zero():
    assert answer_metrics.overlap_f1("alpha beta", "gamma delta") == 0.0


def test_overlap_f1_partial_known_value():
    # candidate {the, cat}, reference {the, dog}: overlap=1.
    # precision = 1/2, recall = 1/2 -> F1 = 0.5.
    assert answer_metrics.overlap_f1("the cat", "the dog") == pytest.approx(0.5)


def test_overlap_f1_empty_is_zero():
    assert answer_metrics.overlap_f1("", "something") == 0.0


# --- citations & grounding -------------------------------------------------

def test_extract_citations_in_order():
    assert answer_metrics.extract_citations("See [vpc] and also [s3].") == ["vpc", "s3"]


def test_grounding_all_cited_docs_retrieved():
    answer = "The answer is grounded [vpc] [s3]."
    assert answer_metrics.grounding_score(answer, ["vpc", "s3", "iam"]) == 1.0


def test_grounding_flags_ungrounded_citation():
    # 'ghost' was cited but never retrieved -> grounding drops to 0.5.
    answer = "Mixed [vpc] and [ghost]."
    retrieved = ["vpc", "s3"]
    assert answer_metrics.grounding_score(answer, retrieved) == pytest.approx(0.5)
    assert answer_metrics.ungrounded_citations(answer, retrieved) == ["ghost"]


def test_grounding_no_citations_is_zero():
    assert answer_metrics.grounding_score("no citations here", ["vpc"]) == 0.0


def test_no_ungrounded_when_all_grounded():
    assert answer_metrics.ungrounded_citations("[vpc] [s3]", ["vpc", "s3"]) == []
