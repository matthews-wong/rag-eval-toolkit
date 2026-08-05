"""End-to-end tests: the harness runs offline over the bundled sample data."""

import pytest

from rageval import datasets
from rageval.harness import EvalReport, ExtractiveAnswerer, evaluate
from rageval.retriever import TfidfRetriever


@pytest.fixture(scope="module")
def bundled():
    documents = datasets.load_corpus()
    evalset = datasets.load_evalset()
    return documents, evalset


def test_bundled_data_loads(bundled):
    documents, evalset = bundled
    assert len(documents) >= 5
    assert len(evalset) >= 5
    # Every relevant id in the eval set must exist in the corpus.
    corpus_ids = {doc.doc_id for doc in documents}
    for example in evalset:
        assert set(example.relevant_ids) <= corpus_ids


def test_harness_runs_end_to_end(bundled):
    documents, evalset = bundled
    retriever = TfidfRetriever(documents)
    answerer = ExtractiveAnswerer(documents)

    report = evaluate(retriever, answerer, evalset, k=3)

    assert isinstance(report, EvalReport)
    assert len(report.per_query) == len(evalset)
    # Aggregates are valid probabilities/scores in [0, 1].
    for value in report.aggregate.values():
        assert 0.0 <= value <= 1.0


def test_harness_is_deterministic(bundled):
    documents, evalset = bundled
    retriever = TfidfRetriever(documents)
    answerer = ExtractiveAnswerer(documents)

    first = evaluate(retriever, answerer, evalset, k=3).aggregate
    second = evaluate(retriever, answerer, evalset, k=3).aggregate
    assert first == second


def test_tfidf_retriever_finds_obvious_match(bundled):
    documents, _ = bundled
    retriever = TfidfRetriever(documents)
    # A query lexically dominated by S3 terms should retrieve the s3 doc first.
    top = retriever.retrieve("object storage bucket durability", k=1)
    assert top == ["s3"]


def test_demo_answers_are_grounded(bundled):
    documents, evalset = bundled
    retriever = TfidfRetriever(documents)
    answerer = ExtractiveAnswerer(documents)
    report = evaluate(retriever, answerer, evalset, k=3)
    # The extractive answerer only ever cites the retrieved top doc.
    assert report.aggregate["grounding"] == pytest.approx(1.0)
