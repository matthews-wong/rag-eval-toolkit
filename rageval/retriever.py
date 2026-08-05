"""Retriever protocol and a TF-IDF demo implementation.

The :class:`Retriever` protocol is the seam the harness depends on — swap in any
object with a matching ``retrieve`` method (a vector store, a BM25 index, a
hosted search API) without touching the evaluation code.

:class:`TfidfRetriever` is the bundled offline default: a scikit-learn TF-IDF
vectorizer with cosine similarity over the corpus. It is deterministic (no
randomness) so scorecard numbers reproduce exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .datasets import Document


@runtime_checkable
class Retriever(Protocol):
    """A ranked document retriever.

    Implementations return the ids of the top-k documents for a query, most
    relevant first.
    """

    def retrieve(self, query: str, k: int) -> list[str]:
        ...


class TfidfRetriever:
    """Deterministic TF-IDF + cosine-similarity retriever over a fixed corpus."""

    def __init__(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("TfidfRetriever requires a non-empty corpus")
        self._doc_ids = [doc.doc_id for doc in documents]
        # English stop words + sublinear TF are sensible defaults for short docs.
        self._vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(doc.text for doc in documents)

    def retrieve(self, query: str, k: int) -> list[str]:
        """Return the top-k doc ids ranked by descending cosine similarity."""
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}")
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]
        # argsort is ascending; take the last k and reverse for descending order.
        # A stable sort on (-score, index) keeps ties in corpus order -> reproducible.
        order = np.lexsort((np.arange(len(scores)), -scores))
        top = order[:k]
        return [self._doc_ids[i] for i in top]
