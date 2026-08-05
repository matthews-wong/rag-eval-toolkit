"""Answer-quality heuristics.

These are deliberately lightweight, dependency-free approximations meant for an
offline portfolio demo — NOT a substitute for a semantic or LLM-judge eval:

* ``overlap_f1`` — a ROUGE-lite unigram F1 between the generated answer and a
  reference answer. It measures lexical overlap only; paraphrases score low
  even when correct.
* ``grounding_score`` — checks that every document the answer *cites* was
  actually retrieved for that query. It verifies citation provenance, not
  factual correctness.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

# Matches inline citations such as [doc3] or [corpus/policy] inside an answer.
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokenization used by the overlap heuristic."""
    return _TOKEN_PATTERN.findall(text.lower())


def overlap_f1(candidate: str, reference: str) -> float:
    """ROUGE-lite unigram F1 between a candidate and reference answer.

    Uses multiset (count-aware) overlap so repeated words are not double
    counted. Returns 0.0 when either side has no tokens.
    """
    candidate_tokens = Counter(tokenize(candidate))
    reference_tokens = Counter(tokenize(reference))
    if not candidate_tokens or not reference_tokens:
        return 0.0

    overlap = sum((candidate_tokens & reference_tokens).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(candidate_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 2 * precision * recall / (precision + recall)


def extract_citations(answer: str) -> list[str]:
    """Return the doc ids cited inline in an answer, in order of appearance.

    Citations are written as ``[doc_id]``. Whitespace around the id is trimmed.
    """
    return [match.strip() for match in _CITATION_PATTERN.findall(answer)]


def grounding_score(answer: str, retrieved: Iterable[str]) -> float:
    """Fraction of cited docs that were actually retrieved, in [0, 1].

    1.0 means every citation is grounded in the retrieved context. An answer
    with no citations returns 0.0 — an uncited answer is treated as ungrounded,
    which is the conservative choice for a grounding check.
    """
    citations = extract_citations(answer)
    if not citations:
        return 0.0
    retrieved_set = set(retrieved)
    grounded = sum(1 for cite in citations if cite in retrieved_set)
    return grounded / len(citations)


def ungrounded_citations(answer: str, retrieved: Iterable[str]) -> list[str]:
    """List cited doc ids that were NOT retrieved (i.e. unsupported claims)."""
    retrieved_set = set(retrieved)
    return [cite for cite in extract_citations(answer) if cite not in retrieved_set]
