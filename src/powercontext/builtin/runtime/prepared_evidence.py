# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Query-driven selection of literal evidence spans from recalled Memory text.

The selector never rewrites, translates, or summarizes: every region it proposes
is a half-open character range of the original entry text, so the caller can
verify ``text[start:end]`` against the cited entry.

Relevance combines rare single terms with adjacent term pairs. A claim such as
"Activated Cdk5 regulates DNA damage-induced neuronal death" shares many terms
with several sentences of an abstract, but only the evidence sentence carries
the adjacent pairs that mark the claim itself. Term rarity is measured over the
sentences of the candidate documents, because a term that recurs throughout a
document says little about which sentence answers the query.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")
_SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?:\s+|$)|\n+")
_STEM_LENGTH = 5
_MIN_PREFIX_MATCH = 4

_STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "must",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "shall",
        "should",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
    }
)


@dataclass(frozen=True, slots=True)
class EvidenceRegion:
    """One literal span of an original entry text and its query coverage."""

    start: int
    end: int
    first: int
    last: int
    terms: int
    phrases: int
    score: float


def _stem(token: str) -> str:
    return token[:_STEM_LENGTH]


def _matches(left: str, right: str) -> bool:
    """Match two stems, allowing a shared prefix of at least four characters.

    Comparing four characters rather than the whole stem keeps related forms
    together: "fibrotic" answers to "fibrosis" and "activation" to "activates".
    """

    if left == right:
        return True
    if len(left) < _MIN_PREFIX_MATCH or len(right) < _MIN_PREFIX_MATCH:
        return False
    if left[:_MIN_PREFIX_MATCH] == right[:_MIN_PREFIX_MATCH]:
        return True
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    return longer.startswith(shorter)


def _token_stems(text: str) -> tuple[str, ...]:
    return tuple(_stem(token) for token in _TOKEN_PATTERN.findall(text.lower()))


def query_terms(query: str) -> tuple[str, ...]:
    """Return the deduplicated query terms used to rank evidence spans."""

    terms: list[str] = []
    for token in _TOKEN_PATTERN.findall(query.lower()):
        if token in _STOP_TERMS:
            continue
        stem = _stem(token)
        if not any(_matches(stem, existing) for existing in terms):
            terms.append(stem)
    return tuple(terms)


def query_phrases(query: str) -> tuple[str, ...]:
    """Return the deduplicated adjacent term pairs of the query."""

    stems = _token_stems(query)
    phrases: list[str] = []
    for index in range(len(stems) - 1):
        left, right = stems[index], stems[index + 1]
        if left in _STOP_TERMS or right in _STOP_TERMS:
            continue
        phrase = f"{left} {right}"
        if phrase not in phrases:
            phrases.append(phrase)
    return tuple(phrases)


def document_stems(text: str) -> tuple[str, ...]:
    """Return the stemmed tokens of one document."""

    return _token_stems(text)


def sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Return the trimmed character spans of the sentences found in ``text``."""

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return tuple(spans)


def covered_terms(
    sentence: str,
    terms: Sequence[str],
    phrases: Sequence[str] = (),
) -> tuple[frozenset[str], frozenset[str]]:
    """Return the query terms and adjacent term pairs that appear in one sentence."""

    stems = _token_stems(sentence)
    if not stems:
        return frozenset(), frozenset()
    present_terms = {term for term in terms if any(_matches(term, stem) for stem in stems)}
    present_phrases = {phrase for phrase in phrases if _phrase_present(phrase, stems)}
    return frozenset(present_terms), frozenset(present_phrases)


def _phrase_present(phrase: str, stems: Sequence[str]) -> bool:
    left, _, right = phrase.partition(" ")
    for index in range(len(stems) - 1):
        if _matches(left, stems[index]) and _matches(right, stems[index + 1]):
            return True
    return False


def term_weights(
    terms: Sequence[str],
    passages: Sequence[Sequence[str]],
    phrases: Sequence[str] = (),
) -> dict[str, float]:
    """Weight each query term by how rare it is across the candidate passages.

    A passage is one sentence of a candidate document. Weighting at that
    granularity separates a term that recurs throughout a document from one that
    marks the single sentence that answers the query.
    """

    total = max(1, len(passages))
    weights: dict[str, float] = {}
    for term in terms:
        occurrences = sum(1 for stems in passages if any(_matches(term, stem) for stem in stems))
        weights[term] = math.log(1.0 + total / (1.0 + occurrences))
    for phrase in phrases:
        occurrences = sum(1 for stems in passages if _phrase_present(phrase, stems))
        weights[phrase] = math.log(1.0 + total / (1.0 + occurrences))
    return weights


def relevance(
    terms: frozenset[str],
    phrases: frozenset[str],
    weights: Mapping[str, float],
    *,
    phrase_bonus: float,
) -> float:
    """Score one passage by the rarity of the query material it carries."""

    return sum(weights.get(term, 0.0) for term in terms) + phrase_bonus * sum(
        weights.get(phrase, 0.0) for phrase in phrases
    )


__all__ = [
    "EvidenceRegion",
    "covered_terms",
    "document_stems",
    "query_phrases",
    "query_terms",
    "relevance",
    "sentence_spans",
    "term_weights",
]
