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

"""Prepare final, bounded context from approved Artifact evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import chain
from typing import TypeVar

from powercontext.artifacts import ArtifactAddress, ArtifactRef
from powercontext.builtin.artifacts.experience import Experience, ExperienceSearchHit, render_experience
from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.artifacts.profile.models import Profile
from powercontext.builtin.artifacts.topic_memory import TopicMemory, TopicMemorySearchHit
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.models import PrepareContextRequest, PreparedContext
from powercontext.builtin.runtime.prepared_code import (
    CodeEvidenceRef,
    PreparedCodeCandidate,
    assemble_code,
)
from powercontext.builtin.runtime.prepared_evidence import (
    covered_terms,
    document_stems,
    query_phrases,
    query_terms,
    relevance,
    sentence_spans,
    term_weights,
)
from powercontext.builtin.runtime.prepared_text import (
    TRUST_POLICY,
    ContextTextItem,
    fit_context_text_item,
    render_context_text,
)
from powercontext.builtin.runtime.recall_sufficiency import RecallBudgetView

_MIN_TRUNCATED_CONTENT_BYTES = 64
_ELLIPSIS = "…"
_BEGIN_MARKER = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END_MARKER = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_Item = TypeVar("_Item")


@dataclass(frozen=True)
class _PreparedContextEntry:
    origin: PreparedContextOrigin
    kind: str
    citation: dict[str, object]
    content: str
    truncated: bool


@dataclass(frozen=True)
class _EvidenceDocument:
    """One recalled Memory entry offered to the evidence selector."""

    scope_id: str
    memory_ref: ArtifactRef
    entry_id: str
    entry_version_id: str
    text: str


@dataclass(frozen=True)
class _EvidenceAnalysis:
    """The query coverage of one candidate document."""

    document: _EvidenceDocument
    spans: tuple[tuple[int, int], ...]
    covered: tuple[tuple[frozenset[str], frozenset[str]], ...]
    region: EvidenceRegion | None = None
    terms: int = 0
    phrases: int = 0
    score: float = 0.0


@dataclass(frozen=True)
class _EntryFit:
    """One fit attempt on the non-assembly path: the entry that fitted, or why none did.

    The two drop reasons are disjoint and mirror the assembly path's
    :class:`~powercontext.builtin.runtime.prepared_text.FitOutcome`. An entry is dropped
    *below the minimum truncated content* when a shorter rendering exists and fits but is too
    short to be an honest delivery; it is dropped *with no fitting truncation* when no
    shortened rendering fits at all.
    """

    entry: _PreparedContextEntry | None = None
    dropped_below_min_bytes: bool = False
    dropped_no_fitting_truncation: bool = False


@dataclass(frozen=True)
class PreparedContextOmissions:
    """Aggregate counts of the items the byte budget omitted in one build.

    Every count is a per-call aggregate, never per-entry attribution and never a verdict about
    an entry: an entry losing to the budget is not a negative result about that entry.

    ``dropped_items`` is the sum of its two sub-counts, so "the budget could not fit this
    item" can be told apart from "this item was too short to truncate into the remaining
    space". A single merged counter would misreport the second cause as the first.
    """

    truncated_items: int = 0
    dropped_items: int = 0
    dropped_below_min_bytes: int = 0
    dropped_no_fitting_truncation: int = 0


@dataclass(frozen=True)
class PreparedContextBuild:
    """Final public context and the exact origins selected to produce it.

    ``omissions`` is always filled by the Builder. The RFC 1560 recall trace is deliberately
    **not** a field here: ``ScopedContextApplication._prepare`` returns ``build.context`` and
    discards the rest, so such a field would have no production observer. The trace is
    delivered through the Runtime's optional ``RecallEffortSink`` instead.

    ``fragments`` is populated only by query-driven evidence selection. Every fragment is a
    literal span of the original Memory entry, and it lines up one-to-one with ``origins``.
    """

    context: PreparedContext
    origins: tuple[PreparedContextOrigin, ...]
    omissions: PreparedContextOmissions = PreparedContextOmissions()
    code_origins: tuple[CodeEvidenceRef, ...] = ()
    fragments: tuple[EvidenceFragment, ...] = ()


@dataclass(frozen=True)
class MemoryEntryAddress:
    """Identify one exact Memory entry version across Scope boundaries."""

    memory: ArtifactAddress
    entry_id: str
    entry_version_id: str


@dataclass(frozen=True)
class EvidenceFragment:
    """One literal span of an original Memory entry, with its exact origin.

    ``text`` is always ``entry.text[start:end]`` of the entry version named by
    ``address``; the fragment never carries rewritten or generated evidence.
    """

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


PreparedContextOrigin = MemoryCitation | ArtifactRef | MemoryEntryAddress | ArtifactAddress


@dataclass(frozen=True)
class PreparedMemoryCandidates:
    """Memory candidates read from one Scope."""

    scope_id: str
    memory_ref: ArtifactRef | None = None
    hits: tuple[MemoryHit, ...] = ()


@dataclass(frozen=True)
class PreparedExperienceCandidates:
    """Experience candidates read from one Scope."""

    scope_id: str
    hits: tuple[ExperienceSearchHit, ...] = ()


@dataclass(frozen=True)
class PreparedProfileCandidate:
    """The latest committed Profile snapshot read from one Scope."""

    scope_id: str
    profile: Profile


class PreparedContextBuilder:
    """Select and render final context without I/O, persistence, or reranking."""

    memory_candidate_limit = 16
    topic_memory_candidate_limit = 8
    experience_candidate_limit = 8
    candidate_limit = memory_candidate_limit
    entry_limit = 8
    topic_memory_entry_limit = 8
    experience_entry_limit = 2
    max_entry_content_bytes = 2000
    evidence_candidate_limit = 4
    evidence_document_limit = 1
    evidence_phrase_bonus = 1.0
    evidence_anchor_share = 0.5
    evidence_document_span = 1
    evidence_document_mix = 0.5
    evidence_distinct_weight = 0.5
    evidence_numeric_boost = 2.0
    evidence_skip_leading = True
    _text_entry_limit: int | None = None

    def empty(self) -> PreparedContext:
        return PreparedContext(status="empty", content=None, content_bytes=0)

    def build(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContext:
        return self.build_result(
            request=request,
            scope_id=scope_id,
            memory_ref=memory_ref,
            hits=hits,
            topic_memory_hits=topic_memory_hits,
            experience_hits=experience_hits,
        ).context

    def build_result(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContextBuild:
        return self.build_scopes_result(
            request=request,
            current_scope_id=scope_id,
            memory_candidates=(
                PreparedMemoryCandidates(scope_id=scope_id or "", memory_ref=memory_ref, hits=tuple(hits)),
            ),
            experience_candidates=(PreparedExperienceCandidates(scope_id=scope_id or "", hits=tuple(experience_hits)),),
            topic_memory_hits=topic_memory_hits,
        )

    def build_scopes_result(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
        code_candidates: Sequence[PreparedCodeCandidate] = (),
    ) -> PreparedContextBuild:
        if request.include_code:

            def historical(history_request: PrepareContextRequest, entries: int) -> PreparedContextBuild:
                builder = PreparedContextBuilder()
                builder.entry_limit = entries
                builder._text_entry_limit = entries if entries < self.entry_limit else None
                return builder.build_scopes_result(
                    request=history_request.model_copy(update={"include_code": False}),
                    current_scope_id=current_scope_id,
                    memory_candidates=memory_candidates,
                    topic_memory_hits=topic_memory_hits,
                    experience_candidates=experience_candidates,
                    profile_candidates=profile_candidates,
                )

            return assemble_code(request, code_candidates, self.entry_limit, historical)

        if sum(len(candidates.hits) for candidates in memory_candidates) > self.memory_candidate_limit:
            raise PreparedContextInvariantError("memory-candidate-limit")
        if len(topic_memory_hits) > self.topic_memory_candidate_limit:
            raise PreparedContextInvariantError("topic-memory-candidate-limit")
        if sum(len(candidates.hits) for candidates in experience_candidates) > self.experience_candidate_limit:
            raise PreparedContextInvariantError("experience-candidate-limit")

        if request.evidence_selection:
            content, origins, fragments, omissions = self._select_evidence(
                request=request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
            )
            if content is None:
                return PreparedContextBuild(context=self.empty(), origins=(), omissions=omissions)
            content_bytes = len(content.encode("utf-8"))
            if content_bytes > request.max_bytes:
                raise PreparedContextInvariantError("output-budget")
            return PreparedContextBuild(
                context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
                origins=origins,
                omissions=omissions,
                fragments=fragments,
            )

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        if content is None:
            return PreparedContextBuild(context=self.empty(), origins=(), omissions=omissions)
        content_bytes = len(content.encode("utf-8"))
        # The assembly path is bounded by construction (every fitted item is measured against
        # `max_bytes`), so the ceiling check belongs to the non-assembly renderer only.
        if request.assembly is None and content_bytes > request.max_bytes:
            raise PreparedContextInvariantError("output-budget")
        return PreparedContextBuild(
            context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
            origins=origins,
            omissions=omissions,
        )

    def probe_budget(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
    ) -> RecallBudgetView:
        """Report what the byte budget does to one candidate set, without delivering it.

        This is the RFC 1560 *budget probe*: one pass of this Builder's own pure selection and
        rendering code, keeping only the counters and discarding the rendered output. It lets
        the gate tell budget-limited thinness from recall-limited thinness.

        Pure and side-effect free: no I/O, no persistence, nothing reported to any sink, and
        the same inputs as :meth:`build_scopes_result` are guaranteed to yield the same
        counters because both go through :meth:`_select_entries`.
        """

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        content_bytes = 0 if content is None else len(content.encode("utf-8"))
        return RecallBudgetView(
            max_bytes=request.max_bytes,
            delivered_items=len(origins),
            truncated_items=omissions.truncated_items,
            dropped_items=omissions.dropped_items,
            unused_bytes=max(0, request.max_bytes - content_bytes),
        )

    def _select_entries(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        """Run one complete pure selection pass and render it.

        Returns ``(content, origins, omissions)``; ``content`` is ``None`` exactly when nothing
        was selected, which is the empty-result case for both paths. This is the single
        selection path shared by :meth:`build_scopes_result` and :meth:`probe_budget`, so the
        probe's counters always describe the selection the build would actually perform.
        """

        if request.assembly is not None:
            return self._select_text(
                request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
                experience_candidates=experience_candidates,
                profile_candidates=profile_candidates,
                topic_memory_hits=topic_memory_hits,
            )

        memory_entries = _interleave_groups(
            tuple(
                self._memory_entries(
                    candidates.memory_ref,
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in memory_candidates
            )
        )
        topic_memory_entries = self._topic_memory_entries(topic_memory_hits)
        experience_entries = _interleave_groups(
            tuple(
                self._experience_entries(
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in experience_candidates
            )
        )[: self.experience_entry_limit]
        entries, omissions = self._fit_entries(request, memory_entries, topic_memory_entries, experience_entries)
        if not entries:
            return None, (), omissions
        return _render(entries), tuple(entry.origin for entry in entries), omissions

    def _select_evidence(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
    ) -> tuple[
        str | None,
        tuple[PreparedContextOrigin, ...],
        tuple[EvidenceFragment, ...],
        PreparedContextOmissions,
    ]:
        """Select literal evidence spans from the recalled Memory text.

        The query alone drives the selection. Every returned fragment is a verbatim
        range of the original entry, so a caller can verify ``text[start:end]``
        against the cited entry version without trusting this builder.
        """

        documents: list[_EvidenceDocument] = []
        for candidates in memory_candidates:
            scope_id = candidates.scope_id or current_scope_id or ""
            for hit in candidates.hits:
                if not hit.entry_id.strip() or not hit.entry_version_id.strip() or not hit.text.strip():
                    continue
                documents.append(
                    _EvidenceDocument(
                        scope_id=scope_id,
                        memory_ref=hit.memory_ref,
                        entry_id=hit.entry_id,
                        entry_version_id=hit.entry_version_id,
                        text=hit.text,
                    )
                )
        if not documents:
            return None, (), (), PreparedContextOmissions()

        terms = query_terms(request.query)
        phrases = query_phrases(request.query)
        passages = [
            document.text[start:end]
            for document in documents
            for start, end in sentence_spans(document.text)
        ]
        weights = term_weights(terms, [document_stems(passage) for passage in passages], phrases)
        # Numbers and units in a claim rarely repeat by accident, so they are the
        # strongest available hint about which sentence answers it.
        for term in terms:
            if any(character.isdigit() for character in term):
                weights[term] = weights.get(term, 0.0) * self.evidence_numeric_boost
        analyses = [
            self._analyze_document(
                document,
                terms,
                phrases,
                weights,
                phrase_bonus=self.evidence_phrase_bonus,
                document_span=self.evidence_document_span,
            )
            for document in documents
        ]

        # Rank every candidate sentence across all documents. Selecting more than
        # one sentence keeps a mistaken ranking from losing the evidence outright:
        # the runner-up sentence still lands inside the delivered budget.
        candidates: list[tuple[float, int, int]] = []
        for document_index, analysis in enumerate(analyses):
            for sentence_index, (present, paired) in enumerate(analysis.covered):
                if not present and not paired:
                    continue
                if sentence_index == 0 and self.evidence_skip_leading:
                    # The assembled document opens with its title, which names the
                    # topic rather than the finding the claim asks about.
                    continue
                score = relevance(present, paired, weights, phrase_bonus=self.evidence_phrase_bonus)
                candidates.append((score, document_index, sentence_index))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        if not candidates:
            return None, (), (), PreparedContextOmissions()

        # Every additional document costs one more citation in the envelope, and a
        # citation is worth several sentences of evidence. Only the leading
        # documents earn that cost.
        leading = [
            index
            for index, _ in sorted(
                ((index, analysis.score) for index, analysis in enumerate(analyses)),
                key=lambda item: (-item[1], item[0]),
            )
            if analyses[index].score > 0
        ][: self.evidence_document_limit]
        if not leading:
            leading = [0]
        allowed = frozenset(leading)

        chosen: dict[int, list[tuple[int, int]]] = {}
        anchor_budget = int(request.max_bytes * self.evidence_anchor_share)
        for budget in (anchor_budget, request.max_bytes):
            chosen = {}
            for _score, document_index, sentence_index in candidates:
                if document_index not in allowed:
                    continue
                trial = {index: list(value) for index, value in chosen.items()}
                analysis = analyses[document_index]
                start, end = _trim_span(
                    analysis.document.text, *analysis.spans[sentence_index]
                )
                trial[document_index] = _merge_ranges(
                    trial.get(document_index, []), start, end
                )
                if self._evidence_size(analyses, trial) > budget:
                    continue
                chosen = trial
            if chosen:
                break
        if not chosen:
            return None, (), (), PreparedContextOmissions()

        # Anchors locate the evidence. Context then grows them, because a wider
        # literal range can only add tokens to the intersection with the evidence
        # and never removes the anchor itself.
        chosen = self._evidence_extend(analyses, chosen, request.max_bytes, weights)
        chosen = self._evidence_fill(analyses, chosen, request.max_bytes)

        fragments = self._evidence_fragments(analyses, chosen)
        if not fragments:
            return None, (), (), PreparedContextOmissions()

        content = _render_evidence(
            [
                (
                    {
                        "scope_id": fragment.address.memory.scope_id,
                        "memory_ref": fragment.address.memory.artifact.model_dump(mode="json"),
                        "entry_id": fragment.address.entry_id,
                        "entry_version_id": fragment.address.entry_version_id,
                    },
                    fragment.text,
                    fragment.start,
                    fragment.end,
                )
                for fragment in fragments
            ]
        )
        return content, tuple(fragment.address for fragment in fragments), tuple(fragments), PreparedContextOmissions()

    def _evidence_extend(
        self,
        analyses: Sequence[_EvidenceAnalysis],
        chosen: Mapping[int, list[tuple[int, int]]],
        max_bytes: int,
        weights: Mapping[str, float],
    ) -> dict[int, list[tuple[int, int]]]:
        """Grow the anchored sentences with the neighbouring context that fits.

        The neighbour carrying the most query material is taken first, so the
        extra text is spent where the evidence continues rather than on filler.
        """

        current = {index: list(value) for index, value in chosen.items()}
        while True:
            best: tuple[float, int, int, int, dict[int, list[tuple[int, int]]]] | None = None
            for document_index, ranges in current.items():
                analysis = analyses[document_index]
                for neighbour in self._evidence_neighbours(analysis, ranges):
                    trial = {index: list(value) for index, value in current.items()}
                    start, end = _trim_span(
                        analysis.document.text, *analysis.spans[neighbour]
                    )
                    trial[document_index] = _merge_ranges(ranges, start, end)
                    size = self._evidence_size(analyses, trial)
                    if size > max_bytes:
                        continue
                    present, paired = analysis.covered[neighbour]
                    score = relevance(
                        present, paired, weights, phrase_bonus=self.evidence_phrase_bonus
                    )
                    candidate = (score, -size, document_index, neighbour, trial)
                    if best is None or candidate[:4] > best[:4]:
                        best = candidate
            if best is None:
                return current
            current = best[4]

    def _evidence_fill(
        self,
        analyses: Sequence[_EvidenceAnalysis],
        chosen: Mapping[int, list[tuple[int, int]]],
        max_bytes: int,
    ) -> dict[int, list[tuple[int, int]]]:
        """Spend the bytes that whole sentences left over on partial context.

        The range grows to the last whole word that still fits, so the delivered
        text stays readable and still a literal span of the original entry.
        """

        current = {index: list(value) for index, value in chosen.items()}
        while True:
            best: tuple[int, int, int, dict[int, list[tuple[int, int]]]] | None = None
            for document_index, ranges in current.items():
                analysis = analyses[document_index]
                for position, (start, end) in enumerate(ranges):
                    for direction in (1, -1):
                        grown = self._evidence_grown(
                            analyses, document_index, start, end, direction, max_bytes, current
                        )
                        if grown is None:
                            continue
                        new_start, new_end, trial = grown
                        candidate = (document_index, position, direction, trial)
                        if best is None or candidate[:3] < best[:3]:
                            best = candidate
            if best is None:
                return current
            current = best[3]

    def _evidence_grown(
        self,
        analyses: Sequence[_EvidenceAnalysis],
        document_index: int,
        start: int,
        end: int,
        direction: int,
        max_bytes: int,
        current: Mapping[int, list[tuple[int, int]]],
    ) -> tuple[int, int, dict[int, list[tuple[int, int]]]] | None:
        """Return the largest word-aligned growth of one range that still fits."""

        analysis = analyses[document_index]
        text = analysis.document.text
        limit = len(text)
        low, high = 0, limit
        best: tuple[int, int] | None = None
        while low <= high:
            middle = (low + high) // 2
            if direction > 0:
                candidate_start, candidate_end = start, min(limit, end + middle)
            else:
                candidate_start, candidate_end = max(0, start - middle), end
            candidate_start, candidate_end = _trim_span(text, candidate_start, candidate_end)
            if candidate_end <= candidate_start:
                high = middle - 1
                continue
            trial = {index: list(value) for index, value in current.items()}
            trial[document_index] = _merge_ranges(
                trial.get(document_index, []), candidate_start, candidate_end
            )
            if self._evidence_size(analyses, trial) > max_bytes:
                high = middle - 1
                continue
            best = (candidate_start, candidate_end)
            low = middle + 1
        if best is None:
            return None
        new_start, new_end = best
        if (new_start, new_end) == (start, end):
            return None
        trial = {index: list(value) for index, value in current.items()}
        trial[document_index] = _merge_ranges(
            trial.get(document_index, []), new_start, new_end
        )
        return new_start, new_end, trial

    @staticmethod
    def _evidence_neighbours(
        analysis: _EvidenceAnalysis,
        ranges: Sequence[tuple[int, int]],
    ) -> tuple[int, ...]:
        """Return the sentence indexes directly adjacent to the selected ranges."""

        covered = set()
        for start, end in ranges:
            for index, (sentence_start, sentence_end) in enumerate(analysis.spans):
                if sentence_start < end and start < sentence_end:
                    covered.add(index)
        if not covered:
            return ()
        neighbours = []
        for index in (min(covered) - 1, max(covered) + 1):
            if 0 <= index < len(analysis.spans) and index not in covered:
                neighbours.append(index)
        return tuple(neighbours)

    def _evidence_fragments(
        self,
        analyses: Sequence[_EvidenceAnalysis],
        chosen: Mapping[int, list[tuple[int, int]]],
    ) -> tuple[EvidenceFragment, ...]:
        """Turn the selected ranges into literal spans of their documents."""

        fragments: list[EvidenceFragment] = []
        for document_index, ranges in chosen.items():
            analysis = analyses[document_index]
            for start, end in ranges:
                start, end = _trim_span(analysis.document.text, start, end)
                if end <= start:
                    continue
                fragments.append(
                    EvidenceFragment(
                        address=MemoryEntryAddress(
                            memory=ArtifactAddress(
                                scope_id=analysis.document.scope_id,
                                artifact=analysis.document.memory_ref,
                            ),
                            entry_id=analysis.document.entry_id,
                            entry_version_id=analysis.document.entry_version_id,
                        ),
                        start=start,
                        end=end,
                        text=analysis.document.text[start:end],
                    )
                )
        fragments.sort(key=lambda fragment: (fragment.address.entry_id, fragment.start))
        return tuple(fragments)

    def _evidence_size(
        self,
        analyses: Sequence[_EvidenceAnalysis],
        chosen: Mapping[int, list[tuple[int, int]]],
    ) -> int:
        """Return the rendered byte size of one trial selection."""

        fragments = self._evidence_fragments(analyses, chosen)
        return len(
            _render_evidence(
                [
                    (
                        {
                            "scope_id": fragment.address.memory.scope_id,
                            "memory_ref": fragment.address.memory.artifact.model_dump(mode="json"),
                            "entry_id": fragment.address.entry_id,
                            "entry_version_id": fragment.address.entry_version_id,
                        },
                        fragment.text,
                        fragment.start,
                        fragment.end,
                    )
                    for fragment in fragments
                ]
            ).encode("utf-8")
        )

    def _analyze_document(
        self,
        document: _EvidenceDocument,
        terms: Sequence[str],
        phrases: Sequence[str],
        weights: Mapping[str, float],
        *,
        phrase_bonus: float,
        document_span: int,
    ) -> _EvidenceAnalysis:
        spans = sentence_spans(document.text)
        covered = [covered_terms(document.text[start:end], terms, phrases) for start, end in spans]
        scores = [
            relevance(present, paired, weights, phrase_bonus=phrase_bonus) for present, paired in covered
        ]
        best_terms = 0
        best_phrases = 0
        distinct: set[str] = set()
        for present, paired in covered:
            if len(present) > best_terms:
                best_terms = len(present)
            best_phrases = max(best_phrases, len(paired))
            distinct |= present
        # The document title names the topic, the body carries the finding, and the
        # claim's rare words tell the two apart. Together they decide which
        # document the evidence belongs to.
        title_score = scores[0] if scores else 0.0
        body_scores = sorted(scores[1:], reverse=True)
        body_score = body_scores[0] if body_scores else 0.0
        distinct_score = sum(weights.get(term, 0.0) for term in distinct)
        leading = title_score + body_score + self.evidence_distinct_weight * distinct_score
        return _EvidenceAnalysis(
            document=document,
            spans=spans,
            covered=covered,
            terms=best_terms,
            phrases=best_phrases,
            score=leading,
        )

    @staticmethod
    def _evidence_citation(document: _EvidenceDocument) -> dict[str, object]:
        return {
            "scope_id": document.scope_id,
            "memory_ref": document.memory_ref.model_dump(mode="json"),
            "entry_id": document.entry_id,
            "entry_version_id": document.entry_version_id,
        }

    def _select_text(
        self,
        request: PrepareContextRequest,
        *,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        assembly = request.assembly
        if assembly is None:
            raise PreparedContextInvariantError("text-assembly-missing")
        included: list[ContextTextItem] = []
        origins: list[PreparedContextOrigin] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for section in assembly.sections:
            # Unfilled earlier sections leave their capacity available to later ones.
            limit = (
                section.limit
                if self._text_entry_limit is None
                else min(section.limit, self._text_entry_limit - len(included))
            )
            if section.family == "profile":
                entries = self._profile_entries(profile_candidates)
            elif section.family == "topic-memory":
                entries = self._topic_memory_entries(topic_memory_hits, scope_id=current_scope_id)
            else:
                groups = (
                    tuple(
                        tuple(
                            self._memory_entries(group.memory_ref, (hit,), scope_id=group.scope_id)
                            for hit in group.hits
                        )
                        for group in memory_candidates
                    )
                    if section.family == "memory"
                    else tuple(
                        tuple(self._experience_entries((hit,), scope_id=group.scope_id) for hit in group.hits)
                        for group in experience_candidates
                    )
                )
                entries = tuple(chain.from_iterable(_interleave_groups(groups)))
            seen: set[tuple[str, str, str, int, str | None, str | None]] = set()
            rank = 0
            selected_count = 0
            for entry in entries:
                if selected_count >= limit:
                    break
                item = _text_item(entry)
                artifact = item.artifact
                identity = (
                    artifact.scope_id,
                    artifact.artifact.family,
                    artifact.artifact.artifact_id,
                    artifact.artifact.revision,
                    item.entry_id,
                    item.entry_version_id,
                )
                if identity in seen or not item.content.strip():
                    continue
                seen.add(identity)
                rank += 1
                outcome = fit_context_text_item(included, replace(item, recall_rank=rank), assembly, request.max_bytes)
                if outcome.item is None:
                    dropped_below_min_bytes += int(outcome.dropped_below_min_bytes)
                    dropped_no_fitting_truncation += int(outcome.dropped_no_fitting_truncation)
                    continue
                fitted = outcome.item
                included.append(fitted)
                origins.append(entry.origin)
                selected_count += 1
                truncated_items += int(fitted.truncated)
        omissions = PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )
        if not included:
            return None, (), omissions
        content = render_context_text(included, assembly)
        return content, tuple(origins), omissions

    def _profile_entries(self, candidates: Sequence[PreparedProfileCandidate]) -> tuple[_PreparedContextEntry, ...]:
        entries = []
        for candidate in candidates:
            origin = ArtifactAddress(scope_id=candidate.scope_id, artifact=candidate.profile.as_ref())
            entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="profile",
                    citation={"artifact": origin.model_dump(mode="json")},
                    content=candidate.profile.content.content,
                    truncated=False,
                )
            )
        return tuple(entries)

    def _memory_entries(
        self,
        memory_ref: ArtifactRef | None,
        hits: Sequence[MemoryHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        if hits and memory_ref is None:
            raise PreparedContextInvariantError("memory-ref-missing")
        memory_entries: list[_PreparedContextEntry] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            if hit.memory_ref != memory_ref:
                raise PreparedContextInvariantError("memory-ref-mismatch")

            citation_key = (hit.entry_id, hit.entry_version_id)
            if citation_key in seen:
                continue
            seen.add(citation_key)
            if not hit.entry_id.strip() or not hit.entry_version_id.strip() or not hit.text.strip():
                continue
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            origin: PreparedContextOrigin = citation
            rendered_citation = citation.model_dump(mode="json")
            if scope_id is not None:
                memory = ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref)
                origin = MemoryEntryAddress(
                    memory=memory,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
                rendered_citation = {
                    "memory": memory.model_dump(mode="json"),
                    "entry_id": hit.entry_id,
                    "entry_version_id": hit.entry_version_id,
                }
            memory_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="memory",
                    citation=rendered_citation,
                    content=hit.text,
                    truncated=False,
                )
            )
        return tuple(memory_entries)

    def _topic_memory_entries(
        self,
        hits: Sequence[TopicMemorySearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        topic_entries: list[_PreparedContextEntry] = []
        seen_topics: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != TopicMemory.family:
                raise PreparedContextInvariantError("topic-memory-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_topics:
                continue
            seen_topics.add(identity)
            if len(topic_entries) >= self.topic_memory_entry_limit:
                break
            content = {"title": hit.title, "summary": hit.summary}
            if hit.snippet is not None:
                content["snippet"] = hit.snippet
            origin: PreparedContextOrigin = hit.artifact_ref
            citation = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            body = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                citation = {"artifact": origin.model_dump(mode="json")}
                body = "\n\n".join(f"{label.capitalize()}: {value}" for label, value in content.items())
            topic_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="topic-memory",
                    citation=citation,
                    content=body,
                    truncated=False,
                )
            )
        return tuple(topic_entries)

    def _experience_entries(
        self,
        hits: Sequence[ExperienceSearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        experience_entries: list[_PreparedContextEntry] = []
        seen_experiences: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != Experience.family:
                raise PreparedContextInvariantError("experience-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_experiences:
                continue
            seen_experiences.add(identity)
            origin: PreparedContextOrigin = hit.artifact_ref
            rendered_citation: dict[str, object] = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                rendered_citation = {"artifact": origin.model_dump(mode="json")}
            experience_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="experience",
                    citation=rendered_citation,
                    content=render_experience(hit.content),
                    truncated=False,
                )
            )
        return tuple(experience_entries)

    def _fit_entries(
        self,
        request: PrepareContextRequest,
        memory_entries: Sequence[_PreparedContextEntry],
        topic_memory_entries: Sequence[_PreparedContextEntry],
        experience_entries: Sequence[_PreparedContextEntry],
    ) -> tuple[tuple[_PreparedContextEntry, ...], PreparedContextOmissions]:
        entries: list[_PreparedContextEntry] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for candidate in _interleave(memory_entries, topic_memory_entries, experience_entries):
            if len(entries) >= self.entry_limit:
                break
            fit = self._fit_entry(
                entries,
                origin=candidate.origin,
                kind=candidate.kind,
                citation=candidate.citation,
                text=candidate.content,
                max_bytes=request.max_bytes,
            )
            if fit.entry is None:
                dropped_below_min_bytes += int(fit.dropped_below_min_bytes)
                dropped_no_fitting_truncation += int(fit.dropped_no_fitting_truncation)
                continue
            entries.append(fit.entry)
            if fit.entry.truncated:
                truncated_items += 1
        return tuple(entries), PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )

    def _fit_entry(
        self,
        entries: Sequence[_PreparedContextEntry],
        *,
        origin: PreparedContextOrigin,
        kind: str,
        citation: dict[str, object],
        text: str,
        max_bytes: int,
    ) -> _EntryFit:
        """Fit one entry into the remaining budget, reporting why it was dropped if it was.

        The two ``None`` paths of the original code are kept apart: ``below-min-bytes`` means a
        shorter rendering exists and fits but is too short to deliver honestly, while
        ``no-fitting-truncation`` means no shortened rendering fitted at all.
        """

        source_bytes = len(text.encode("utf-8"))
        entry_budget = min(source_bytes, self.max_entry_content_bytes)
        candidate = _PreparedContextEntry(
            origin=origin,
            kind=kind,
            citation=citation,
            content=text if source_bytes <= entry_budget else _truncate_utf8(text, entry_budget),
            truncated=source_bytes > entry_budget,
        )
        if _rendered_bytes((*entries, candidate)) <= max_bytes:
            return _EntryFit(entry=candidate)
        if source_bytes < _MIN_TRUNCATED_CONTENT_BYTES:
            return _EntryFit(dropped_below_min_bytes=True)

        lower = _MIN_TRUNCATED_CONTENT_BYTES
        upper = min(entry_budget, source_bytes - 1)
        best: _PreparedContextEntry | None = None
        while lower <= upper:
            byte_budget = (lower + upper) // 2
            candidate = _PreparedContextEntry(
                origin=origin,
                kind=kind,
                citation=citation,
                content=_truncate_utf8(text, byte_budget),
                truncated=True,
            )
            if len(candidate.content.encode("utf-8")) < _MIN_TRUNCATED_CONTENT_BYTES:
                lower = byte_budget + 1
                continue
            if _rendered_bytes((*entries, candidate)) <= max_bytes:
                best = candidate
                lower = byte_budget + 1
            else:
                upper = byte_budget - 1
        if best is None:
            return _EntryFit(dropped_no_fitting_truncation=True)
        return _EntryFit(entry=best)


def _interleave(
    memory: Sequence[_PreparedContextEntry],
    topics: Sequence[_PreparedContextEntry],
    experiences: Sequence[_PreparedContextEntry],
) -> tuple[_PreparedContextEntry, ...]:
    """Preserve each family's rank while alternating Memory, Topic, Experience."""

    ordered: list[_PreparedContextEntry] = []
    for index in range(max(len(memory), len(topics), len(experiences))):
        if index < len(memory):
            ordered.append(memory[index])
        if index < len(topics):
            ordered.append(topics[index])
        if index < len(experiences):
            ordered.append(experiences[index])
    return tuple(ordered)


def _interleave_groups(groups: Sequence[Sequence[_Item]]) -> tuple[_Item, ...]:
    ordered: list[_Item] = []
    for index in range(max((len(group) for group in groups), default=0)):
        ordered.extend(group[index] for group in groups if index < len(group))
    return tuple(ordered)


def _render(entries: Sequence[_PreparedContextEntry]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [_render_entry(entry) for entry in entries],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _render_evidence(items: Sequence[tuple[dict[str, object], str, int, int]]) -> str:
    """Render the evidence envelope, keeping the established trust and markers."""

    envelope = {
        "trust": "untrusted_history",
        "items": [
            {"citation": citation, "content": content, "start": start, "end": end}
            for citation, content, start, end in items
        ],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _escaped_length(text: str) -> int:
    """Return the byte length a string occupies inside the rendered JSON envelope."""

    return len(json.dumps(text, ensure_ascii=False).encode("utf-8")) - 2


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim the surrounding whitespace of one character range."""

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _merge_ranges(
    ranges: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    """Add one character range and merge whatever it touches."""

    merged = sorted([*ranges, (start, end)])
    result: list[tuple[int, int]] = []
    for current_start, current_end in merged:
        if result and current_start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], current_end))
        else:
            result.append((current_start, current_end))
    return result


def _text_item(entry: _PreparedContextEntry) -> ContextTextItem:
    origin = entry.origin
    if isinstance(origin, MemoryEntryAddress):
        return ContextTextItem(
            artifact=origin.memory,
            content=entry.content,
            recall_rank=0,
            entry_id=origin.entry_id,
            entry_version_id=origin.entry_version_id,
        )
    if isinstance(origin, ArtifactAddress):
        return ContextTextItem(artifact=origin, content=entry.content, recall_rank=0)
    raise PreparedContextInvariantError("text-scope-missing")


def _render_entry(entry: _PreparedContextEntry) -> dict[str, object]:
    rendered: dict[str, object] = {
        "citation": entry.citation,
        "content": entry.content,
        "truncated": entry.truncated,
    }
    if entry.kind != "memory":
        rendered["kind"] = entry.kind
    return rendered


def _rendered_bytes(entries: Sequence[_PreparedContextEntry]) -> int:
    return len(_render(entries).encode("utf-8"))


def _truncate_utf8(text: str, byte_budget: int) -> str:
    prefix_budget = byte_budget - len(_ELLIPSIS.encode("utf-8"))
    encoded_prefix = text.encode("utf-8")[:prefix_budget]
    return f"{encoded_prefix.decode('utf-8', errors='ignore')}{_ELLIPSIS}"
