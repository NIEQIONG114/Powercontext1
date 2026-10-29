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

"""HTTP-based reranking service adapter for Memory search."""

from __future__ import annotations

import math
from typing import Any, NoReturn

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    _validate_bounds,
)
from powercontext.builtin.inference import InferenceUsage


class _ContractViolation(ValueError):
    """A caller or a service value outside the Memory reranking contract."""


def _reject(message: str) -> NoReturn:
    raise _ContractViolation(message)


class HTTPMemoryReranker:
    """Rerank Memory candidates through a batched HTTP scoring service.

    Candidates travel in batches of at most ``batch_size``, and every batch keeps the
    caller's one-based global ranks. Because the ranks are global, the per-batch scores
    merge back into a single ordering over the whole pool: score descending, ties broken
    by the original rank ascending.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if batch_size < 1:
            _reject("batch_size must be at least 1")
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Score every candidate exactly once, then select the globally best ``limit`` ranks."""

        _validate_bounds(len(candidates), limit)

        scored: list[tuple[int, float]] = []
        requests_made = 0
        for start in range(0, len(candidates), self._batch_size):
            ranks = range(start + 1, min(start + self._batch_size, len(candidates)) + 1)
            scored.extend(await self._score_batch(query, candidates, ranks))
            requests_made += 1

        scored.sort(key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _ in scored[:limit]),
            usage=InferenceUsage(requests=requests_made),
        )

    async def _score_batch(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        ranks: range,
    ) -> list[tuple[int, float]]:
        """Send one batch and return its ``(global rank, score)`` pairs."""

        request_body = {
            "query": query,
            "candidates": [{"rank": rank, "text": candidates[rank - 1].text} for rank in ranks],
        }
        response = await self._client.post(self._endpoint, json=request_body)
        # A rejected status must surface as the httpx status error even when the body
        # happens to parse: an error page is never a set of candidate scores.
        response.raise_for_status()
        return _read_scores(response, expected_ranks=set(ranks))


def _read_scores(response: httpx.Response, *, expected_ranks: set[int]) -> list[tuple[int, float]]:
    """Parse one batch response and enforce exact coverage of that batch's ranks."""

    try:
        payload = response.json()
    except (ValueError, TypeError) as error:
        message = f"reranker response is not JSON: {error}"
        raise _ContractViolation(message) from error

    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        _reject("reranker response must be an object with exactly one key: 'scores'")
    entries = payload["scores"]
    if not isinstance(entries, list):
        _reject("reranker response 'scores' must be a list")

    observed: dict[int, float] = {}
    for entry in entries:
        rank, score = _read_score_entry(entry)
        if rank in observed:
            _reject(f"reranker response repeats rank {rank}")
        observed[rank] = score

    if set(observed) != expected_ranks:
        _reject(f"reranker response covered ranks {sorted(observed)} instead of {sorted(expected_ranks)}")
    return list(observed.items())


def _read_score_entry(entry: Any) -> tuple[int, float]:
    """Validate one ``{"rank", "score"}`` record."""

    if not isinstance(entry, dict) or set(entry) != {"rank", "score"}:
        _reject("each reranker score must be an object with exactly 'rank' and 'score'")
    rank = entry["rank"]
    score = entry["score"]
    if isinstance(rank, bool) or not isinstance(rank, int):
        _reject(f"reranker score rank must be an integer, got {type(rank).__name__}")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        _reject(f"reranker score value must be a number, got {type(score).__name__}")
    if not math.isfinite(score):
        _reject(f"reranker score value must be finite, got {score!r}")
    return rank, float(score)


__all__ = ["HTTPMemoryReranker"]
