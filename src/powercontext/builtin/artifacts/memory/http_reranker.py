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
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    _validate_bounds,
)
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    """Rerank Memory candidates via an HTTP scoring service with batched requests.

    Splits candidates into batches of at most batch_size, sends each batch to the
    endpoint for scoring, validates responses, then globally sorts all candidates
    by score descending (ties broken by original rank ascending) and returns the
    top limit ranks.
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
            raise ValueError("batch_size must be at least 1")
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
        """Score candidates via HTTP batches and return the globally top-ranked selection."""

        _validate_bounds(len(candidates), limit)

        all_scores: list[tuple[int, float]] = []
        requests_made = 0

        for batch_start in range(0, len(candidates), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]
            batch_ranks = list(range(batch_start + 1, batch_end + 1))

            request_body = {
                "query": query,
                "candidates": [
                    {"rank": rank, "text": candidates[rank - 1].text}
                    for rank in batch_ranks
                ],
            }

            response = await self._client.post(self._endpoint, json=request_body)
            requests_made += 1

            response_data = _parse_response(response)
            _validate_scores(response_data, batch_ranks)

            for score_entry in response_data["scores"]:
                all_scores.append((score_entry["rank"], score_entry["score"]))

        all_scores.sort(key=lambda x: (-x[1], x[0]))
        selected_ranks = tuple(rank for rank, _ in all_scores[:limit])

        return MemoryRerankDecision(
            selected_ranks=selected_ranks,
            usage=InferenceUsage(requests=requests_made),
        )


def _parse_response(response: httpx.Response) -> dict[str, Any]:
    """Parse and validate the top-level response schema."""

    try:
        data = response.json()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid JSON response: {exc}") from exc

    if not isinstance(data, dict) or set(data.keys()) != {"scores"}:
        raise ValueError("response must have exactly one key: 'scores'")

    if not isinstance(data["scores"], list):
        raise ValueError("'scores' must be a list")

    return data


def _validate_scores(response_data: dict[str, Any], expected_ranks: list[int]) -> None:
    """Validate each score entry and ensure exact rank coverage for the batch."""

    scores = response_data["scores"]
    seen_ranks: set[int] = set()

    for entry in scores:
        if not isinstance(entry, dict) or set(entry.keys()) != {"rank", "score"}:
            raise ValueError("each score entry must have exactly keys: 'rank' and 'score'")

        rank = entry["rank"]
        score = entry["score"]

        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(f"rank must be an integer, got {type(rank).__name__}")

        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"score must be a number, got {type(score).__name__}")

        if not math.isfinite(score):
            raise ValueError(f"score must be finite, got {score}")

        if rank in seen_ranks:
            raise ValueError(f"duplicate rank {rank} in response")
        seen_ranks.add(rank)

    if seen_ranks != set(expected_ranks):
        missing = set(expected_ranks) - seen_ranks
        extra = seen_ranks - set(expected_ranks)
        parts = []
        if missing:
            parts.append(f"missing ranks: {sorted(missing)}")
        if extra:
            parts.append(f"extra ranks: {sorted(extra)}")
        raise ValueError(f"rank coverage mismatch: {', '.join(parts)}")


__all__ = ["HTTPMemoryReranker"]
