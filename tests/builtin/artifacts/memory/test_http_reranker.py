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

"""Behavior tests for the HTTP Memory reranking adapter."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import HTTPMemoryReranker, MemoryHit

_ENDPOINT = "https://reranker.invalid/score"


def _hit(rank: int) -> MemoryHit:
    return MemoryHit(
        memory_ref=ArtifactRef(family="memory", artifact_id="memory", revision=1),
        entry_id=f"entry-{rank}",
        entry_version_id=f"entry-{rank}-v1",
        text=f"Candidate {rank}",
        score=1 / rank,
        matched_by=("fts",),
    )


def _batch_scores(request: httpx.Request, scores: dict[int, float]) -> httpx.Response:
    body = json.loads(request.content)
    assert body["query"] == "query"
    return httpx.Response(
        200,
        json={"scores": [{"rank": item["rank"], "score": scores[item["rank"]]} for item in body["candidates"]]},
    )


def test_batches_keep_global_ranks_and_order_the_whole_pool() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return _batch_scores(request, {1: 0.2, 2: 0.9, 3: 0.1, 4: 0.9, 5: 0.95})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(_ENDPOINT, client=client, batch_size=2)
            decision = await reranker.rerank("query", tuple(_hit(rank) for rank in range(1, 6)), 3)

        assert decision.selected_ranks == (5, 2, 4)
        assert decision.usage.requests == 3
        sent = [item for body in bodies for item in body["candidates"]]
        assert [item["rank"] for item in sent] == [1, 2, 3, 4, 5]
        assert [item["text"] for item in sent] == [_hit(rank).text for rank in range(1, 6)]
        assert all(len(body["candidates"]) <= 2 for body in bodies)

    asyncio.run(scenario())


def test_rejected_status_is_not_read_as_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"scores": [{"rank": 1, "score": 0.9}]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(_ENDPOINT, client=client)
            with pytest.raises(httpx.HTTPStatusError):
                await reranker.rerank("query", (_hit(1),), 1)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "scores",
    [
        [{"rank": 3, "score": 0.5}, {"rank": 2, "score": 0.4}],
        [{"rank": 2, "score": 0.5}],
        [{"rank": 2, "score": 0.5}, {"rank": 2, "score": 0.4}],
    ],
)
def test_each_batch_must_cover_its_own_ranks_once(scores: list[dict[str, object]]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"scores": scores})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reranker = HTTPMemoryReranker(_ENDPOINT, client=client, batch_size=2)
            with pytest.raises(ValueError):
                await reranker.rerank("query", tuple(_hit(rank) for rank in range(1, 4)), 1)

    asyncio.run(scenario())


def test_exposes_policy_identity_and_rejects_out_of_range_arguments() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: _batch_scores(request, {1: 0.5}))
        ) as client:
            assert HTTPMemoryReranker(_ENDPOINT, client=client).policy_id == "http-cross-encoder-v1"
            reranker = HTTPMemoryReranker(_ENDPOINT, client=client, policy_id="custom-http-v2")
            assert reranker.policy_id == "custom-http-v2"

            with pytest.raises(ValueError):
                await reranker.rerank("query", (), 1)
            with pytest.raises(ValueError):
                await reranker.rerank("query", (_hit(1),), 2)
            with pytest.raises(ValueError):
                HTTPMemoryReranker(_ENDPOINT, client=client, batch_size=0)

    asyncio.run(scenario())
