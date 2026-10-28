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

"""Tests for the coalescing embedding wrapper."""

from __future__ import annotations

import asyncio

import pytest

from powercontext.builtin.inference import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.coalescing import CoalescingEmbeddingModel


class RecordingEmbedding:
    """Deterministic provider that records every underlying request."""

    def __init__(self, *, requests: int = 1) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._requests = requests

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        self.calls.append(tuple(texts))
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) for text in texts),
            usage=InferenceUsage(requests=self._requests, input_tokens=5, output_tokens=1),
        )


def test_disabled_cache_still_shares_running_requests() -> None:
    async def scenario() -> None:
        base = RecordingEmbedding()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=0)
        first, second = await asyncio.gather(model.embed(("alpha",)), model.embed(("alpha",)))
        assert first.vectors == second.vectors == ((5.0,),)
        assert base.calls == [("alpha",)]
        # The cross-call cache is off, so a later call asks the model again.
        await model.embed(("alpha",))
        assert base.calls == [("alpha",), ("alpha",)]

    asyncio.run(scenario())


def test_usage_keeps_the_provider_request_and_token_counts() -> None:
    async def scenario() -> None:
        base = RecordingEmbedding(requests=3)
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=8)
        result = await model.embed(("alpha", "beta"))
        assert result.usage.requests == 3
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 1
        cached = await model.embed(("alpha",))
        assert cached.usage.requests == 0
        assert cached.usage.input_tokens is None

    asyncio.run(scenario())


def test_late_caller_reuses_a_running_request() -> None:
    class Slow(RecordingEmbedding):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            self.started.set()
            await self.release.wait()
            return EmbeddingResult(vectors=tuple((1.0,) for _ in texts))

    async def scenario() -> None:
        base = Slow()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=8)
        first = asyncio.create_task(model.embed(("alpha",)))
        await base.started.wait()
        late = asyncio.create_task(model.embed(("alpha",)))
        await asyncio.sleep(0.05)
        base.release.set()
        assert (await first).vectors == ((1.0,),)
        assert (await late).vectors == ((1.0,),)
        assert base.calls == [("alpha",)]

    asyncio.run(scenario())


def test_cache_eviction_leaves_running_requests_untouched() -> None:
    class Slow(RecordingEmbedding):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            await self.release.wait()
            return EmbeddingResult(vectors=tuple((1.0,) for _ in texts))

    async def scenario() -> None:
        base = Slow()
        model = CoalescingEmbeddingModel(base, coalesce_ms=0, cache_size=1)
        first = asyncio.create_task(model.embed(("alpha",)))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(model.embed(("beta",)))
        await asyncio.sleep(0.05)
        base.release.set()
        await asyncio.gather(first, second)
        assert sorted(text for call in base.calls for text in call) == ["alpha", "beta"]

    asyncio.run(scenario())


def test_invalid_vector_count_raises_value_error() -> None:
    class Broken(RecordingEmbedding):
        async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
            self.calls.append(tuple(texts))
            return EmbeddingResult(vectors=())

    async def scenario() -> None:
        model = CoalescingEmbeddingModel(Broken(), coalesce_ms=0)
        with pytest.raises(ValueError):
            await model.embed(("alpha",))

    asyncio.run(scenario())
