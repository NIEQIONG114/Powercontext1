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

"""Cross-call embedding request coalescing and bounded vector caching."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True)
class _Slot:
    """One text whose vector is pending, in flight or already resolved.

    The slot outlives a single caller, so concurrent and late callers share
    the same underlying request instead of starting a second one.
    """

    text: str
    future: asyncio.Future[tuple[float, ...]]
    batch: _Batch | None = None


@dataclass(slots=True)
class _Batch:
    """One underlying model request and the slots that depend on it."""

    texts: tuple[str, ...]
    slots: tuple[_Slot, ...]
    usage: InferenceUsage = field(default_factory=lambda: InferenceUsage(requests=0))


class CoalescingEmbeddingModel:
    """Merge concurrent embedding requests and cache successful vectors."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must be non-negative")  # noqa: TRY003
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")  # noqa: TRY003
        self._model = model
        self._max_batch_size = int(max_batch_size)
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._slots: dict[str, _Slot] = {}
        self._pending: list[str] = []
        self._round: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def profile(self) -> Any:
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        unique = list(dict.fromkeys(texts))
        cached: dict[str, tuple[float, ...]] = {}
        slots: dict[str, _Slot] = {}

        async with self._lock:
            for text in unique:
                if text in self._cache:
                    self._cache.move_to_end(text)
                    cached[text] = self._cache[text]
                    continue
                slot = self._slots.get(text)
                if slot is None:
                    slot = _Slot(text=text, future=asyncio.get_running_loop().create_future())
                    # A text whose only caller disappears must not leave an
                    # unretrieved failure behind in the event loop.
                    slot.future.add_done_callback(_consume_outcome)
                    self._slots[text] = slot
                    self._pending.append(text)
                slots[text] = slot
            if self._pending and (self._round is None or self._round.done()):
                self._round = self._spawn(self._run_round())

        vectors: list[tuple[float, ...]] = []
        batches: dict[int, _Batch] = {}
        for text in texts:
            if text in cached:
                vectors.append(cached[text])
                continue
            slot = slots[text]
            vectors.append(await asyncio.shield(slot.future))
            if slot.batch is not None:
                batches[id(slot.batch)] = slot.batch

        return EmbeddingResult(vectors=tuple(vectors), usage=_merge_usage(batches.values()))

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return await self.embed(texts)

    async def _run_round(self) -> None:
        """Close one coalescing window and turn the collected texts into batches."""

        if self._coalesce_seconds > 0:
            await asyncio.sleep(self._coalesce_seconds)
        async with self._lock:
            texts = list(self._pending)
            self._pending.clear()
            self._round = None
            batches = [
                _Batch(texts=tuple(chunk), slots=tuple(self._slots[text] for text in chunk))
                for chunk in _chunks(texts, self._max_batch_size)
            ]
            for batch in batches:
                for slot in batch.slots:
                    slot.batch = batch
        if not batches:
            return
        await asyncio.gather(*(self._execute_batch(batch) for batch in batches), return_exceptions=True)

    async def _execute_batch(self, batch: _Batch) -> None:
        try:
            result = await self._model.embed(batch.texts)
            vectors = _checked_vectors(batch.texts, result.vectors)
            batch.usage = result.usage
            self._cache_vectors(vectors)
        except asyncio.CancelledError:
            _fail(batch, asyncio.CancelledError())
            raise
        except Exception as error:
            # A failed batch is never cached, so the next caller retries it.
            _fail(batch, error)
        else:
            _settle(batch, vectors)
        finally:
            self._release(batch)

    def _cache_vectors(self, vectors: dict[str, tuple[float, ...]]) -> None:
        if self._cache_size <= 0:
            return
        for text, vector in vectors.items():
            if text in self._cache:
                self._cache.move_to_end(text)
            else:
                self._cache[text] = vector
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def _release(self, batch: _Batch) -> None:
        """Stop tracking a finished request so later calls start fresh work.

        The registry only changes without an intervening await, so the removal
        cannot interleave with a call that registers the same text.
        """

        for slot in batch.slots:
            if self._slots.get(slot.text) is slot:
                del self._slots[slot.text]

    def _spawn(self, coroutine: Any) -> asyncio.Task[None]:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._retire)
        return task

    def _retire(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            task.exception()


def _chunks(texts: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(texts), size):
        yield texts[start : start + size]


def _checked_vectors(texts: tuple[str, ...], vectors: tuple[tuple[float, ...], ...]) -> dict[str, tuple[float, ...]]:
    """Pair the provider vectors with the texts they belong to."""

    if len(vectors) != len(texts):
        raise ValueError(  # noqa: TRY003
            f"Embedding model returned {len(vectors)} vectors for {len(texts)} texts"
        )
    return dict(zip(texts, vectors, strict=True))


def _settle(batch: _Batch, vectors: dict[str, tuple[float, ...]]) -> None:
    """Resolve every slot that waits for this batch."""

    for slot in batch.slots:
        if not slot.future.done():
            slot.future.set_result(vectors[slot.text])


def _fail(batch: _Batch, error: BaseException) -> None:
    """Report a failed batch to every slot that waits for it."""

    for slot in batch.slots:
        if not slot.future.done():
            slot.future.set_exception(error)


def _consume_outcome(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


def _merge_usage(batches: Iterable[_Batch]) -> InferenceUsage:
    """Sum the provider usage of the distinct batches one call waited for."""

    requests = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    for batch in batches:
        usage = batch.usage
        requests += usage.requests
        if usage.input_tokens is not None:
            input_tokens = (input_tokens or 0) + usage.input_tokens
        if usage.output_tokens is not None:
            output_tokens = (output_tokens or 0) + usage.output_tokens
    return InferenceUsage(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens)


__all__ = ["CoalescingEmbeddingModel"]
