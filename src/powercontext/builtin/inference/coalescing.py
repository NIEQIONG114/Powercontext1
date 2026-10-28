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
from dataclasses import dataclass, field
from typing import Any

from powercontext.builtin.inference.models import EmbeddingResult, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(slots=True)
class _BatchExecution:
    """One underlying batch request with its result future."""

    texts: tuple[str, ...]
    future: asyncio.Future[dict[str, tuple[float, ...]]]
    usage: InferenceUsage = field(default_factory=lambda: InferenceUsage(requests=0))


@dataclass(slots=True)
class _CallContext:
    """Tracks one embed() call's registration and results."""

    task: asyncio.Task[Any]
    texts: list[str]
    batches: list[_BatchExecution] = field(default_factory=list)
    text_to_batch: dict[str, _BatchExecution] = field(default_factory=dict)
    event: asyncio.Event = field(default_factory=asyncio.Event)


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
            raise ValueError("max_batch_size must be positive")
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must be non-negative")
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._active_calls: list[_CallContext] = []
        self._text_to_batch: dict[str, _BatchExecution] = {}
        self._scheduled = False

    @property
    def profile(self) -> Any:
        return self._model.profile

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        current_task = asyncio.current_task()
        unique_texts = list(dict.fromkeys(texts))

        cache_hits: dict[str, tuple[float, ...]] = {}
        call_context = _CallContext(task=current_task, texts=unique_texts)

        async with self._lock:
            for text in unique_texts:
                if text in self._cache:
                    self._cache.move_to_end(text)
                    cache_hits[text] = self._cache[text]

            texts_needing_work = [t for t in unique_texts if t not in cache_hits]

            if texts_needing_work:
                self._active_calls.append(call_context)

                for text in texts_needing_work:
                    if text in self._text_to_batch:
                        batch = self._text_to_batch[text]
                        call_context.batches.append(batch)
                        call_context.text_to_batch[text] = batch
                    else:
                        pass

                if not self._scheduled and self._active_calls:
                    self._scheduled = True
                    asyncio.create_task(self._process_coalesced_batches())

        if texts_needing_work:
            try:
                await call_context.event.wait()
            except asyncio.CancelledError:
                async with self._lock:
                    if call_context in self._active_calls:
                        self._active_calls.remove(call_context)
                raise

        result_vectors: list[tuple[float, ...]] = []
        usage_requests = 0
        usage_input_tokens: int | None = None
        usage_output_tokens: int | None = None

        waited_batches: set[int] = set()

        for text in texts:
            if text in cache_hits:
                result_vectors.append(cache_hits[text])
            else:
                batch = call_context.text_to_batch.get(text)
                if batch is None:
                    raise ValueError(f"No batch found for text: {text!r}")

                batch_id = id(batch)
                if batch_id not in waited_batches:
                    try:
                        batch_result = await asyncio.shield(batch.future)
                        waited_batches.add(batch_id)

                        if batch.usage.requests > 0:
                            usage_requests += 1
                            if batch.usage.input_tokens is not None:
                                usage_input_tokens = (usage_input_tokens or 0) + batch.usage.input_tokens
                            if batch.usage.output_tokens is not None:
                                usage_output_tokens = (usage_output_tokens or 0) + batch.usage.output_tokens
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        raise

                vector_map = batch.future.result() if batch.future.done() else {}
                if text not in vector_map:
                    raise ValueError(f"Embedding model did not return a vector for text: {text!r}")
                result_vectors.append(vector_map[text])

        return EmbeddingResult(
            vectors=tuple(result_vectors),
            usage=InferenceUsage(
                requests=usage_requests,
                input_tokens=usage_input_tokens,
                output_tokens=usage_output_tokens,
            ),
        )

    async def _process_coalesced_batches(self) -> None:
        if self._coalesce_seconds > 0:
            await asyncio.sleep(self._coalesce_seconds)

        async with self._lock:
            all_texts = []
            seen = set()
            calls_to_notify = list(self._active_calls)

            for call in calls_to_notify:
                for text in call.texts:
                    if text not in seen and text not in self._text_to_batch:
                        all_texts.append(text)
                        seen.add(text)

            for call in calls_to_notify:
                if call in self._active_calls:
                    self._active_calls.remove(call)

            self._scheduled = False

        if not all_texts:
            for call in calls_to_notify:
                call.event.set()
            return

        batches: list[_BatchExecution] = []
        text_to_batch: dict[str, _BatchExecution] = {}

        for i in range(0, len(all_texts), self._max_batch_size):
            batch_texts = tuple(all_texts[i:i + self._max_batch_size])
            future = asyncio.get_running_loop().create_future()
            batch = _BatchExecution(texts=batch_texts, future=future)
            batches.append(batch)
            for text in batch_texts:
                text_to_batch[text] = batch
            asyncio.create_task(self._execute_batch(batch))

        async with self._lock:
            self._text_to_batch.update(text_to_batch)

            for call in calls_to_notify:
                for text in call.texts:
                    if text in text_to_batch:
                        batch = text_to_batch[text]
                        if batch not in call.batches:
                            call.batches.append(batch)
                        call.text_to_batch[text] = batch

        for call in calls_to_notify:
            call.event.set()

    async def _execute_batch(self, batch: _BatchExecution) -> None:
        try:
            result = await self._model.embed(batch.texts)

            if len(result.vectors) != len(batch.texts):
                raise ValueError(
                    f"Embedding model returned {len(result.vectors)} vectors for {len(batch.texts)} texts"
                )

            vector_map = dict(zip(batch.texts, result.vectors, strict=True))
            batch.usage = result.usage

            async with self._lock:
                if self._cache_size > 0:
                    for text, vector in vector_map.items():
                        if text in self._cache:
                            self._cache.move_to_end(text)
                        else:
                            self._cache[text] = vector
                            if len(self._cache) > self._cache_size:
                                evicted_key, _ = self._cache.popitem(last=False)
                                if evicted_key in self._text_to_batch:
                                    del self._text_to_batch[evicted_key]

            if not batch.future.done():
                batch.future.set_result(vector_map)

        except asyncio.CancelledError:
            async with self._lock:
                self._release_batch(batch)
            if not batch.future.done():
                batch.future.cancel()
            raise

        except Exception as error:
            async with self._lock:
                # A failed batch must not be reused: the next call has to run the
                # underlying model again rather than replay the same failure.
                self._release_batch(batch)
            if not batch.future.done():
                batch.future.set_exception(error)

    def _release_batch(self, batch: _BatchExecution) -> None:
        """Drop a finished batch so later calls start fresh work."""

        for text in batch.texts:
            if self._text_to_batch.get(text) is batch:
                del self._text_to_batch[text]

    async def embed_query(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        return await self.embed(texts)


__all__ = ["CoalescingEmbeddingModel"]
