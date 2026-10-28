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

"""Tests for the single-statement hybrid candidate query."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemoryEntryInput, MemorySearchRequest
from powercontext.builtin.artifacts.search import analyze_fts_query
from powercontext.builtin.inference import EmbeddingResult
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex
from powercontext.builtin.persistence.oceanbase.memory_index import _OCEANBASE_HYBRID_SEARCH_SQL
from powercontext.builtin.persistence.seekdb import SeekDBConfig
from powercontext.builtin.runtime import BuiltinConfig, open_builtin_contexts

PROFILE = EmbeddingProfile(
    profile_id="hybrid-parity-v1",
    model="deterministic",
    dimension=3,
    distance="l2",
    normalization="unit",
)

_TOPICS = {
    "alpha": (1.0, 0.0, 0.0),
    "beta": (0.0, 1.0, 0.0),
    "gamma": (0.0, 0.0, 1.0),
}


class _Embedding:
    profile = PROFILE

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        vectors = []
        for value in texts:
            folded = value.casefold()
            vectors.append(next((vector for name, vector in _TOPICS.items() if name in folded), (0.0, 0.0, 1.0)))
        return EmbeddingResult(vectors=tuple(vectors))


def test_hybrid_statement_keeps_the_version_join_inside_each_branch() -> None:
    # A join above the ordered branches re-sorts their rows and drops the
    # ranking the native indexes produced.
    statement = _OCEANBASE_HYBRID_SEARCH_SQL.format(fts_tag="", vector_tag="")
    branches, _, combined = statement.partition("SELECT\n    'fts' AS channel")
    assert branches.count("JOIN pc_memory_entry_versions") == 2
    assert "JOIN pc_memory_entry_versions" not in combined
    assert "UNION ALL" in combined


@pytest.mark.parametrize("query", ("alpha", "beta", "gamma"))
def test_hybrid_candidates_match_the_frozen_two_query_path(query: str) -> None:
    pytest.importorskip("pylibseekdb")

    async def scenario() -> None:
        directory = Path(tempfile.mkdtemp(prefix="pc-hybrid-"))
        config = BuiltinConfig(database=SeekDBConfig(path=directory / "db"))
        async with open_builtin_contexts(config, embedding_model=_Embedding()) as contexts:
            memory = (await contexts.get("project")).artifacts.memory
            entries = tuple(
                MemoryEntryInput(kind="fact", text=f"{name} record {index} discusses {name} workloads.")
                for index, name in enumerate(_TOPICS)
            )
            head = await memory.remember(memory=None, entries=entries, mode="append")
            assert head is not None

            index = contexts.index
            assert isinstance(index, CompositeMemoryIndex)
            frozen = CompositeMemoryIndex(*index.indexes)
            request = MemorySearchRequest(
                query=query,
                analyzed_query=analyze_fts_query(query),
                memories=(head.as_ref(),),
                candidate_limit=40,
                mode="hybrid",
                query_vector=_TOPICS[query],
                embedding_profile=PROFILE,
            )
            async with contexts.database.transaction() as connection:
                hybrid = await index._hybrid.search(connection, "project", request)
                two_query = await frozen.search(connection, "project", request)

            def identities(channels: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
                return (
                    tuple((hit.entry_id, hit.entry_version_id) for hit in channels.fts),
                    tuple((hit.entry_id, hit.entry_version_id) for hit in channels.vector),
                )

            assert identities(hybrid) == identities(two_query)

    import asyncio

    asyncio.run(scenario())
