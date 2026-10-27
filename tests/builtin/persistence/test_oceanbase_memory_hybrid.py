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

"""One statement must carry both retrieval channels for a hybrid search."""

from __future__ import annotations

import asyncio
import re
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import mysql
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef
from powercontext.builtin.artifacts.memory import EmbeddingProfile, MemorySearchRequest
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex
from powercontext.builtin.persistence.oceanbase.memory_index import (
    OceanBaseMemoryFTSIndex,
    OceanBaseMemoryHybridIndex,
    OceanBaseMemoryVectorIndex,
    _embedding_hash,
)
from powercontext.builtin.tags import TagFilter

_HASH = "a" * 64


def _profile() -> EmbeddingProfile:
    return EmbeddingProfile(
        profile_id="hybrid-test-v1",
        model="deterministic",
        dimension=3,
        distance="l2",
        normalization="unit",
    )


def _request(mode: str = "hybrid") -> MemorySearchRequest:
    return MemorySearchRequest(
        query="alpha",
        analyzed_query="alpha",
        memories=(ArtifactRef(family="memory", artifact_id="mem_art_a", revision=3),),
        candidate_limit=32,
        mode=cast(Any, mode),
        query_vector=(1.0, 0.0, 0.0),
        embedding_profile=_profile(),
    )


def _connection(rows: list[dict[str, object]]) -> AsyncMock:
    """A connection whose vector completeness guard passes."""

    connection = AsyncMock(spec=AsyncConnection)
    connection.dialect = mysql.dialect()
    connection.scalar.return_value = 3

    def execute(statement, *_args, **_kwargs):
        rendered = str(statement.compile(dialect=mysql.dialect()))
        result = MagicMock()
        if "l2_distance" in rendered or "MATCH" in rendered.upper():
            result.mappings = MagicMock(return_value=rows)
            return result
        result.all = MagicMock(
            return_value=[("mem_ent_a", "mem_ver_a", _HASH)]
            if "pc_memory_vector_entries" not in rendered
            else [
                (
                    "mem_ent_a",
                    "mem_ver_a",
                    _HASH,
                    _embedding_hash(_profile(), _HASH),
                )
            ]
        )
        return result

    connection.execute.side_effect = execute
    return connection


def _rows() -> list[dict[str, object]]:
    return [
        {
            "channel": "fts",
            "memory_artifact_id": "mem_art_a",
            "head_revision": 3,
            "entry_id": "mem_ent_a",
            "entry_version_id": "mem_ver_a",
            "text": "Alpha semantic record.",
            "distance": None,
        },
        {
            "channel": "vector",
            "memory_artifact_id": "mem_art_a",
            "head_revision": 3,
            "entry_id": "mem_ent_a",
            "entry_version_id": "mem_ver_a",
            "text": "Alpha semantic record.",
            "distance": 0.0,
        },
    ]


def test_hybrid_uses_one_statement_carrying_both_channels() -> None:
    async def scenario() -> None:
        connection = _connection(_rows())
        vector = OceanBaseMemoryVectorIndex(_profile())
        hybrid = OceanBaseMemoryHybridIndex(vector)
        composite = CompositeMemoryIndex(OceanBaseMemoryFTSIndex(), vector, hybrid=hybrid)

        result = await composite.search(cast(AsyncConnection, connection), "scope-a", _request())

        statements = [
            str(call.args[0].compile(dialect=mysql.dialect()))
            for call in connection.execute.await_args_list
        ]
        candidate_statements = [
            statement
            for statement in statements
            if "pc_memory_entry_heads" in statement
            and "pc_memory_vector_entries" in statement
            and "AGAINST" in statement.upper()
            and "l2_distance" in statement.lower()
        ]
        assert len(candidate_statements) == 1, statements
        statement = candidate_statements[0]
        # The native indexes carry the work; nothing is computed in Python.
        assert "MATCH (f.searchable_text) AGAINST" in statement
        assert "l2_distance(m.embedding" in statement
        assert "ORDER BY l2_distance(m.embedding" in statement
        assert "APPROXIMATE" in statement
        assert re.search(r"APPROXIMATE\s+LIMIT\s+", statement)
        # Filters and the candidate limit stay inside the database.
        assert "f.scope_id = %s" in statement
        assert "m.scope_id = %s" in statement
        assert statement.count("memory_artifact_id IN") == 2
        assert statement.count("LIMIT %s") == 2
        # Both channels come back, split by the marker column.
        assert len(result.fts) == 1
        assert len(result.vector) == 1
        assert result.fts[0].entry_id == "mem_ent_a"
        assert result.fts[0].distance is None
        assert result.vector[0].distance == 0.0
        assert result.fts[0].memory_ref == ArtifactRef(
            family="memory", artifact_id="mem_art_a", revision=3
        )

    asyncio.run(scenario())


def test_single_channel_modes_keep_their_own_statements() -> None:
    async def scenario() -> None:
        connection = _connection([])
        vector = OceanBaseMemoryVectorIndex(_profile())
        hybrid = OceanBaseMemoryHybridIndex(vector)
        composite = CompositeMemoryIndex(OceanBaseMemoryFTSIndex(), vector, hybrid=hybrid)

        await composite.search(cast(AsyncConnection, connection), "scope-a", _request("fts"))
        await composite.search(cast(AsyncConnection, connection), "scope-a", _request("vector"))

        statements = [
            str(call.args[0].compile(dialect=mysql.dialect()))
            for call in connection.execute.await_args_list
        ]
        hybrid_statements = [
            statement
            for statement in statements
            if "pc_memory_entry_heads" in statement
            and "pc_memory_vector_entries" in statement
            and "AGAINST" in statement.upper()
            and "l2_distance" in statement.lower()
        ]
        assert hybrid_statements == []
        assert any("AGAINST" in statement for statement in statements)
        assert any("l2_distance" in statement for statement in statements)

    asyncio.run(scenario())


def test_tagged_hybrid_keeps_one_statement_and_database_side_filters() -> None:
    async def scenario() -> None:
        connection = _connection([])
        vector = OceanBaseMemoryVectorIndex(_profile())
        hybrid = OceanBaseMemoryHybridIndex(vector)
        request = _request().model_copy(
            update={"tag_filter": TagFilter(tags=("selected",))}
        )

        await hybrid.search(cast(AsyncConnection, connection), "scope-a", request)

        statements = [
            str(call.args[0].compile(dialect=mysql.dialect()))
            for call in connection.execute.await_args_list
        ]
        statements = [statement for statement in statements if "pc_artifact_tags" in statement]
        assert len(statements) == 1
        statement = statements[0]
        assert "pc_artifact_tags" in statement
        assert "tags.scope_id = f.scope_id" in statement
        assert "tags.scope_id = m.scope_id" in statement
        # The tagged vector path of the frozen backend ranks by exact distance.
        assert "APPROXIMATE" not in statement

    asyncio.run(scenario())
