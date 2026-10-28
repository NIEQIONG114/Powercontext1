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

"""Source lifecycle dependency-graph persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.persistence.tables import (
    SOURCE_LIFECYCLE_NODES_TABLE,
    SOURCE_LIFECYCLE_SUSPENSIONS_TABLE,
)

NodeType = Literal["source", "memory", "topic"]
NodePolicy = Literal["all", "any"]


@dataclass(frozen=True)
class StoredLifecycleNode:
    """One registered lifecycle node with its persisted binding."""

    node_id: str
    node_type: NodeType
    parents: tuple[str, ...]
    policy: NodePolicy
    revision: int
    sequence: int
    event_available: bool | None
    withdrawn: bool
    reference: str


class SourceLifecycleRepository:
    """Persist lifecycle registrations, event state, and suspended Topic artifacts."""

    async def load_nodes(self, connection: AsyncConnection, scope_id: str, /) -> tuple[StoredLifecycleNode, ...]:
        rows = (
            await connection.execute(
                select(
                    SOURCE_LIFECYCLE_NODES_TABLE.c.node_id,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.node_type,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.parents,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.policy,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.revision,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.sequence,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.event_available,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.withdrawn,
                    SOURCE_LIFECYCLE_NODES_TABLE.c.reference,
                ).where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
            )
        ).mappings()
        return tuple(
            StoredLifecycleNode(
                node_id=str(row["node_id"]),
                node_type=str(row["node_type"]),  # type: ignore[arg-type]
                parents=tuple(json.loads(str(row["parents"]))),
                policy=str(row["policy"]),  # type: ignore[arg-type]
                revision=int(row["revision"]),
                sequence=int(row["sequence"]),
                event_available=None if row["event_available"] is None else bool(row["event_available"]),
                withdrawn=bool(row["withdrawn"]),
                reference=str(row["reference"]),
            )
            for row in rows
        )

    async def upsert_node(self, connection: AsyncConnection, scope_id: str, node: StoredLifecycleNode, /) -> None:
        values = {
            "scope_id": scope_id,
            "node_id": node.node_id,
            "node_type": node.node_type,
            "parents": json.dumps(list(node.parents)),
            "policy": node.policy,
            "revision": node.revision,
            "sequence": node.sequence,
            "event_available": node.event_available,
            "withdrawn": node.withdrawn,
            "reference": node.reference,
        }
        updated = await connection.execute(
            update(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node.node_id,
            )
            .values(
                node_type=node.node_type,
                parents=values["parents"],
                policy=node.policy,
                revision=node.revision,
                sequence=node.sequence,
                event_available=node.event_available,
                withdrawn=node.withdrawn,
                reference=node.reference,
            )
        )
        if updated.rowcount:
            return
        await connection.execute(insert(SOURCE_LIFECYCLE_NODES_TABLE).values(**values))

    async def update_source_state(
        self,
        connection: AsyncConnection,
        scope_id: str,
        node_id: str,
        /,
        *,
        revision: int,
        sequence: int,
        event_available: bool,
        withdrawn: bool,
    ) -> None:
        await connection.execute(
            update(SOURCE_LIFECYCLE_NODES_TABLE)
            .where(
                SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node_id,
            )
            .values(
                revision=revision,
                sequence=sequence,
                event_available=event_available,
                withdrawn=withdrawn,
            )
        )

    async def load_suspensions(self, connection: AsyncConnection, scope_id: str, /) -> tuple[tuple[str, int], ...]:
        rows = await connection.execute(
            select(
                SOURCE_LIFECYCLE_SUSPENSIONS_TABLE.c.artifact_id,
                SOURCE_LIFECYCLE_SUSPENSIONS_TABLE.c.revision,
            ).where(SOURCE_LIFECYCLE_SUSPENSIONS_TABLE.c.scope_id == scope_id)
        )
        return tuple((str(row[0]), int(row[1])) for row in rows)

    async def suspend_topic(
        self,
        connection: AsyncConnection,
        scope_id: str,
        artifact_id: str,
        revision: int,
        /,
    ) -> None:
        await connection.execute(
            insert(SOURCE_LIFECYCLE_SUSPENSIONS_TABLE).values(
                scope_id=scope_id,
                artifact_id=artifact_id,
                revision=revision,
            )
        )

    async def resume_topic(self, connection: AsyncConnection, scope_id: str, artifact_id: str, /) -> None:
        await connection.execute(
            delete(SOURCE_LIFECYCLE_SUSPENSIONS_TABLE).where(
                SOURCE_LIFECYCLE_SUSPENSIONS_TABLE.c.scope_id == scope_id,
                SOURCE_LIFECYCLE_SUSPENSIONS_TABLE.c.artifact_id == artifact_id,
            )
        )
