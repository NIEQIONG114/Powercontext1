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

"""Source lifecycle management for Memory and Topic Memory dependency tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.source_lifecycle import (
    SourceLifecycleRepository,
    StoredLifecycleNode,
)
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

NodeType = Literal["source", "memory", "topic"]
NodePolicy = Literal["all", "any"]
NodePurpose = Literal["generation", "audit"]


class SourceLifecycleError(ValueError):
    """Raised when a lifecycle operation violates constraints."""

    pass


class SourceLifecycleEvent(BaseModel):
    """A lifecycle event for a registered source node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    revision: int = Field(..., gt=0)
    sequence: int = Field(..., gt=0)
    available: bool

    @field_validator("node_id", "reason")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")  # noqa: TRY003
        return value

    @field_validator("revision", "sequence", mode="before")
    @classmethod
    def _reject_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("must be a positive integer")  # noqa: TRY003, TRY004
        return value


@dataclass
class _NodeRegistration:
    """One registered lifecycle node and its dependency edges.

    ``withdrawn`` records the availability this node was last given in the
    storage it drives: for a Source it mirrors the last event, and for a
    derived node it mirrors the availability computed from its parents.
    """

    node_id: str
    node_type: NodeType
    parents: tuple[str, ...]
    policy: NodePolicy
    reference: SourceRef | MemoryCitation | ArtifactRef
    revision: int = 1
    sequence: int = 0
    event_available: bool | None = None
    withdrawn: bool = False


@dataclass
class _ScopeGraph:
    """The dependency graph of one scope."""

    nodes: dict[str, _NodeRegistration] = field(default_factory=dict)

    def validate_dag(self, node_id: str, parents: tuple[str, ...]) -> None:
        if node_id in parents:
            raise SourceLifecycleError(f"Self-loop detected: {node_id}")  # noqa: TRY003
        if len(set(parents)) != len(parents):
            raise SourceLifecycleError(f"Duplicate parent detected: {node_id}")  # noqa: TRY003
        for parent in parents:
            if parent not in self.nodes:
                raise SourceLifecycleError(f"Parent not registered: {parent}")  # noqa: TRY003
        visited: set[str] = set()
        stack = [node_id]
        while stack:
            current = stack.pop()
            if current in visited:
                raise SourceLifecycleError(f"Cycle detected involving {current}")  # noqa: TRY003
            visited.add(current)
            if current in self.nodes:
                stack.extend(self.nodes[current].parents)

    def availability(self, node_id: str, memo: dict[str, bool] | None = None) -> bool:
        """Return whether one node is usable, following its parents' policy."""

        cached = {} if memo is None else memo
        if node_id in cached:
            return cached[node_id]
        node = self.nodes.get(node_id)
        if node is None:
            return False
        if node.node_type == "source":
            cached[node_id] = not node.withdrawn
            return cached[node_id]
        support = [self.availability(parent, cached) for parent in node.parents]
        available = all(support) if node.policy == "all" else any(support)
        cached[node_id] = available
        return available

    def available_nodes(self) -> list[str]:
        memo: dict[str, bool] = {}
        return sorted(node_id for node_id in self.nodes if self.availability(node_id, memo))


class SourceLifecycleService:
    """Manage one Scope's source lifecycle events and dependency propagation."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str, database: AsyncDatabase) -> None:
        self._runtime = runtime
        self._scope_id = scope_id
        self._database = database
        self._repository = SourceLifecycleRepository()

    async def register_source(self, node_id: str, reference: SourceRef, *, revision: int = 1) -> None:
        if revision < 1:
            raise SourceLifecycleError("Source revision must be a positive integer")  # noqa: TRY003
        await self._require_source(reference)
        async with self._database.transaction() as connection:
            graph = await self._load_graph(connection)
            existing = graph.nodes.get(node_id)
            if existing is not None:
                if (
                    existing.node_type != "source"
                    or existing.reference != reference
                    or existing.revision != revision
                ):
                    raise SourceLifecycleError(f"Node {node_id} already registered with different binding")  # noqa: TRY003
                return
            graph.nodes[node_id] = _NodeRegistration(
                node_id=node_id,
                node_type="source",
                parents=(),
                policy="all",
                reference=reference,
                revision=revision,
            )
            await self._save_node(connection, graph.nodes[node_id])

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: NodePolicy,
    ) -> None:
        if not parents:
            raise SourceLifecycleError("Memory node must have at least one parent")  # noqa: TRY003
        await self._require_memory(citation)
        async with self._database.transaction() as connection:
            graph = await self._load_graph(connection)
            existing = graph.nodes.get(node_id)
            if existing is not None:
                if (
                    existing.node_type != "memory"
                    or existing.reference != citation
                    or existing.parents != parents
                    or existing.policy != policy
                ):
                    raise SourceLifecycleError(f"Node {node_id} already registered with different binding")  # noqa: TRY003
                return
            graph.validate_dag(node_id, parents)
            node = _NodeRegistration(
                node_id=node_id,
                node_type="memory",
                parents=parents,
                policy=policy,
                reference=citation,
            )
            graph.nodes[node_id] = node
            node.withdrawn = not graph.availability(node_id)
            await self._save_node(connection, node)
        await self._apply_memory_state(node_id, not node.withdrawn)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: NodePolicy,
    ) -> None:
        if not parents:
            raise SourceLifecycleError("Topic node must have at least one parent")  # noqa: TRY003
        await self._require_topic(reference)
        async with self._database.transaction() as connection:
            graph = await self._load_graph(connection)
            existing = graph.nodes.get(node_id)
            if existing is not None:
                if (
                    existing.node_type != "topic"
                    or existing.reference != reference
                    or existing.parents != parents
                    or existing.policy != policy
                ):
                    raise SourceLifecycleError(f"Node {node_id} already registered with different binding")  # noqa: TRY003
                return
            graph.validate_dag(node_id, parents)
            node = _NodeRegistration(
                node_id=node_id,
                node_type="topic",
                parents=parents,
                policy=policy,
                reference=reference,
            )
            graph.nodes[node_id] = node
            node.withdrawn = not graph.availability(node_id)
            await self._save_node(connection, node)
        await self._apply_topic_state(node_id, not node.withdrawn)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        async with self._database.transaction() as connection:
            graph = await self._load_graph(connection)
            node = graph.nodes.get(event.node_id)
            if node is None:
                raise SourceLifecycleError(f"Unknown node: {event.node_id}")  # noqa: TRY003
            if node.node_type != "source":
                raise SourceLifecycleError(f"Event can only be applied to source nodes: {event.node_id}")  # noqa: TRY003
            if node.revision != event.revision:
                raise SourceLifecycleError(  # noqa: TRY003
                    f"Revision mismatch for {event.node_id}: expected {node.revision}, got {event.revision}"
                )
            if event.sequence < node.sequence:
                raise SourceLifecycleError(f"Sequence must be monotonically increasing for {event.node_id}")  # noqa: TRY003
            if event.sequence == node.sequence:
                if node.event_available != event.available:
                    raise SourceLifecycleError(  # noqa: TRY003
                        f"Conflicting replay for {event.node_id} at sequence {event.sequence}"
                    )
                return graph.available_nodes()
            node.sequence = event.sequence
            node.event_available = event.available
            node.withdrawn = not event.available
            changed = self._propagate(graph)
        # The native state is written before the event is recorded, so an
        # interrupted propagation leaves the event unapplied and the same event
        # can be retried. Native writes are idempotent, so a retry after a
        # partial failure only finishes what is left.
        for changed_id, available in changed:
            await self._apply_effects(changed_id, available)
        async with self._database.transaction() as connection:
            await self._save_node(connection, graph.nodes[event.node_id])
            for changed_id, _ in changed:
                await self._save_node(connection, graph.nodes[changed_id])
        return graph.available_nodes()

    async def resolve(
        self, node_id: str, *, purpose: NodePurpose
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError(f"Unknown purpose: {purpose}")  # noqa: TRY003
        async with self._database.transaction() as connection:
            graph = await self._load_graph(connection)
        node = graph.nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError(f"Unknown node: {node_id}")  # noqa: TRY003
        if purpose == "generation" and not graph.availability(node_id):
            raise SourceLifecycleError(f"Node {node_id} is not available for generation")  # noqa: TRY003
        return node.reference

    def _propagate(self, graph: _ScopeGraph) -> list[tuple[str, bool]]:
        memo: dict[str, bool] = {}
        changed: list[tuple[str, bool]] = []
        for node_id, node in graph.nodes.items():
            if node.node_type == "source":
                continue
            available = graph.availability(node_id, memo)
            if available == (not node.withdrawn):
                continue
            node.withdrawn = not available
            changed.append((node_id, available))
        return changed

    async def _apply_effects(self, node_id: str, available: bool) -> None:
        async with self._database.transaction() as connection:
            graph = await self._load_graph(connection)
        node = graph.nodes.get(node_id)
        if node is None:
            return
        if node.node_type == "memory":
            await self._apply_memory_state(node_id, available)
        elif node.node_type == "topic":
            await self._apply_topic_state(node_id, available)

    async def _apply_memory_state(self, node_id: str, available: bool) -> None:
        async with self._database.transaction() as connection:
            graph = await self._load_graph(connection)
        node = graph.nodes.get(node_id)
        if node is None or node.node_type != "memory":
            return
        citation = node.reference
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError(f"Node {node_id} does not reference a Memory entry")  # noqa: TRY003
        async with self._runtime._context(self._scope_id) as context:
            service = context.artifacts.memory
            current = await service.head(context.artifacts.memory_artifact_id)
            entry = next(
                (item for item in await service.entries(current) if item.entry_id == citation.entry_id),
                None,
            )
            if entry is None:
                raise SourceLifecycleError(f"Memory entry not found: {citation.entry_id}")  # noqa: TRY003
            if available:
                await service.reactivate(current, entries=(entry,))
            else:
                await service.forget(current, entries=(entry,))

    async def _apply_topic_state(self, node_id: str, available: bool) -> None:
        async with self._database.transaction() as connection:
            graph = await self._load_graph(connection)
        node = graph.nodes.get(node_id)
        if node is None or node.node_type != "topic":
            return
        reference = node.reference
        if not isinstance(reference, ArtifactRef):
            raise SourceLifecycleError(f"Node {node_id} does not reference a Topic artifact")  # noqa: TRY003
        async with self._database.transaction() as connection:
            if available:
                await self._repository.resume_topic(connection, self._scope_id, reference.artifact_id)
            else:
                await self._repository.suspend_topic(
                    connection,
                    self._scope_id,
                    reference.artifact_id,
                    reference.revision,
                )

    async def _require_source(self, reference: SourceRef) -> None:
        try:
            await self._runtime.records.for_scope(self._scope_id).get_source(
                reference.source_type,
                reference.source_id,
            )
        except Exception as error:
            raise SourceLifecycleError(  # noqa: TRY003
                f"Source is not readable in this scope: {reference.source_id}"
            ) from error

    async def _require_memory(self, citation: MemoryCitation) -> None:
        async with self._runtime._context(self._scope_id) as context:
            service = context.artifacts.memory
            try:
                await service.validate_citation(citation)
            except Exception as error:
                raise SourceLifecycleError(  # noqa: TRY003
                    f"Memory entry is not readable: {citation.entry_id}"
                ) from error

    async def _require_topic(self, reference: ArtifactRef) -> None:
        try:
            await self._runtime.records.for_scope(self._scope_id).get_artifact_revision(
                reference.family,
                reference.artifact_id,
                reference.revision,
            )
        except Exception as error:
            raise SourceLifecycleError(  # noqa: TRY003
                f"Topic artifact is not readable: {reference.artifact_id}"
            ) from error

    async def _load_graph(self, connection: AsyncConnection) -> _ScopeGraph:
        stored = await self._repository.load_nodes(connection, self._scope_id)
        graph = _ScopeGraph()
        for item in stored:
            graph.nodes[item.node_id] = _NodeRegistration(
                node_id=item.node_id,
                node_type=item.node_type,
                parents=item.parents,
                policy=item.policy,
                reference=_decode_reference(item),
                revision=item.revision,
                sequence=item.sequence,
                event_available=item.event_available,
                withdrawn=item.withdrawn,
            )
        return graph

    async def _save_node(self, connection: AsyncConnection, node: _NodeRegistration) -> None:
        await self._repository.upsert_node(
            connection,
            self._scope_id,
            StoredLifecycleNode(
                node_id=node.node_id,
                node_type=node.node_type,
                parents=node.parents,
                policy=node.policy,
                revision=node.revision,
                sequence=node.sequence,
                event_available=node.event_available,
                withdrawn=node.withdrawn,
                reference=_encode_reference(node.reference),
            ),
        )


def _encode_reference(reference: SourceRef | MemoryCitation | ArtifactRef) -> str:
    return json.dumps(reference.model_dump(mode="json"), sort_keys=True)


def _decode_reference(item: StoredLifecycleNode) -> SourceRef | MemoryCitation | ArtifactRef:
    payload = json.loads(item.reference)
    if item.node_type == "source":
        return SourceRef.model_validate(payload)
    if item.node_type == "memory":
        return MemoryCitation.model_validate(payload)
    return ArtifactRef.model_validate(payload)


__all__ = ["SourceLifecycleError", "SourceLifecycleEvent", "SourceLifecycleService"]
