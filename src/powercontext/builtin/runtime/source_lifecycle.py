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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


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
            raise ValueError("must not be blank")
        return value


@dataclass
class _NodeRegistration:
    """Internal representation of a registered node."""

    node_id: str
    node_type: Literal["source", "memory", "topic"]
    parents: tuple[str, ...]
    policy: Literal["all", "any"]
    available: bool
    reference: SourceRef | MemoryCitation | ArtifactRef
    revision: int = 1
    sequence: int = 0


@dataclass
class _ScopeGraph:
    """Dependency graph for one scope."""

    nodes: dict[str, _NodeRegistration] = field(default_factory=dict)

    def validate_dag(self, node_id: str, parents: tuple[str, ...]) -> None:
        if node_id in parents:
            raise SourceLifecycleError(f"Self-loop detected: {node_id}")
        for parent in parents:
            if parent not in self.nodes:
                raise SourceLifecycleError(f"Parent not registered: {parent}")
        if parents:
            visited = set()
            stack = [node_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    raise SourceLifecycleError(f"Cycle detected involving {current}")
                visited.add(current)
                if current in self.nodes:
                    stack.extend(self.nodes[current].parents)

    def compute_availability(self, node_id: str, withdrawn: set[str]) -> bool:
        if node_id in withdrawn:
            return False
        node = self.nodes.get(node_id)
        if node is None or not node.parents:
            return node_id not in withdrawn
        parent_availability = [self.compute_availability(p, withdrawn) for p in node.parents]
        if node.policy == "all":
            return all(parent_availability)
        return any(parent_availability)


class SourceLifecycleService:
    """Manages source lifecycle events and dependency propagation for one scope."""

    def __init__(self, runtime: BuiltinRuntime, scope_id: str, database: AsyncDatabase) -> None:
        self._runtime = runtime
        self._scope_id = scope_id
        self._database = database
        self._graph = _ScopeGraph()
        self._withdrawn: set[str] = set()

    async def register_source(
        self, node_id: str, reference: SourceRef, *, revision: int = 1
    ) -> None:
        if node_id in self._graph.nodes:
            existing = self._graph.nodes[node_id]
            if existing.node_type != "source" or existing.reference != reference:
                raise SourceLifecycleError(f"Node {node_id} already registered with different binding")
            return
        self._graph.nodes[node_id] = _NodeRegistration(
            node_id=node_id,
            node_type="source",
            parents=(),
            policy="all",
            available=True,
            reference=reference,
            revision=revision,
        )

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        if not parents:
            raise SourceLifecycleError("Memory node must have at least one parent")
        if node_id in self._graph.nodes:
            existing = self._graph.nodes[node_id]
            if existing.node_type != "memory" or existing.reference != citation:
                raise SourceLifecycleError(f"Node {node_id} already registered with different binding")
            return
        self._graph.validate_dag(node_id, parents)
        initial_available = self._graph.compute_availability(parents[0], self._withdrawn) if policy == "any" else all(
            self._graph.compute_availability(p, self._withdrawn) for p in parents
        )
        self._graph.nodes[node_id] = _NodeRegistration(
            node_id=node_id,
            node_type="memory",
            parents=parents,
            policy=policy,
            available=initial_available,
            reference=citation,
        )
        await self._apply_memory_state(node_id, initial_available)

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        *,
        parents: tuple[str, ...],
        policy: Literal["all", "any"],
    ) -> None:
        if not parents:
            raise SourceLifecycleError("Topic node must have at least one parent")
        if node_id in self._graph.nodes:
            existing = self._graph.nodes[node_id]
            if existing.node_type != "topic" or existing.reference != reference:
                raise SourceLifecycleError(f"Node {node_id} already registered with different binding")
            return
        self._graph.validate_dag(node_id, parents)
        initial_available = self._graph.compute_availability(parents[0], self._withdrawn) if policy == "any" else all(
            self._graph.compute_availability(p, self._withdrawn) for p in parents
        )
        self._graph.nodes[node_id] = _NodeRegistration(
            node_id=node_id,
            node_type="topic",
            parents=parents,
            policy=policy,
            available=initial_available,
            reference=reference,
        )

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        node = self._graph.nodes.get(event.node_id)
        if node is None:
            raise SourceLifecycleError(f"Unknown node: {event.node_id}")
        if node.node_type != "source":
            raise SourceLifecycleError(f"Event can only be applied to source nodes: {event.node_id}")
        if node.revision != event.revision:
            raise SourceLifecycleError(f"Revision mismatch for {event.node_id}: expected {node.revision}, got {event.revision}")
        if event.sequence < node.sequence:
            raise SourceLifecycleError(f"Sequence must be monotonically increasing for {event.node_id}")
        if event.sequence == node.sequence:
            return self.available_nodes()
        node.sequence = event.sequence
        if event.available:
            self._withdrawn.discard(event.node_id)
        else:
            self._withdrawn.add(event.node_id)
        await self._propagate_state()
        return self.available_nodes()

    def available_nodes(self) -> list[str]:
        active = set()
        for node_id in self._graph.nodes:
            if self._graph.compute_availability(node_id, self._withdrawn):
                active.add(node_id)
        return sorted(active)

    async def resolve(
        self, node_id: str, *, purpose: Literal["generation", "audit"]
    ) -> SourceRef | MemoryCitation | ArtifactRef:
        node = self._graph.nodes.get(node_id)
        if node is None:
            raise SourceLifecycleError(f"Unknown node: {node_id}")
        is_available = self._graph.compute_availability(node_id, self._withdrawn)
        if purpose == "generation" and not is_available:
            raise SourceLifecycleError(f"Node {node_id} is not available for generation")
        return node.reference

    async def _propagate_state(self) -> None:
        for node_id, node in self._graph.nodes.items():
            if node.node_type == "memory":
                available = self._graph.compute_availability(node_id, self._withdrawn)
                if available != node.available:
                    node.available = available
                    await self._apply_memory_state(node_id, available)

    async def _apply_memory_state(self, node_id: str, available: bool) -> None:
        node = self._graph.nodes[node_id]
        if node.node_type != "memory":
            return
        citation = node.reference
        assert isinstance(citation, MemoryCitation)
        memory_service = self._runtime.memory.for_scope(self._scope_id)
        try:
            if available:
                await memory_service.revise(
                    citation=citation,
                    text=None,
                    state="active",
                )
            else:
                await memory_service.retire(citation=citation)
        except Exception:
            pass


__all__ = ["SourceLifecycleError", "SourceLifecycleEvent", "SourceLifecycleService"]
