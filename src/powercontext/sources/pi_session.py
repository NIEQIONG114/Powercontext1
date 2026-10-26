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

"""Pi session branch collection with checkpoint recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.adapters import SourceAdapter
from powercontext.sources.connectors import (
    Connector,
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence


class PiSessionInput(BaseModel):
    """Input for parsing a Pi session JSONL file."""

    model_config = ConfigDict(frozen=True)

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """A single visible event from a Pi session branch."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured Source for a single Pi event."""

    model_config = ConfigDict(frozen=True)

    source_type: str = "pi-event"
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]

    @property
    def source_id(self) -> str:
        return pi_event_source_id(self.session_id, self.event_id)

    @property
    def name(self) -> str:
        return self.source_id


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Compute the stable source identity for a Pi event.

    Format: pi:<url-encoded-session-id>:<url-encoded-event-id>
    """
    encoded_session = quote(session_id, safe="")
    encoded_event = quote(event_id, safe="")
    identity = f"pi:{encoded_session}:{encoded_event}"

    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")

    return identity


def _extract_visible_content(record: dict[str, Any]) -> tuple[str, str] | None:
    """Extract (role, content) from a record if it has visible content.

    Returns None for control events that don't generate text evidence.
    """
    event_type = record.get("type", "")

    if event_type == "message":
        message = record.get("message", {})
        role = message.get("role", "")
        content_value = message.get("content")

        if role == "toolResult":
            tool_name = message.get("toolName", "")
            tool_call_id = message.get("toolCallId", "")
            text = _extract_text_from_content(content_value)
            body = f"[tool_result {tool_name} {tool_call_id}] {text}"
            return (role, body)

        if role == "bashExecution":
            command = message.get("command", "")
            output = message.get("output", "")
            body = f"$ {command}\n{output}"
            return (role, body)

        if isinstance(content_value, str):
            return (role, content_value)

        if isinstance(content_value, list):
            parts = []
            tool_calls = []
            for block in content_value:
                block_type = block.get("type", "")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "toolCall":
                    name = block.get("name", "")
                    call_id = block.get("id", "")
                    args = block.get("arguments", {})
                    args_json = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    tool_calls.append(f"[tool_call {name} {call_id}] {args_json}")

            if tool_calls:
                return (role, "\n".join(tool_calls))
            if parts:
                return (role, "\n".join(parts))

        return None

    if event_type in ("compaction", "branch_summary"):
        role = event_type
        summary = record.get("summary", "")
        return (role, summary)

    if event_type == "custom_message":
        content_value = record.get("content")
        if isinstance(content_value, str):
            return ("custom", content_value)
        if isinstance(content_value, list):
            parts = []
            for block in content_value:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            if parts:
                return ("custom", "\n".join(parts))
        return None

    return None


def _extract_text_from_content(content: Any) -> str:
    """Extract text from various content formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def _is_control_event(record: dict[str, Any]) -> bool:
    """Check if a record is a control event that doesn't generate text evidence."""
    event_type = record.get("type", "")
    if event_type not in ("message", "compaction", "branch_summary", "custom_message"):
        return True

    if event_type == "message":
        message = record.get("message", {})
        role = message.get("role", "")
        content_value = message.get("content")

        if role == "bashExecution":
            return False

        if role in ("user", "assistant", "toolResult"):
            if isinstance(content_value, str) and content_value:
                return False
            if isinstance(content_value, list):
                for block in content_value:
                    block_type = block.get("type", "")
                    if block_type == "text" and block.get("text"):
                        return False
                    if block_type == "toolCall":
                        return False
            return True

        return True

    if event_type in ("compaction", "branch_summary"):
        return not record.get("summary")

    if event_type == "custom_message":
        content_value = record.get("content")
        if isinstance(content_value, str) and content_value:
            return False
        if isinstance(content_value, list):
            for block in content_value:
                if block.get("type") == "text" and block.get("text"):
                    return False
        return True

    return True


def parse_pi_session(input: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse a Pi session JSONL file and return events on the selected branch.

    - UTF-8 with optional BOM
    - First record must be type=session, version=3, with non-empty id
    - Events form a tree via id/parentId relationships
    - Returns ancestor chain from root to selected leaf
    - Control events maintain graph connectivity but don't generate text evidence
    """
    content = input.content
    leaf_id = input.leaf_id

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 encoding: {error}") from error

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty session file")

    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON: {error}") from error
        records.append(record)

    if not records:
        raise ValueError("empty session file")

    header = records[0]
    if not isinstance(header, dict):
        raise ValueError("first record must be a JSON object")
    if header.get("type") != "session":
        raise ValueError("first record must be type=session")
    if header.get("version") != 3:
        raise ValueError("session version must be 3")
    session_id = header.get("id")
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session id must be a non-empty string")

    event_records = records[1:]
    by_id: dict[str, dict[str, Any]] = {}
    parent_map: dict[str, str | None] = {}

    for record in event_records:
        event_id = record.get("id")
        if not event_id or not isinstance(event_id, str):
            raise ValueError("event must have a non-empty string id")
        if event_id in by_id:
            raise ValueError(f"duplicate event id: {event_id!r}")

        parent_id = record.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError(f"parentId must be null or string for event {event_id!r}")
        if parent_id is not None and parent_id not in by_id and parent_id != event_id:
            if parent_id not in {r.get("id") for r in event_records}:
                pass

        by_id[event_id] = record
        parent_map[event_id] = parent_id

    for event_id, parent_id in parent_map.items():
        if parent_id is not None:
            if parent_id == event_id:
                raise ValueError(f"event {event_id!r} has self-referencing parentId")
            if parent_id not in by_id:
                raise ValueError(f"event {event_id!r} references missing parent {parent_id!r}")

    for event_id in by_id:
        visited: set[str] = set()
        current = event_id
        while current is not None:
            if current in visited:
                raise ValueError(f"cycle detected involving event {current!r}")
            visited.add(current)
            current = parent_map.get(current)

    if leaf_id is None:
        if event_records:
            leaf_id = event_records[-1].get("id")
        else:
            return ()

    if leaf_id not in by_id:
        raise ValueError(f"leaf_id {leaf_id!r} not found in session")

    ancestor_chain: list[str] = []
    current: str | None = leaf_id
    while current is not None:
        ancestor_chain.append(current)
        current = parent_map.get(current)
    ancestor_chain.reverse()

    events: list[PiEventInput] = []
    for event_id in ancestor_chain:
        record = by_id[event_id]

        if _is_control_event(record):
            parent_id = parent_map.get(event_id)
            events.append(
                PiEventInput(
                    session_id=session_id,
                    event_id=event_id,
                    parent_id=parent_id,
                    event_type=record.get("type", ""),
                    role="",
                    content="",
                    record=record,
                )
            )
            continue

        result = _extract_visible_content(record)
        if result is None:
            parent_id = parent_map.get(event_id)
            events.append(
                PiEventInput(
                    session_id=session_id,
                    event_id=event_id,
                    parent_id=parent_id,
                    event_type=record.get("type", ""),
                    role="",
                    content="",
                    record=record,
                )
            )
            continue

        role, content_text = result
        parent_id = parent_map.get(event_id)
        formatted_content = f"{role}: {content_text}"

        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=parent_id,
                event_type=record.get("type", ""),
                role=role,
                content=formatted_content,
                record=record,
            )
        )

    visible_events = [e for e in events if e.content]
    return tuple(visible_events)


class PiEventTextEvidenceProjection:
    """Project PiEventSource to TextEvidence for memory retrieval."""

    name = "powercontext.text-evidence"
    version = "1"
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pi-event",
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


class _PiEventAdapter:
    """Internal adapter for PiEventInput -> PiEventSource."""

    input_class = PiEventInput
    name = "pi-event"
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(
            name=pi_event_source_id(value.session_id, value.event_id),
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


PI_EVENT_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=_PiEventAdapter(),
    version="1",
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Connector for incremental Pi session collection with checkpoint recovery."""

    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        content = self._path.read_bytes()
        pi_input = PiSessionInput(content=content, leaf_id=self._leaf_id)

        try:
            events = parse_pi_session(pi_input)
        except ValueError:
            raise

        checkpoint_data = session.checkpoint or {}
        imported_event_ids: set[str] = set(checkpoint_data.get("imported_event_ids", []))
        imported_event_hashes: dict[str, str] = dict(checkpoint_data.get("imported_event_hashes", {}))

        for event in events:
            if event.event_id in imported_event_ids:
                stored_hash = imported_event_hashes.get(event.event_id)
                current_hash = _compute_event_hash(event)
                if stored_hash and stored_hash != current_hash:
                    raise ValueError(
                        f"imported event {event.event_id!r} has been modified"
                    )
                continue

            outcome = await session.submit(
                event.event_id,
                "pi-event",
                event,
            )

            if outcome.status == ConnectorSubmissionStatus.ACCEPTED:
                imported_event_ids.add(event.event_id)
                imported_event_hashes[event.event_id] = _compute_event_hash(event)

        new_checkpoint: JsonValue = {
            "imported_event_ids": sorted(imported_event_ids),
            "imported_event_hashes": imported_event_hashes,
        }

        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE,
            checkpoint=new_checkpoint,
        )


def _compute_event_hash(event: PiEventInput) -> str:
    """Compute a hash of the event for mutation detection."""
    canonical = json.dumps(event.record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "PI_EVENT_SOURCE_DEFINITION",
    "PiEventInput",
    "PiEventSource",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
