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

"""Context pack and delta protocol for incremental context transmission."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Self

from powercontext.builtin.runtime.errors import PowerContextError

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(PowerContextError, ValueError):
    """Protocol or chunk validation error in context pack operations."""

    pass


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A complete context pack with chunked items and digest verification."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> Self:
        """Construct a pack from a bytes tuple, computing digest and byte_count."""
        digest = _compute_digest(items)
        pack = cls(items=items, digest=digest, byte_count=0)
        byte_count = len(pack.to_bytes())
        return cls(items=items, digest=digest, byte_count=byte_count)

    @classmethod
    def from_content(cls, content: str, *, chunk_bytes: int = 512) -> Self:
        """Chunk content string into a pack with deterministic boundaries.
        
        Each chunk must be valid UTF-8 and not exceed chunk_bytes.
        """
        if chunk_bytes < 1:
            raise ContextPackError("chunk_bytes must be positive")

        content_bytes = content.encode("utf-8")
        if not content_bytes:
            return cls.build(())

        items: list[bytes] = []
        pos = 0
        while pos < len(content_bytes):
            # Find the largest end position <= pos + chunk_bytes that is a valid UTF-8 boundary
            end = min(pos + chunk_bytes, len(content_bytes))

            # Try to decode the chunk; if it fails, we need to back up to find a valid boundary
            while end > pos:
                chunk = content_bytes[pos:end]
                try:
                    chunk.decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end -= 1
            else:
                # Could not find a valid UTF-8 boundary within chunk_bytes
                raise ContextPackError(
                    f"cannot form valid UTF-8 chunk within {chunk_bytes} bytes"
                )

            items.append(chunk)
            pos = end

        return cls.build(tuple(items))

    def to_bytes(self) -> bytes:
        """Serialize pack to JSON wire format."""
        payload = {
            "schema": PACK_SCHEMA,
            "items": [base64.b64encode(item).decode("ascii") for item in self.items],
            "digest": self.digest,
            "byte_count": 0,  # Placeholder, will be computed
        }
        # Compute byte_count including itself
        temp_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["byte_count"] = len(temp_json)
        # Recompute with actual byte_count
        final_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        # Verify byte_count is stable
        if len(final_json) != payload["byte_count"]:
            payload["byte_count"] = len(final_json)
            final_json = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return final_json

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Deserialize pack from JSON wire format with validation."""
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as e:
            raise ContextPackError(f"invalid JSON: {e}") from e

        # Validate required fields
        required = {"schema", "items", "digest", "byte_count"}
        if set(payload.keys()) != required:
            missing = required - set(payload.keys())
            extra = set(payload.keys()) - required
            parts = []
            if missing:
                parts.append(f"missing: {missing}")
            if extra:
                parts.append(f"extra: {extra}")
            raise ContextPackError(f"invalid pack fields: {', '.join(parts)}")

        if payload["schema"] != PACK_SCHEMA:
            raise ContextPackError(f"invalid schema: {payload['schema']}")

        # Decode items
        try:
            items = tuple(base64.b64decode(item) for item in payload["items"])
        except Exception as e:
            raise ContextPackError(f"invalid base64 encoding: {e}") from e

        # Verify digest
        expected_digest = _compute_digest(items)
        if payload["digest"] != expected_digest:
            raise ContextPackError("digest mismatch")

        # Verify byte_count
        pack = cls(items=items, digest=payload["digest"], byte_count=payload["byte_count"])
        if pack.byte_count != payload["byte_count"]:
            raise ContextPackError("byte_count mismatch")

        return pack


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """A delta between two context packs using prefix reuse."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @property
    def byte_count(self) -> int:
        """Compute serialized byte count."""
        return len(self.to_bytes())

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> Self:
        """Create a delta from base to target using common prefix."""
        # Find common prefix length
        prefix_len = 0
        for i in range(min(len(base.items), len(target.items))):
            if base.items[i] == target.items[i]:
                prefix_len = i + 1
            else:
                break

        appended = target.items[prefix_len:]
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix_len,
            appended=appended,
        )

    def to_bytes(self) -> bytes:
        """Serialize delta to JSON wire format."""
        payload = {
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": [base64.b64encode(item).decode("ascii") for item in self.appended],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Deserialize delta from JSON wire format with validation."""
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as e:
            raise ContextPackError(f"invalid JSON: {e}") from e

        # Validate required fields
        required = {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
        if set(payload.keys()) != required:
            missing = required - set(payload.keys())
            extra = set(payload.keys()) - required
            parts = []
            if missing:
                parts.append(f"missing: {missing}")
            if extra:
                parts.append(f"extra: {extra}")
            raise ContextPackError(f"invalid delta fields: {', '.join(parts)}")

        if payload["schema"] != DELTA_SCHEMA:
            raise ContextPackError(f"invalid schema: {payload['schema']}")

        # Validate prefix_items
        prefix = payload["prefix_items"]
        if not isinstance(prefix, int) or isinstance(prefix, bool):
            raise ContextPackError(f"prefix_items must be integer, got {type(prefix).__name__}")
        if prefix < 0:
            raise ContextPackError(f"prefix_items must be non-negative, got {prefix}")

        # Decode appended
        try:
            appended = tuple(base64.b64decode(item) for item in payload["appended"])
        except Exception as e:
            raise ContextPackError(f"invalid base64 encoding: {e}") from e

        return cls(
            base_digest=payload["base_digest"],
            target_digest=payload["target_digest"],
            prefix_items=prefix,
            appended=appended,
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    """Apply a delta to a base pack to reconstruct the target pack."""
    # Validate base digest
    if base.digest != delta.base_digest:
        raise ContextPackError("base digest mismatch")

    # Validate prefix_items bounds
    if delta.prefix_items > len(base.items):
        raise ContextPackError(
            f"prefix_items {delta.prefix_items} exceeds base items count {len(base.items)}"
        )

    # Reconstruct target items
    target_items = base.items[:delta.prefix_items] + delta.appended

    # Build target pack
    target = ContextPack.build(target_items)

    # Validate target digest
    if target.digest != delta.target_digest:
        raise ContextPackError("target digest mismatch after reconstruction")

    return target


def _compute_digest(items: tuple[bytes, ...]) -> str:
    """Compute SHA-256 digest for pack items."""
    # Digest is computed on JSON with only schema and items fields
    unsigned = {
        "schema": PACK_SCHEMA,
        "items": [base64.b64encode(item).decode("ascii") for item in items],
    }
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


__all__ = [
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
