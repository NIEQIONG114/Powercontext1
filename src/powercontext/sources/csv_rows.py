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

"""CSV row-level Source Definition for table evidence capture."""

from __future__ import annotations

import csv
import io
import json
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.adapters import SourceAdapter
from powercontext.sources.definitions import AdapterSourceDefinition, SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence


class CSVDocumentInput(BaseModel):
    """Input for parsing a complete CSV document into row-level inputs."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """A single row from a CSV document with identity components."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """Captured Source for a single CSV row."""

    model_config = ConfigDict(frozen=True)

    source_type: str = "csv-row"
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @property
    def source_id(self) -> str:
        return csv_row_source_id(self.document_id, self.primary_key_value)

    @property
    def name(self) -> str:
        return self.source_id


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Compute the stable source identity for a CSV row.

    Format: csv:<url-encoded-document-id>:<url-encoded-primary-key-value>
    """
    if document_id != document_id.strip():
        raise ValueError("document_id must not have leading or trailing whitespace")
    if not document_id:
        raise ValueError("document_id must be non-empty")

    encoded_doc = quote(document_id, safe="")
    encoded_key = quote(primary_key_value, safe="")
    identity = f"csv:{encoded_doc}:{encoded_key}"

    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")

    return identity


def parse_csv_document(input: CSVDocumentInput, /) -> tuple[CSVRowInput, ...]:
    """Parse a CSV document into row-level inputs.

    - Strict UTF-8 decoding with optional BOM
    - First record is header; fields must be non-empty, non-blank, unique
    - Primary key must exist in header, values must be non-empty and unique
    - Empty trailing records are ignored
    - All validation errors raise ValueError atomically
    """
    document_id = input.document_id
    primary_key = input.primary_key
    content = input.content

    if document_id != document_id.strip():
        raise ValueError("document_id must not have leading or trailing whitespace")
    if not document_id:
        raise ValueError("document_id must be non-empty")

    if not primary_key:
        raise ValueError("primary_key must be non-empty")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 encoding: {error}") from error

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)

    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("empty CSV document")

    if not header:
        raise ValueError("empty CSV header")

    for field in header:
        if not field or not field.strip():
            raise ValueError(f"header field must be non-empty and non-blank, got {field!r}")

    if len(header) != len(set(header)):
        raise ValueError("header fields must be unique")

    if primary_key not in header:
        raise ValueError(f"primary_key {primary_key!r} not found in header")

    primary_key_index = header.index(primary_key)
    columns = tuple(header)

    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()

    try:
        for record in reader:
            if not record or all(field == "" for field in record):
                continue

            if len(record) != len(header):
                raise ValueError(
                    f"record width {len(record)} does not match header width {len(header)}"
                )

            primary_key_value = record[primary_key_index]
            if not primary_key_value or not primary_key_value.strip():
                raise ValueError("primary key value must be non-empty")

            if primary_key_value in seen_keys:
                raise ValueError(f"duplicate primary key value: {primary_key_value!r}")
            seen_keys.add(primary_key_value)

            rows.append(
                CSVRowInput(
                    document_id=document_id,
                    primary_key=primary_key,
                    primary_key_value=primary_key_value,
                    columns=columns,
                    values=tuple(record),
                )
            )
    except csv.Error as error:
        raise ValueError(f"CSV parsing error: {error}") from error

    return tuple(rows)


class CSVRowTextEvidenceProjection:
    """Project CSVRowSource to TextEvidence for memory retrieval."""

    name = "powercontext.text-evidence"
    version = "1"
    source_class = CSVRowSource
    output_class = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row_mapping = {column: value for column, value in zip(source.columns, source.values)}
        content = json.dumps(row_mapping, ensure_ascii=False, separators=(",", ":"))

        return TextEvidence(
            source_type="csv-row",
            source_id=source.source_id,
            content=content,
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row_mapping,
            },
        )


class CSVRowSourceDefinition(AdapterSourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput]):
    """Source Definition for csv-row sources."""

    def __init__(self) -> None:
        super().__init__(
            adapter=_CSVRowAdapter(),
            version="1",
            projections=(CSVRowTextEvidenceProjection(),),
        )


class _CSVRowAdapter:
    """Internal adapter for CSVRowInput -> CSVRowSource."""

    input_class = CSVRowInput
    name = "csv-row"
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        return CSVRowSource(
            name=csv_row_source_id(value.document_id, value.primary_key_value),
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


CSV_ROW_SOURCE_DEFINITION = CSVRowSourceDefinition()

__all__ = [
    "CSV_ROW_SOURCE_DEFINITION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "csv_row_source_id",
    "parse_csv_document",
]
