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

"""Behavior tests for the PDF page-level Source."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from powercontext.artifacts import ArtifactRef
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.pdf_pages import (
    PDFDocumentInput,
    parse_pdf_document,
    pdf_page_source_id,
    search_pdf_pages,
)

DIGEST = "a" * 64


def _write(writer: PdfWriter) -> bytes:
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _document(pages: int = 2) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=200)
    return _write(writer)


def test_parse_keeps_every_physical_page_in_order() -> None:
    pages = parse_pdf_document(value=PDFDocumentInput(document_id="report", content=_document()))

    assert [page.page_number for page in pages] == [1, 2]
    assert [page.text for page in pages] == ["", ""]
    assert {page.document_id for page in pages} == {"report"}
    assert len({page.document_sha256 for page in pages}) == 1


def test_document_without_pages_fails_instead_of_returning_nothing() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=200)
    writer.remove_page(0)

    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="empty-tree", content=_write(writer)))


def test_document_that_needs_a_user_password_fails() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=200)
    writer.encrypt(user_password="secret", owner_password="owner")  # noqa: S106

    with pytest.raises(ValueError):
        parse_pdf_document(PDFDocumentInput(document_id="locked", content=_write(writer)))


def test_page_identity_stays_bounded_and_distinct() -> None:
    long_id = "财报/" + "x" * 4_000
    other_long_id = "财报/" + "y" * 4_000
    identity = pdf_page_source_id(long_id, DIGEST, 1)

    assert len(identity) <= MAX_SOURCE_ID_LENGTH
    assert identity == pdf_page_source_id(long_id, DIGEST, 1)
    assert identity != pdf_page_source_id(other_long_id, DIGEST, 1)
    assert pdf_page_source_id("report", DIGEST, 1) != pdf_page_source_id("report", DIGEST, 2)
    assert pdf_page_source_id("report", DIGEST, 1) != pdf_page_source_id("report", "b" * 64, 1)


def test_search_asks_the_scope_memory_for_exact_citations() -> None:
    ref = ArtifactRef(family="memory", artifact_id="memory", revision=1)
    hits = [
        SimpleNamespace(memory_ref=ref, entry_id="entry-1", entry_version_id="version-1"),
        SimpleNamespace(memory_ref=ref, entry_id="entry-1", entry_version_id="version-1"),
        SimpleNamespace(memory_ref=ref, entry_id="entry-2", entry_version_id="version-2"),
        SimpleNamespace(memory_ref=ref, entry_id="entry-3", entry_version_id="version-3"),
    ]
    asked: list[object] = []

    class _Memory:
        async def search(self, request, /):
            asked.append(request)
            return SimpleNamespace(hits=hits)

    async def scenario() -> None:
        citations = await search_pdf_pages(_Memory(), "what was the revenue?", limit=2)

        assert citations
        assert asked[0].query == "what was the revenue?"
        assert asked[0].limit == 2
        assert asked[0].mode == "fts"
        assert [citation.entry_id for citation in citations] == ["entry-1", "entry-2"]

    asyncio.run(scenario())
