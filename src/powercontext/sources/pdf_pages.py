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

"""PDF page-level Source Definition for financial report retrieval."""

from __future__ import annotations

import hashlib
import io
from typing import Any, Protocol
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.adapters import SourceAdapter
from powercontext.sources.definitions import AdapterSourceDefinition, SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence


class PDFDocumentInput(BaseModel):
    """Input for parsing a complete PDF document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """A single page from a PDF document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured Source for a single PDF page."""

    model_config = ConfigDict(frozen=True)

    source_type: str = "pdf-page"
    document_id: str
    document_sha256: str
    page_number: int
    text: str

    @property
    def source_id(self) -> str:
        return pdf_page_source_id(self.document_id, self.document_sha256, self.page_number)

    @property
    def name(self) -> str:
        return self.source_id


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Compute the stable source identity for a PDF page.

    Format: pdf:<url-encoded-doc-id>:<sha256-prefix>:<page-number>
    """
    encoded_doc = quote(document_id, safe="")
    sha_prefix = document_sha256[:16]
    identity = f"pdf:{encoded_doc}:{sha_prefix}:{page_number}"

    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id must not exceed {MAX_SOURCE_ID_LENGTH} characters")

    return identity


def parse_pdf_document(input: PDFDocumentInput, /) -> tuple[PDFPageInput, ...]:
    """Parse a PDF document and return all physical pages.

    - Pages are numbered from 1
    - Blank pages are included with empty text
    - Document SHA-256 is computed from the full input bytes
    - Invalid/corrupt PDFs raise ValueError atomically
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    document_id = input.document_id
    content = input.content

    if not document_id or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")

    if not content:
        raise ValueError("empty PDF content")

    if not content.startswith(b"%PDF"):
        raise ValueError("not a valid PDF file")

    document_sha256 = hashlib.sha256(content).hexdigest()

    try:
        reader = PdfReader(io.BytesIO(content))

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                pass

        pages: list[PDFPageInput] = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=page_num,
                    text=text,
                )
            )

        return tuple(pages)

    except PdfReadError as error:
        raise ValueError(f"invalid PDF: {error}") from error
    except Exception as error:
        raise ValueError(f"PDF parsing error: {error}") from error


class PDFPageTextEvidenceProjection:
    """Project PDFPageSource to TextEvidence for memory retrieval."""

    name = "powercontext.text-evidence"
    version = "1"
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        content = f"{source.document_id}\n{source.text}"
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.source_id,
            content=content,
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class _PDFPageAdapter:
    """Internal adapter for PDFPageInput -> PDFPageSource."""

    input_class = PDFPageInput
    name = "pdf-page"
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        return PDFPageSource(
            name=pdf_page_source_id(value.document_id, value.document_sha256, value.page_number),
            definition_version="1",
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=_PDFPageAdapter(),
    version="1",
    projections=(PDFPageTextEvidenceProjection(),),
)


class _MemoryProtocol(Protocol):
    """Protocol for the scoped Memory interface."""

    async def search(self, request: Any, /) -> Any: ...


async def search_pdf_pages(
    memory: _MemoryProtocol,
    question: str,
    limit: int = 10,
) -> tuple[Any, ...]:
    """Search PDF pages using the scoped Memory API.

    Returns up to `limit` distinct MemoryCitation objects for pages
    matching the natural language question.
    """
    from powercontext.builtin.artifacts.memory import SearchMemoryRequest

    request = SearchMemoryRequest(
        query=question,
        mode="fts",
        limit=limit,
    )

    result = await memory.search(request)

    citations = []
    seen = set()

    for hit in result.hits:
        from powercontext.builtin.artifacts.memory.models import MemoryCitation

        citation = MemoryCitation(
            memory_ref=hit.memory_ref,
            entry_id=hit.entry_id,
            entry_version_id=hit.entry_version_id,
        )

        citation_key = (hit.memory_ref, hit.entry_id, hit.entry_version_id)
        if citation_key not in seen:
            seen.add(citation_key)
            citations.append(citation)

        if len(citations) >= limit:
            break

    return tuple(citations)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
