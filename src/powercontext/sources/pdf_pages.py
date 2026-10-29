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
from typing import Any, NoReturn, Protocol
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TextEvidence

#: Room left for the revision prefix and the page number inside one page identity.
_DOCUMENT_ID_BUDGET = MAX_SOURCE_ID_LENGTH - 64


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


class _PDFInputError(ValueError):
    """An input from which a complete page set cannot be produced."""


def _reject(message: str) -> NoReturn:
    raise _PDFInputError(message)


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the identity of one page of one document revision.

    The identity is `(document id, document revision, physical page)`, percent-encoded so no
    component can impersonate a separator. A document id longer than the budget travels as its
    own digest, which keeps the identity stable and inside the platform limit for any input.
    """

    encoded = quote(document_id, safe="")
    if len(encoded) > _DOCUMENT_ID_BUDGET:
        encoded = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
    identity = f"pdf:{encoded}:{document_sha256[:16]}:{page_number}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        _reject("page identity must stay within the source id limit")
    return identity


def _read_pages(content: bytes) -> list[tuple[int, str]]:
    """Return ``(physical page number, extracted text)`` for every page of the document."""

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            _reject("PDF needs a user password before its pages can be read")
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
    except _PDFInputError:
        raise
    except Exception as error:
        message = f"PDF could not be parsed completely: {error}"
        raise _PDFInputError(message) from error


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse a complete PDF into every physical page, in document order.

    Page numbers start at one and blank pages stay in the sequence with empty text. A document
    whose pages cannot all be read fails as a whole, so a caller never receives a partial page
    list.
    """

    document_id = value.document_id
    content = value.content

    if not document_id.strip():
        _reject("document_id must not be empty or whitespace")
    if not content:
        _reject("PDF content must not be empty")
    if not content.startswith(b"%PDF"):
        _reject("content is not a PDF file")

    pages = _read_pages(content)
    if not pages:
        _reject("PDF has no pages")

    document_sha256 = hashlib.sha256(content).hexdigest()
    return tuple(
        PDFPageInput(
            document_id=document_id,
            document_sha256=document_sha256,
            page_number=page_number,
            text=text,
        )
        for page_number, text in pages
    )


class PDFPageTextEvidenceProjection:
    """Project a PDF page Source onto the standard text-evidence view."""

    name = "powercontext.text-evidence"
    version = "1"
    source_class = PDFPageSource
    output_class = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class _PDFPageAdapter:
    """Turn one PDF page input into its captured Source, and back again."""

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


class _ScopedMemory(Protocol):
    """The public scoped Memory surface the page search is allowed to use."""

    async def search(self, request: Any, /) -> Any: ...


async def search_pdf_pages(
    memory: _ScopedMemory,
    question: str,
    limit: int = 10,
) -> tuple[Any, ...]:
    """Return exact citations for up to ``limit`` pages that answer a natural-language question.

    Retrieval goes through the public Memory search of the caller's scope, so every citation
    points at stored evidence; the offline text index is the only ranking signal available
    without a model.
    """

    from powercontext.builtin.artifacts.memory.models import MemoryCitation
    from powercontext.builtin.runtime.models import SearchMemoryRequest

    page = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=limit))

    citations: list[Any] = []
    seen: set[tuple[str, str, str]] = set()
    for hit in page.hits:
        if len(citations) >= limit:
            break
        key = (hit.memory_ref.artifact_id, hit.entry_id, hit.entry_version_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
        )
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
