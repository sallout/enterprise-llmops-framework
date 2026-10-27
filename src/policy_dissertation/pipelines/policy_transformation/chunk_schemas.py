"""Traceable normalization, quality, and chunk artifacts (not training examples)."""
from typing import Literal

from pydantic import BaseModel, Field

ALGORITHM_VERSION = "structure-v1"
TOKEN_METHOD = "utf8_bytes_upper_bound_v1"


class SourceSpan(BaseModel):
    """Half-open Python character offsets into a page's original extracted text.

    With page_number=None, offsets reference document.extracted_content instead.
    """
    page_number: int | None = None
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class QualityWarning(BaseModel):
    warning_id: str
    code: str
    message: str
    resolution: Literal["corrected", "unresolved", "informational"] = "unresolved"
    source_spans: list[SourceSpan] = Field(default_factory=list)


class NormalizedLine(BaseModel):
    line_id: str
    source_span: SourceSpan
    original_text: str
    normalized_text: str
    excluded: bool = False
    is_code: bool = False
    join_before: Literal["\n", "\n\n", " "] = "\n"
    actions: list[str] = Field(default_factory=list)


class NormalizedPage(BaseModel):
    page_number: int | None = None
    original_content: str
    normalized_content: str
    lines: list[NormalizedLine]


class NormalizationReport(BaseModel):
    algorithm_version: str = ALGORITHM_VERSION
    document_id: str
    source_content_hash: str
    source_path: str
    original_extracted_content: str
    normalized_content: str
    pages: list[NormalizedPage]
    invariant_check: Literal["passed"] = "passed"
    normalization_status: Literal["unchanged", "normalized", "review_required"]


class QualityReport(BaseModel):
    document_id: str
    source_content_hash: str
    algorithm_version: str = ALGORITHM_VERSION
    warnings: list[QualityWarning]


class SectionContext(BaseModel):
    section_id: str | None = None
    title: str
    level: int
    heading: str
    source_span: SourceSpan


class PolicyChunk(BaseModel):
    schema_version: str = "1.0"
    algorithm_version: str = ALGORITHM_VERSION
    chunk_id: str
    document_id: str
    source_content_hash: str
    source_path: str
    section_id: str | None
    section_title: str
    parent_sections: list[SectionContext]
    section_heading: str | None = None
    section_source_spans: list[SourceSpan] = Field(default_factory=list)
    page_numbers: list[int]
    content: str
    token_count: int = Field(ge=0)
    token_method: str = TOKEN_METHOD
    max_tokens: int = Field(gt=0)
    oversized: bool = False
    line_ids: list[str]
    source_spans: list[SourceSpan]
    normalization_status: Literal["unchanged", "normalized", "review_required"]
    quality_warnings: list[QualityWarning]


class ChunkDocumentReference(BaseModel):
    document_id: str
    source_artifact: str
    chunks_path: str
    normalization_path: str
    quality_path: str
    chunk_count: int


class ChunkBatchSummary(BaseModel):
    algorithm_version: str = ALGORITHM_VERSION
    token_method: str = TOKEN_METHOD
    max_tokens: int
    input_directory: str
    output_directory: str
    documents_seen: int = 0
    documents_processed: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunk_count: int = 0
    oversized_chunks: int = 0
    quality_issue_count: int = 0
    corrected_issue_count: int = 0
    unresolved_issue_count: int = 0
    transformation_counts: dict[str, int] = Field(default_factory=dict)
    warnings_by_code: dict[str, int] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    documents: list[ChunkDocumentReference] = Field(default_factory=list)
