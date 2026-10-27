"""Versioned ingestion artifacts; no training-example schemas are defined here."""
from typing import Literal

from pydantic import BaseModel, Field


class PageContent(BaseModel):
    page_number: int = Field(ge=1)
    extracted_content: str
    raw_text: str = ""
    warnings: list[str] = Field(default_factory=list)


class ExtractionMetadata(BaseModel):
    parser: str = "none"
    parser_versions: dict[str, str] = Field(default_factory=dict)
    content_format: Literal["markdown", "text"] = "text"
    encoding: str | None = None
    page_count: int | None = None
    pages: list[PageContent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class DocumentArtifact(BaseModel):
    schema_version: str = "1.0"
    document_id: str
    source_path: str
    source_type: str
    content_hash: str | None = None
    extracted_content: str = ""
    extraction_metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)
    processing_status: Literal["parsed", "failed", "unsupported"]


class ArtifactReference(BaseModel):
    document_id: str
    source_path: str
    processing_status: Literal["parsed", "failed", "unsupported"]
    artifact_path: str


class IngestionSummary(BaseModel):
    input_directory: str
    output_directory: str
    discovered: int = 0
    supported_discovered: int = 0
    selected_documents: int = 0
    excluded_by_limit: int = 0
    limit_documents: int | None = Field(default=None, ge=1)
    selected_source_paths: list[str] = Field(default_factory=list)
    excluded_source_paths: list[str] = Field(default_factory=list)
    parsed: int = 0
    failed: int = 0
    unsupported: int = 0
    discovery_warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
