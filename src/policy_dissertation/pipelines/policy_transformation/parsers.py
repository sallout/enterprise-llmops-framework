"""Local text extraction without OCR, model inference, or semantic chunking."""
from importlib.metadata import version

import pymupdf
import pymupdf4llm

from .schemas import ExtractionMetadata, PageContent

SUPPORTED_TYPES = {".pdf": "pdf", ".md": "markdown", ".markdown": "markdown", ".txt": "txt"}


def parse_text(data: bytes, source_type: str) -> tuple[str, ExtractionMetadata]:
    # Strict decoding avoids silently replacing source characters. Preserve newlines.
    text = data.decode("utf-8-sig")
    if "\x00" in text:
        raise ValueError("NUL characters found: input is not supported UTF-8 text.")
    return text, ExtractionMetadata(
        parser="utf-8", encoding="utf-8-sig",
        content_format="markdown" if source_type == "markdown" else "text",
    )


def parse_pdf(data: bytes) -> tuple[str, ExtractionMetadata]:
    metadata = ExtractionMetadata(
        parser="pymupdf4llm", content_format="markdown",
        parser_versions={name: version(name) for name in ("pymupdf", "pymupdf4llm")},
        warnings=["PDF layout, headings, lists, tables, and reading order are heuristic; glyph decoding and table whitespace may be inaccurate; OCR is disabled."],
    )
    with pymupdf.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("Password-protected PDF requires authentication.")
        metadata.page_count = len(document)
        if document.is_repaired:
            metadata.warnings.append("PyMuPDF repaired the PDF while opening it.")
        # One page at a time gives reliable physical page references and isolates failures.
        headers = pymupdf4llm.IdentifyHeaders(document)
        for index in range(len(document)):
            warnings = []
            raw_text = ""
            try:
                raw_text = document[index].get_text(sort=False)
                text = pymupdf4llm.to_markdown(
                    document, pages=[index], hdr_info=headers, show_progress=False,
                    write_images=False, embed_images=False,
                )
                if not text.strip():
                    warnings.append("No extractable text; page may be blank or require OCR.")
            except Exception as exc:
                text = ""
                warnings.append(f"Page extraction failed ({type(exc).__name__}).")
            metadata.pages.append(PageContent(
                page_number=index + 1, extracted_content=text, raw_text=raw_text, warnings=warnings,
            ))
        if any(page.warnings for page in metadata.pages):
            metadata.warnings.append("One or more pages have extraction warnings; review page records.")
        return "\n\n".join(page.extracted_content for page in metadata.pages), metadata
