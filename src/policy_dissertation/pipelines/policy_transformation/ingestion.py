"""Recursive discovery and on-disk artifacts. Batch results carry references only."""
import hashlib
import os
from pathlib import Path
import tempfile

from pydantic import BaseModel

from policy_dissertation.errors import DissertationError
from .parsers import SUPPORTED_TYPES, parse_pdf, parse_text
from .schemas import ArtifactReference, DocumentArtifact, IngestionSummary


def document_id(path: Path) -> str:
    """Stable identity for an absolute source location, independent of its bytes."""
    return hashlib.sha256(Path(os.path.abspath(path)).as_uri().encode("utf-8")).hexdigest()


def parse_document(path: Path) -> DocumentArtifact:
    path = Path(os.path.abspath(path))
    source_type = SUPPORTED_TYPES.get(path.suffix.lower(), "unsupported")
    artifact = DocumentArtifact(
        document_id=document_id(path), source_path=str(path), source_type=source_type,
        processing_status="unsupported" if source_type == "unsupported" else "failed",
    )
    if path.is_symlink():
        artifact.processing_status = "unsupported"
        artifact.extraction_metadata.warnings.append("Symbolic links are not followed.")
        return artifact
    if source_type == "unsupported":
        artifact.extraction_metadata.warnings.append(f"Unsupported file extension: {path.suffix or '(none)' }.")
        return artifact
    try:
        data = path.read_bytes()
        artifact.content_hash = hashlib.sha256(data).hexdigest()
        if not data:
            raise ValueError("Source file is empty.")
        content, metadata = parse_pdf(data) if source_type == "pdf" else parse_text(data, source_type)
        artifact.extraction_metadata = metadata
        artifact.extracted_content = content
        if not content.strip():
            raise ValueError("Document contains no extractable text.")
        if any(p.warnings for p in metadata.pages):
            raise ValueError("Incomplete PDF extraction; review page warnings and retained content.")
        artifact.processing_status = "parsed"
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        artifact.extraction_metadata.error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        # Isolate parser-library failures without stopping unrelated documents.
        artifact.extraction_metadata.error = f"Unexpected parser failure ({type(exc).__name__})."
    return artifact


def write_json(path: Path, model: BaseModel) -> None:
    """Replace a complete UTF-8 JSON artifact atomically, cleaning failed writes."""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(model.model_dump_json(indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ingest_documents(input_directory: Path, output_directory: Path,
                     limit_documents: int | None = None) -> IngestionSummary:
    if limit_documents is not None and limit_documents < 1:
        raise DissertationError('limit_documents must be positive when supplied.')
    source = input_directory.resolve()
    output = output_directory.resolve()
    if not source.is_dir():
        raise DissertationError(f"Input directory does not exist or is not a directory: {source}")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise DissertationError("Input and output directories must not overlap.")
    summary = IngestionSummary(input_directory=str(source), output_directory=str(output),
                               limit_documents=limit_documents)

    def traversal_error(error: OSError) -> None:
        summary.discovery_warnings.append(f"Cannot scan {error.filename}: {error.strerror}")

    # Sorting both levels makes discovery and manifest order repeatable.
    for root, directories, filenames in os.walk(source, onerror=traversal_error, followlinks=False):
        directories.sort()
        for directory in list(directories):
            path = Path(root) / directory
            if path.is_symlink():
                summary.discovery_warnings.append(f"Skipped symbolic-link directory: {path}")
                directories.remove(directory)
        for filename in sorted(filenames):
            path = Path(root) / filename
            summary.discovered += 1
            supported = path.suffix.lower() in SUPPORTED_TYPES and not path.is_symlink()
            if supported:
                summary.supported_discovered += 1
                if limit_documents is not None and summary.selected_documents >= limit_documents:
                    summary.excluded_by_limit += 1
                    summary.excluded_source_paths.append(str(path))
                    continue
                # Failed/empty supported documents consume a slot; never hide failures
                # by continuing until N successful documents have been found.
                summary.selected_documents += 1
                summary.selected_source_paths.append(str(path))
            artifact = parse_document(path)
            relative = Path("documents") / f"{artifact.document_id}.json"
            try:
                write_json(output / relative, artifact)
            except OSError as exc:
                raise DissertationError(f"Cannot write output artifact: {exc}") from exc
            field = artifact.processing_status
            setattr(summary, field, getattr(summary, field) + 1)
            summary.artifacts.append(ArtifactReference(
                document_id=artifact.document_id, source_path=artifact.source_path,
                processing_status=field, artifact_path=relative.as_posix(),
            ))
    try:
        write_json(output / "summary.json", summary)
    except OSError as exc:
        raise DissertationError(f"Cannot write summary: {exc}") from exc
    return summary
