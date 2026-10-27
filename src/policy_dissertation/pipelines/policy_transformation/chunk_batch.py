"""Consume the ingestion manifest; write JSONL chunks and separate audit reports."""
from collections import Counter
import hashlib
from pathlib import Path
import tempfile

from policy_dissertation.errors import DissertationError
from .chunking import build_chunks
from .chunk_schemas import ChunkBatchSummary, ChunkDocumentReference, PolicyChunk
from .ingestion import write_json
from .normalization import assess_and_normalize
from .schemas import DocumentArtifact, IngestionSummary


def write_jsonl(path: Path, chunks: list[PolicyChunk]) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for chunk in chunks:
                stream.write(chunk.model_dump_json() + '\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def chunk_documents(input_directory: Path, output_directory: Path, max_tokens: int = 2048) -> ChunkBatchSummary:
    source, output = input_directory.resolve(), output_directory.resolve()
    if not source.is_dir():
        raise DissertationError(f'Parsed input directory does not exist: {source}')
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise DissertationError('Input and output directories must not overlap.')
    if max_tokens < 1:
        raise DissertationError('max_tokens must be positive.')
    try:
        manifest = IngestionSummary.model_validate_json((source / 'summary.json').read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise DissertationError(f'Cannot load ingestion summary.json ({type(exc).__name__}).') from exc
    summary = ChunkBatchSummary(input_directory=str(source), output_directory=str(output), max_tokens=max_tokens)
    warning_counts, transformation_counts = Counter(), Counter()
    seen = set()
    for reference in manifest.artifacts:
        summary.documents_seen += 1
        if reference.processing_status != 'parsed':
            summary.documents_skipped += 1
            continue
        try:
            artifact_path = (source / reference.artifact_path).resolve()
            if not artifact_path.is_relative_to(source):
                raise DissertationError('Source artifact reference escapes input directory.')
            document = DocumentArtifact.model_validate_json(artifact_path.read_text(encoding='utf-8'))
            if document.document_id != reference.document_id or document.source_path != reference.source_path:
                raise DissertationError('Manifest identity/source mismatch.')
            if document.document_id in seen:
                raise DissertationError('Duplicate document identity in ingestion manifest.')
            seen.add(document.document_id)
            report, quality = assess_and_normalize(document)
            chunks = build_chunks(report, quality, max_tokens)
            if not chunks:
                raise DissertationError('No chunkable body content found.')
            # Identity is untrusted external JSON: use a hash for safe output names.
            stem = hashlib.sha256(document.document_id.encode()).hexdigest()
            chunk_path = f'documents/{stem}.jsonl'
            norm_path = f'normalization/{stem}.json'
            quality_path = f'quality/{stem}.json'
        except (ValueError, OSError, DissertationError) as exc:
            summary.documents_failed += 1
            summary.errors.append(f'{reference.document_id}: {exc}')
            continue
        # Output errors are fatal: do not publish a misleading complete summary.
        try:
            write_json(output / norm_path, report)
            write_json(output / quality_path, quality)
            write_jsonl(output / chunk_path, chunks)
        except OSError as exc:
            raise DissertationError(f'Cannot write chunk artifacts: {exc}') from exc
        summary.documents_processed += 1
        summary.chunk_count += len(chunks)
        summary.oversized_chunks += sum(chunk.oversized for chunk in chunks)
        summary.quality_issue_count += len(quality.warnings)
        summary.corrected_issue_count += sum(w.resolution == 'corrected' for w in quality.warnings)
        summary.unresolved_issue_count += sum(w.resolution == 'unresolved' for w in quality.warnings)
        warning_counts.update(w.code for w in quality.warnings)
        transformation_counts.update(action for page in report.pages for line in page.lines for action in line.actions)
        summary.documents.append(ChunkDocumentReference(
            document_id=document.document_id, source_artifact=reference.artifact_path,
            chunks_path=chunk_path, normalization_path=norm_path, quality_path=quality_path,
            chunk_count=len(chunks),
        ))
    summary.warnings_by_code = dict(sorted(warning_counts.items()))
    summary.transformation_counts = dict(sorted(transformation_counts.items()))
    try:
        write_json(output / 'summary.json', summary)
    except OSError as exc:
        raise DissertationError(f'Cannot write chunk summary: {exc}') from exc
    return summary
