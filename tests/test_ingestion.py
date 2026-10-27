import hashlib
import json
from pathlib import Path

import pymupdf
import pytest
from typer.testing import CliRunner

from policy_dissertation.cli import app
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.ingestion import (
    document_id, ingest_documents, parse_document,
)
from policy_dissertation.pipelines.policy_transformation.schemas import DocumentArtifact


@pytest.fixture
def pdf_file(tmp_path):
    path = tmp_path / 'policy.pdf'
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), 'Security Policy', fontsize=20)
        page.insert_text((72, 110), 'Employees must protect corporate information.', fontsize=11)
        page.insert_text((72, 140), '- Review permissions regularly.', fontsize=11)
        page = document.new_page()
        page.insert_text((72, 72), 'Incident reporting', fontsize=20)
        page.insert_text((72, 110), 'Report incidents promptly.', fontsize=11)
        document.save(path)
    return path


def test_pdf_parsing_and_page_references(pdf_file):
    before = pdf_file.read_bytes()
    artifact = parse_document(pdf_file)
    assert artifact.processing_status == 'parsed', artifact.extraction_metadata
    assert artifact.source_path == str(pdf_file.resolve())
    assert artifact.source_type == 'pdf'
    assert artifact.content_hash == hashlib.sha256(before).hexdigest()
    assert [p.page_number for p in artifact.extraction_metadata.pages] == [1, 2]
    assert 'Security Policy' in artifact.extraction_metadata.pages[0].extracted_content
    assert 'Report incidents promptly.' in artifact.extraction_metadata.pages[1].extracted_content
    assert 'Security Policy' in artifact.extraction_metadata.pages[0].raw_text
    assert '# ' in artifact.extracted_content
    assert '- Review permissions regularly.' in artifact.extracted_content
    assert pdf_file.read_bytes() == before


@pytest.mark.parametrize('suffix', ['.md', '.markdown', '.TXT'])
def test_text_structure(tmp_path, suffix):
    path = tmp_path / ('policy' + suffix)
    content = '# Heading\r\n\r\nParagraph café.\r\n\r\n- First\r\n- Second\r\n\r\n| A | B |\r\n|---|---|\r\n| 1 | 2 |\r\n'
    path.write_bytes(content.encode('utf-8'))
    artifact = parse_document(path)
    assert artifact.processing_status == 'parsed'
    assert artifact.extracted_content == content
    assert artifact.source_path == str(path.resolve())
    assert artifact.extraction_metadata.pages == []
    assert artifact.extraction_metadata.page_count is None


@pytest.mark.parametrize('name,content', [
    ('empty.txt', b''), ('whitespace.md', b' \n'), ('corrupt.pdf', b'not a pdf'),
    ('invalid.txt', b'\xff'), ('binary.txt', b'a\x00b'),
])
def test_invalid_files(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content)
    artifact = parse_document(path)
    assert artifact.processing_status == 'failed'
    assert artifact.extraction_metadata.error
    assert artifact.content_hash == hashlib.sha256(content).hexdigest()


def test_unsupported_file(tmp_path):
    path = tmp_path / 'policy.docx'
    path.write_bytes(b'unsupported')
    artifact = parse_document(path)
    assert artifact.processing_status == 'unsupported'
    assert artifact.extraction_metadata.warnings
    assert artifact.content_hash is None  # unsupported files are never read


def test_unreadable_file(tmp_path, monkeypatch):
    path = tmp_path / 'unreadable.txt'
    path.write_text('text')
    original = Path.read_bytes

    def read_bytes(self):
        if self == path:
            raise PermissionError('Permission denied')
        return original(self)

    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    artifact = parse_document(path)
    assert artifact.processing_status == 'failed'
    assert artifact.content_hash is None
    assert 'PermissionError' in artifact.extraction_metadata.error


def test_deterministic_identity_and_content(tmp_path):
    path = tmp_path / 'policy.txt'
    path.write_text('first')
    first = parse_document(path)
    assert first == parse_document(path)
    assert first.document_id == document_id(path)
    assert document_id(path) == document_id(path.parent / 'unused' / '..' / path.name)
    path.write_text('second')
    second = parse_document(path)
    assert second.document_id == first.document_id
    assert second.content_hash != first.content_hash
    other = tmp_path / 'other.txt'
    other.write_bytes(path.read_bytes())
    assert parse_document(other).document_id != second.document_id
    assert parse_document(other).content_hash == second.content_hash


def test_recursive_statistics_and_artifacts(tmp_path):
    source = tmp_path / 'input'
    (source / 'nested').mkdir(parents=True)
    (source / 'nested' / 'ok.md').write_text('# Café\n\n- Rule\n')
    (source / 'ok.txt').write_text('Policy')
    (source / 'empty.txt').touch()
    (source / 'bad.pdf').write_bytes(b'broken')
    (source / 'image.png').write_bytes(b'unknown')
    output = tmp_path / 'output'
    summary = ingest_documents(source, output)
    assert (summary.discovered, summary.parsed, summary.failed, summary.unsupported) == (5, 2, 2, 1)
    assert summary.discovered == summary.parsed + summary.failed + summary.unsupported
    assert len(summary.artifacts) == 5
    assert len(list((output / 'documents').glob('*.json'))) == 5
    assert 'extracted_content' not in summary.model_dump_json()
    for reference in summary.artifacts:
        artifact = DocumentArtifact.model_validate_json((output / reference.artifact_path).read_text())
        assert artifact.document_id == reference.document_id
        assert artifact.source_path == reference.source_path
    assert json.loads((output / 'summary.json').read_text()) == summary.model_dump()
    assert summary == ingest_documents(source, output)


def test_cli_success(tmp_path, monkeypatch):
    monkeypatch.setenv('LOG_LEVEL', 'INFO')
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'policy.txt').write_text('Corporate policy')
    result = CliRunner().invoke(app, ['policies', 'parse', '--input', str(source), '--output', str(tmp_path / 'output')])
    assert result.exit_code == 0, result.output
    assert 'Documents discovered: 1' in result.output
    assert 'Successfully parsed: 1' in result.output
    assert 'Failed: 0' in result.output
    assert 'Unsupported: 0' in result.output


def test_cli_partial_failure_continues(tmp_path, monkeypatch):
    monkeypatch.setenv('LOG_LEVEL', 'INFO')
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'a.txt').touch()
    (source / 'z.txt').write_text('valid')
    result = CliRunner().invoke(app, ['policies', 'parse', '--input', str(source), '--output', str(tmp_path / 'output')])
    assert result.exit_code == 1
    assert 'Successfully parsed: 1' in result.output
    assert 'Failed: 1' in result.output
    assert (tmp_path / 'output' / 'summary.json').exists()


def test_input_output_guards(tmp_path):
    with pytest.raises(DissertationError):
        ingest_documents(tmp_path / 'missing', tmp_path / 'output')
    with pytest.raises(DissertationError):
        ingest_documents(tmp_path, tmp_path / 'nested-output')


def test_blank_pdf(tmp_path):
    path = tmp_path / 'blank.pdf'
    with pymupdf.open() as document:
        document.new_page()
        document.save(path)
    artifact = parse_document(path)
    assert artifact.processing_status == 'failed'
    assert artifact.extraction_metadata.pages[0].warnings


def test_symlinks_not_followed(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    target = tmp_path / 'outside.txt'
    target.write_text('do not read')
    (source / 'link.txt').symlink_to(target)
    summary = ingest_documents(source, tmp_path / 'output')
    assert summary.unsupported == 1


def test_page_failure_retains_content(pdf_file, monkeypatch):
    from policy_dissertation.pipelines.policy_transformation import parsers
    original = parsers.pymupdf4llm.to_markdown

    def extract(document, **kwargs):
        if kwargs['pages'] == [1]:
            raise RuntimeError('broken page')
        return original(document, **kwargs)

    monkeypatch.setattr(parsers.pymupdf4llm, 'to_markdown', extract)
    artifact = parse_document(pdf_file)
    assert artifact.processing_status == 'failed'
    assert 'Security Policy' in artifact.extracted_content
    assert artifact.extraction_metadata.pages[1].warnings


def test_pdf_table(tmp_path):
    path = tmp_path / 'table.pdf'
    with pymupdf.open() as document:
        page = document.new_page()
        for x in (72, 220, 368):
            page.draw_line((x, 100), (x, 180))
        for y in (100, 140, 180):
            page.draw_line((72, y), (368, y))
        for point, text in (((80, 125), 'Control'), ((230, 125), 'Owner'),
                            ((80, 165), 'Access'), ((230, 165), 'Security')):
            page.insert_text(point, text)
        document.save(path)
    artifact = parse_document(path)
    assert artifact.processing_status == 'parsed'
    assert '|---' in artifact.extracted_content
    assert 'Access' in artifact.extracted_content
    assert 'Security' in artifact.extracted_content


def test_encrypted_pdf(tmp_path):
    path = tmp_path / 'encrypted.pdf'
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), 'Restricted policy')
        document.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256,
                      owner_pw='owner-test', user_pw='user-test')
    artifact = parse_document(path)
    assert artifact.processing_status == 'failed'
    assert 'Password-protected' in artifact.extraction_metadata.error


def test_output_write_failure(tmp_path, monkeypatch):
    from policy_dissertation.pipelines.policy_transformation import ingestion
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'policy.txt').write_text('Policy')

    def fail(*args):
        raise PermissionError('Output is read-only')

    monkeypatch.setattr(ingestion, 'write_json', fail)
    with pytest.raises(DissertationError, match='Cannot write output artifact'):
        ingest_documents(source, tmp_path / 'output')
