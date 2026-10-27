"""Small, offline fixtures plus regressions modelled on observed policy extraction."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from policy_dissertation.cli import app
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.chunk_batch import chunk_documents
from policy_dissertation.pipelines.policy_transformation.chunking import build_chunks, token_count
from policy_dissertation.pipelines.policy_transformation.chunk_schemas import PolicyChunk
from policy_dissertation.pipelines.policy_transformation.ingestion import ingest_documents
from policy_dissertation.pipelines.policy_transformation.normalization import active, assess_and_normalize, render
from policy_dissertation.pipelines.policy_transformation.schemas import DocumentArtifact, ExtractionMetadata, PageContent
from policy_dissertation.pipelines.policy_transformation.structure import MODALS


def artifact(*pages, raw=None):
    content = '\n\n'.join(pages)
    return DocumentArtifact(
        document_id='fixture-id', source_path='/fixtures/policy.pdf', source_type='pdf',
        content_hash=hashlib.sha256(content.encode()).hexdigest(), extracted_content=content,
        extraction_metadata=ExtractionMetadata(page_count=len(pages), pages=[
            PageContent(page_number=i+1, extracted_content=text, raw_text=(raw or pages)[i])
            for i, text in enumerate(pages)
        ]), processing_status='parsed',
    )


def process(document, limit=2048):
    report, quality = assess_and_normalize(document)
    return report, quality, build_chunks(report, quality, limit)


def codes(quality):
    return {warning.code for warning in quality.warnings}


def test_section_detection_and_parents():
    _, _, chunks = process(artifact('# Policy\n\n## **1. Controls**\n\n### 1.1 Approval\n\n- Requests MUST receive approval.\n\n### 1.2 Review\n\n- Reviews SHOULD be annual.'))
    assert [chunk.section_id for chunk in chunks] == ['1.1', '1.2']
    assert chunks[0].section_title == 'Approval'
    assert [parent.title for parent in chunks[0].parent_sections] == ['Policy', 'Controls']
    assert len(chunks[0].section_source_spans) == 3


def test_plain_numbered_sections_and_not_thresholds():
    _, _, chunks = process(artifact('1. Objective\n\nRequests MUST be reviewed.\n\n2. Controls\n\n72 hours after notification MAY be allowed.'))
    assert [chunk.section_id for chunk in chunks] == ['1', '2']
    assert '72 hours' in chunks[1].content


def test_mandatory_language_and_conditions_are_unchanged():
    text = ('# Policy\n\n## 1. Requirements\n\n'
            'G Production data MUST NOT be used unless Legal approval is recorded.\n\n'
            'Only when a documented exception exists MAY access continue.\n\n'
            'G Teams SHOULD review controls. Disclosure is Strictly Prohibited.\n\n'
            'G Exceptions MUST expire within 72 hours, provided the DPO approves.\n')
    doc = artifact(text)
    original = doc.model_dump_json()
    report, _, chunks = process(doc, 140)
    bodies = '\n'.join(chunk.content for chunk in chunks)
    assert Counter(MODALS.findall(text)) == Counter(MODALS.findall(bodies))
    first = next(chunk for chunk in chunks if 'Production data' in chunk.content)
    assert 'unless Legal approval is recorded.' in first.content
    assert 'Only when a documented exception exists MAY access continue.' in first.content
    assert '72 hours, provided the DPO approves.' in bodies
    assert doc.model_dump_json() == original
    assert report.original_extracted_content == text


def test_cross_page_privacy_clause_regression():
    doc = artifact(
        '## 6. Operational Control Requirements\n\nG Breach assessment MUST determine containment, and\n\n'
        'Synthetic enterprise policy template - requires organization-specific legal and regulatory validation before adoption. Page 1',
        'POL-PRV-002 INTERNAL - CONTROLLED DOCUMENT\n\nregulatory/individual notification obligations. Where GDPR Article 33 applies, notify within 72 hours, unless the breach is unlikely to result in risk.\n\n'
        'Synthetic enterprise policy template - requires organization-specific legal and regulatory validation before adoption. Page 2',
        'POL-PRV-002 INTERNAL - CONTROLLED DOCUMENT\n\n## 7. Records\n\nRecords MUST be retained.\n\n'
        'Synthetic enterprise policy template - requires organization-specific legal and regulatory validation before adoption. Page 3',
        'POL-PRV-002 INTERNAL - CONTROLLED DOCUMENT\n\n## 8. Review\n\nOwners MUST review.\n\n'
        'Synthetic enterprise policy template - requires organization-specific legal and regulatory validation before adoption. Page 4',
    )
    _, quality, chunks = process(doc)
    chunk = next(c for c in chunks if '72 hours' in c.content)
    assert chunk.page_numbers == [1, 2]
    assert 'and regulatory/individual' in chunk.content
    assert 'unless the breach is unlikely' in chunk.content
    assert chunk.section_id == '6'
    assert 'cross_page_continuation' in codes(quality)
    assert 'CONTROLLED DOCUMENT' not in chunk.content
    assert 'Synthetic enterprise' not in chunk.content


def test_ambiguous_bullet_not_rewritten_and_unicode_bullet_safe():
    text = '## 1. Rules\n\nG Access MUST be approved.\n\n• Access MUST NOT be public.'
    report, quality, chunks = process(artifact(text))
    assert 'G Access MUST' in report.normalized_content
    assert '- Access MUST NOT' in report.normalized_content
    assert {'suspicious_bullet', 'explicit_bullet'} <= codes(quality)
    assert next(w for w in quality.warnings if w.code == 'suspicious_bullet').resolution == 'unresolved'
    assert next(w for w in quality.warnings if w.code == 'explicit_bullet').resolution == 'corrected'
    assert all(chunk.normalization_status == 'review_required' for chunk in chunks)


@pytest.mark.parametrize('table', [
    '| A | B |\n|---|---|\n| 1 |',
    '| A | B |\n| 1 | 2 |',
])
def test_malformed_tables_are_flagged_not_repaired(table):
    report, quality, _ = process(artifact('## 1. Metadata\n\n' + table))
    assert 'malformed_table' in codes(quality)
    assert table in report.normalized_content


def test_merged_table_header_regression():
    text = '## 1. Metadata\n\n|PolicyOwner|Policyalignment|\n|---|---|\n|Legal|Required|'
    raw = 'Policy Owner\nPolicy alignment\nLegal\nRequired'
    report, quality, _ = process(artifact(text, raw=[raw]))
    assert 'table_spacing_suspect' in codes(quality)
    assert '|PolicyOwner|Policyalignment|' in report.normalized_content
    assert not any('repaired' in line.actions for p in report.pages for line in p.lines)


def test_repeated_requirements_never_removed():
    pages = [f'Employees MUST retain records.\n\n## {i}. Section\n\nBody text.\n\nPage {i}' for i in range(1,5)]
    report, quality, _ = process(artifact(*pages))
    assert report.normalized_content.count('Employees MUST retain records.') == 4
    assert any(w.code == 'repeated_header_footer' and w.resolution == 'unresolved' for w in quality.warnings)
    assert not any(line.excluded and 'MUST' in line.original_text for p in report.pages for line in p.lines)
    assert all('Page ' not in p.normalized_content for p in report.pages)


def test_unique_footnote_and_two_page_repetition_kept():
    report, _, _ = process(artifact('## 1. Rules\n\nG Owners MUST approve.\n\nPage 1',
                                    'G Owners MUST retain.\n\nPage 2'))
    assert 'Page 1' in report.normalized_content
    assert 'Page 2' in report.normalized_content


def test_numbering_gaps_duplicates_and_missing_parents():
    _, quality, _ = process(artifact('## 1. First\n\nText.\n\n## 3. Gap\n\nText.\n\n## 3. Repeat\n\nText.\n\n### 4.1 Orphan\n\nText.'))
    assert sum(w.code == 'section_numbering' for w in quality.warnings) == 3


def test_oversized_clause_not_truncated_and_ids_stable():
    doc = artifact('## 1. Requirements\n\n- Data MUST NOT be disclosed unless approved by Legal. ' + 'Condition applies. ' * 20)
    _, quality, first = process(doc, 64)
    _, _, second = process(doc, 64)
    assert len(first) == 1
    assert first == second
    assert first[0].oversized
    assert first[0].token_count > 64
    assert 'oversized_clause' in codes(quality)
    assert first[0].content.endswith('Condition applies.')
    assert first[0].chunk_id == second[0].chunk_id


def test_limits_avoid_overlap_and_keep_clauses():
    clauses = [f'- Rule {i} MUST be retained unless Legal approves.' for i in range(6)]
    _, _, chunks = process(artifact('## 1. Controls\n\n' + '\n\n'.join(clauses)), 100)
    assert all(c.token_count <= 100 for c in chunks)
    for clause in clauses:
        assert sum(clause in c.content for c in chunks) == 1
    ids = [i for c in chunks for i in c.line_ids]
    assert len(ids) == len(set(ids))


def test_source_offsets_and_roundtrip_reconstruction():
    doc = artifact('## 1. Controls\r\n\r\n• Owners MUST approve. \r\n\r\n\r\nG No public access.\r\n')
    report, _, chunks = process(doc)
    lookup = {line.line_id: line for p in report.pages for line in p.lines}
    for line in lookup.values():
        span = line.source_span
        assert doc.extraction_metadata.pages[span.page_number-1].extracted_content[span.start:span.end] == line.original_text
    for chunk in chunks:
        assert chunk.content == render([lookup[key] for key in chunk.line_ids])
        assert chunk.source_content_hash == doc.content_hash
        assert chunk.source_path == doc.source_path
        assert chunk.page_numbers == [1]


def test_text_sources_do_not_invent_page_numbers():
    doc = artifact('## 1. Rules\n\nOwners MUST approve.')
    doc.source_type = 'markdown'
    doc.extraction_metadata.pages = []
    report, _, chunks = process(doc)
    assert report.pages[0].page_number is None
    assert chunks[0].page_numbers == []
    assert all(span.page_number is None for span in chunks[0].source_spans)


def test_code_blocks_and_markdown_hard_breaks_preserved():
    text = '## 1. Code\n\n```text\nG keep this  \n\nMUST NOT replace\n```\n\nBody  \nnext line.'
    report, quality, chunks = process(artifact(text))
    assert '```text\nG keep this  \n\nMUST NOT replace\n```' in report.normalized_content
    assert 'Body  \nnext line.' in chunks[0].content
    assert 'suspicious_bullet' not in codes(quality)


def test_uncertain_boundary_not_joined():
    _, quality, chunks = process(artifact('## 1. Rules\n\n- Rule MUST remain incom-', 'plete but ambiguous.'))
    assert 'cross_page_uncertain' in codes(quality)
    assert 'incom-\nplete' in chunks[0].content


def test_conditional_list_kept_with_prior_requirement():
    _, quality, chunks = process(artifact('## 1. Rules\n\n- Owners MUST approve.\n\n- Unless Legal authorizes an exception.'), 40)
    assert len(chunks) == 1
    assert 'Unless Legal' in chunks[0].content
    assert chunks[0].oversized
    assert 'conditional_list_continuation' in codes(quality)


def test_token_method_is_documented_byte_upper_bound():
    assert token_count('café') == 5
    _, _, chunks = process(artifact('## 1. Rules\n\nOwners MUST approve.'))
    assert chunks[0].token_method == 'utf8_bytes_upper_bound_v1'
    with pytest.raises(ValueError):
        build_chunks(*assess_and_normalize(artifact('## 1. Rules\n\nText.')), max_tokens=0)


def test_end_to_end_parse_chunk_and_cli(tmp_path, monkeypatch):
    monkeypatch.setenv('LOG_LEVEL', 'INFO')
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'policy.md').write_text('# Policy\n\n## 1. Rules\n\n- Owners MUST approve unless Legal grants an exception.', encoding='utf-8')
    (source / 'unsupported.bin').write_bytes(b'unknown')
    parsed, output = tmp_path / 'parsed', tmp_path / 'chunks'
    ingest_documents(source, parsed)
    before = {p: p.read_bytes() for p in parsed.rglob('*.json')}
    result = CliRunner().invoke(app, ['policies', 'chunk', '--input', str(parsed), '--output', str(output), '--max-tokens', '80'])
    assert result.exit_code == 0, result.output
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['documents_processed'] == 1
    assert summary['documents_skipped'] == 1
    assert summary['documents_failed'] == 0
    doc = summary['documents'][0]
    chunks = [PolicyChunk.model_validate_json(line) for line in (output / doc['chunks_path']).read_text().splitlines()]
    assert len(chunks) == summary['chunk_count'] == 1
    assert 'unless Legal grants an exception.' in chunks[0].content
    assert all(p.read_bytes() == content for p, content in before.items())
    assert chunk_documents(parsed, output, 80).model_dump() == summary


def test_bad_artifact_does_not_stop_other_documents(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    (source / 'a.md').write_text('## 1. Rules\n\nValid.')
    (source / 'b.md').write_text('## 1. Rules\n\nValid.')
    parsed = tmp_path / 'parsed'
    manifest = ingest_documents(source, parsed)
    (parsed / manifest.artifacts[0].artifact_path).write_text('{broken json')
    summary = chunk_documents(parsed, tmp_path / 'chunks')
    assert summary.documents_failed == 1
    assert summary.documents_processed == 1
    assert summary.errors


def test_bad_page_references_and_content_rejected():
    doc = artifact('## 1. Rules\n\nText.')
    doc.extracted_content += 'extra'
    with pytest.raises(DissertationError, match='inconsistent'):
        assess_and_normalize(doc)


def test_overlapping_output_rejected(tmp_path):
    with pytest.raises(DissertationError, match='overlap'):
        chunk_documents(tmp_path, tmp_path / 'output')


def test_abnormal_formatting_and_absent_section_numbering():
    text = '# Rules\n\nOwners    MUST retain records with \ufffd undecoded glyph.'
    report, quality, chunks = process(artifact(text))
    assert {'abnormal_formatting', 'section_numbering_absent'} <= codes(quality)
    assert 'Owners    MUST' in chunks[0].content
    assert '\ufffd' in report.normalized_content


def test_unknown_repeated_header_is_kept():
    report, quality, _ = process(artifact(*[
        f'Confidential Review Note\n\n## {i}. Rules\n\nOwners MUST approve.' for i in range(1, 5)
    ]))
    assert report.normalized_content.count('Confidential Review Note') == 4
    assert any(w.code == 'repeated_header_footer' and w.resolution == 'unresolved' for w in quality.warnings)


def test_pdf_parsing_to_chunking(tmp_path):
    import pymupdf

    source = tmp_path / 'policies'
    source.mkdir()
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), '1. Operational Controls', fontsize=20)
        page.insert_text((72, 120), 'Teams MUST report within 72 hours unless Legal approves.', fontsize=11)
        document.save(source / 'policy.pdf')
    parsed, output = tmp_path / 'parsed', tmp_path / 'chunks'
    ingest_documents(source, parsed)
    summary = chunk_documents(parsed, output)
    assert summary.documents_processed == 1
    chunk = PolicyChunk.model_validate_json((output / summary.documents[0].chunks_path).read_text().strip())
    assert chunk.page_numbers == [1]
    assert chunk.section_id == '1'
    assert '72 hours unless Legal approves.' in chunk.content


def test_different_budget_same_chunk_identity_when_grouping_unchanged():
    doc = artifact('## 1. Controls\n\nOwners MUST approve.')
    _, _, small = process(doc, 100)
    _, _, large = process(doc, 200)
    assert small[0].chunk_id == large[0].chunk_id


def test_source_changes_change_chunk_identity():
    _, _, first = process(artifact('## 1. Controls\n\nOwners MUST approve.'))
    _, _, second = process(artifact('## 1. Controls\n\nOwners MUST NOT approve.'))
    assert first[0].chunk_id != second[0].chunk_id


def test_inconsistent_heading_depth_retains_numeric_parent_context():
    _, quality, chunks = process(artifact('# Policy\n\n## 1. Controls\n\nRule.\n\n## 1.1 Approval\n\nOwners MUST approve.\n\n### 2. Review\n\nReview annually.'))
    approval = next(c for c in chunks if c.section_id == '1.1')
    assert [s.section_id for s in approval.parent_sections] == [None, '1']
    review = next(c for c in chunks if c.section_id == '2')
    assert [s.title for s in review.parent_sections] == ['Policy']
    assert 'section_hierarchy' in codes(quality)


def test_missing_number_on_peer_heading_is_flagged():
    _, quality, _ = process(artifact('# Policy\n\n## 1. Controls\n\nText.\n\n## Exceptions\n\nLegal MUST approve.'))
    assert 'section_numbering' in codes(quality)
