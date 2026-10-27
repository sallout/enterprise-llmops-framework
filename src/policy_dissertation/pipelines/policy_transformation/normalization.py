"""Conservative, auditable whitespace normalization and quality assessment."""
from collections import Counter
import hashlib
import math
import re

from policy_dissertation.errors import DissertationError
from .chunk_schemas import (
    NormalizationReport, NormalizedLine, NormalizedPage, QualityReport,
    QualityWarning, SourceSpan,
)
from .schemas import DocumentArtifact
from .structure import MODALS, heading, is_table, is_table_separator, table_cells


def issue(code: str, message: str, lines: list[NormalizedLine], resolution: str = 'unresolved') -> QualityWarning:
    spans = [line.source_span for line in lines]
    payload = code + repr([s.model_dump() for s in spans])
    return QualityWarning(warning_id=hashlib.sha256(payload.encode()).hexdigest(),
                          code=code, message=message, source_spans=spans, resolution=resolution)


def active(page: NormalizedPage) -> list[NormalizedLine]:
    return [line for line in page.lines if not line.excluded and (line.normalized_text.strip() or line.is_code)]


def render(lines: list[NormalizedLine]) -> str:
    return ''.join(('' if i == 0 else line.join_before) + line.normalized_text
                   for i, line in enumerate(lines))


def boundary_key(text: str) -> str:
    return re.sub(r'\bPage\s+\d+\s*$', 'Page <number>', text.strip())


def removable_boilerplate(text: str) -> bool:
    if MODALS.search(text):
        return False
    return bool(
        re.fullmatch(r'[A-Z]+-[A-Z0-9-]+ INTERNAL - CONTROLLED DOCUMENT', text)
        or re.fullmatch(r'Page <number>', text)
        or re.fullmatch(r'Synthetic enterprise policy template - requires organization-specific legal and regulatory validation before adoption\. Page <number>', text)
    )


def assess_and_normalize(document: DocumentArtifact) -> tuple[NormalizationReport, QualityReport]:
    if document.processing_status != 'parsed' or not document.content_hash:
        raise DissertationError('Only successfully parsed artifacts with a source hash can be normalized.')
    original_snapshot = document.model_dump_json()
    warnings = []
    pages = []
    sources = [(p.page_number, p.extracted_content, p.raw_text) for p in document.extraction_metadata.pages]
    if sources:
        if '\n\n'.join(text for _, text, _ in sources) != document.extracted_content:
            raise DissertationError('Page content is inconsistent with document extracted_content.')
        if [number for number, _, _ in sources] != list(range(1, len(sources) + 1)):
            raise DissertationError('PDF page references must be consecutive physical page numbers.')
    else:
        sources = [(None, document.extracted_content, '')]
    in_fence = False
    fence = ''
    for number, text, raw_text in sources:
        lines = []
        offset = 0
        blank_before = False
        for original in text.splitlines(keepends=True):
            value = original.rstrip('\r\n')
            fence_match = re.match(r'^ {0,3}(`{3,}|~{3,})', value)
            code = in_fence or bool(fence_match) or value.startswith(('    ', '\t'))
            line = NormalizedLine(
                line_id=f'{number if number is not None else "document"}:{offset}:{offset + len(value)}',
                source_span=SourceSpan(page_number=number, start=offset, end=offset + len(value)),
                original_text=value, normalized_text=value, is_code=code,
                join_before='\n\n' if blank_before and not code else '\n',
            )
            if original.endswith('\r\n') and not code:
                line.actions.append('line_endings_normalized')
            if not code:
                # Keep indentation and Markdown's two-space hard breaks. No word replacements.
                if not value.strip():
                    if blank_before:
                        line.actions.append('blank_line_collapsed')
                    if value:
                        line.normalized_text = ''
                        line.actions.append('blank_whitespace_normalized')
                elif value.endswith(' ') and not value.endswith('  '):
                    line.normalized_text = value[:-1]
                    line.actions.append('single_trailing_space_removed')
                if re.match(r'^[•●▪]\s+', line.normalized_text):
                    line.normalized_text = re.sub(r'^[•●▪](\s+)', r'-\1', line.normalized_text)
                    line.actions.append('explicit_bullet_normalized')
                    warnings.append(issue('explicit_bullet', 'Unambiguous Unicode list marker normalized to Markdown; words unchanged.', [line], 'corrected'))
                if re.match(r'^G(?:\s+|$)', value):
                    warnings.append(issue('suspicious_bullet', 'Leading G may be a misdecoded bullet; kept unchanged and treated only as a tentative clause boundary.', [line]))
                if re.search(r'[^\s] {4,}[^\s]|\ufffd', value):
                    warnings.append(issue('abnormal_formatting', 'Large internal space run or replacement glyph; preserved for review.', [line]))
            lines.append(line)
            blank_before = not value.strip()
            if fence_match:
                marker = fence_match[1]
                if not in_fence:
                    in_fence, fence = True, marker
                elif marker[0] == fence[0] and len(marker) >= len(fence):
                    in_fence = False
            offset += len(original)
        if any('blank_line_collapsed' in line.actions for line in lines):
            warnings.append(issue('excessive_whitespace', 'Excess blank lines collapsed outside code; no words rewritten.',
                                  [line for line in lines if not line.is_code and not line.original_text.strip()][:1], 'corrected'))
        pages.append(NormalizedPage(page_number=number, original_content=text, normalized_content='', lines=lines))
        # Contiguous Markdown table blocks: diagnose syntax and potential spacing loss only.
        block = []
        for line in [*lines, None]:
            if line is not None and not line.is_code and is_table(line.original_text):
                block.append(line)
                continue
            if block:
                widths = [len(table_cells(row.original_text)) for row in block]
                if len(block) < 2 or not is_table_separator(block[1].original_text) or len(set(widths)) != 1:
                    warnings.append(issue('malformed_table', 'Table lacks a valid separator or has inconsistent column counts; no repair attempted.', block))
                for row in block:
                    suspicious = False
                    for cell in table_cells(row.original_text):
                        cell = cell.replace('**', '')
                        if re.search(r'[a-z][A-Z]', cell):
                            suspicious = True
                        if raw_text and re.fullmatch(r'[A-Za-z/]{6,70}', cell):
                            match = re.search(r'\b' + r'\s*'.join(re.escape(c) for c in cell) + r'\b', raw_text)
                            if match and re.search(r'\s', match[0]):
                                suspicious = True
                    if suspicious:
                        warnings.append(issue('table_spacing_suspect', 'Possible merged table words (heuristic or raw-text comparison); content retained unchanged.', [row]))
                block = []
    for message in document.extraction_metadata.warnings:
        warnings.append(issue('parser_warning', message, []))
    # Detect recurring edge lines, but remove only an explicit, nonnormative allowlist.
    candidates = {}
    for page in pages:
        meaningful = active(page)
        for line in {item.line_id: item for item in meaningful[:2] + meaningful[-2:]}.values():
            if not line.is_code:
                candidates.setdefault(boundary_key(line.normalized_text), []).append(line)
    required = max(3, math.ceil(len(pages) * .75))
    for key, lines in candidates.items():
        if len({line.source_span.page_number for line in lines}) < required:
            continue
        safe = removable_boilerplate(key)
        warnings.append(issue('repeated_header_footer',
                              'Repeated boundary line excluded from normalized content.' if safe else 'Repeated boundary text retained: insufficient evidence that it is harmless boilerplate.',
                              lines, 'corrected' if safe else 'unresolved'))
        if safe:
            for line in lines:
                line.excluded = True
                line.actions.append('repeated_boilerplate_excluded')
    # Diagnose section order without inventing or renumbering headings.
    previous = {}
    seen = set()
    numbered_count = 0
    heading_lines = []
    number_levels = {}
    for page in pages:
        for line in active(page):
            section = heading(line.normalized_text, line.is_code)
            if section:
                heading_lines.append((line, section))
            if section and section[1]:
                numbered_count += 1
                numbers = tuple(map(int, section[1].split('.')))
                prefix, last = numbers[:-1], numbers[-1]
                expected = previous.get(prefix, 0) + 1
                if numbers in seen or last != expected or (prefix and prefix not in seen):
                    warnings.append(issue('section_numbering', 'Missing, repeated, out-of-order, or orphaned section number; no numbering repaired.', [line]))
                if prefix in number_levels and section[0] != number_levels[prefix] + 1:
                    warnings.append(issue('section_hierarchy', 'Numbered subsection has inconsistent Markdown depth; numeric ancestry is retained as context.', [line]))
                number_levels[numbers] = section[0]
                seen.add(numbers)
                previous[prefix] = last
    numbered_levels = {section[0] for _, section in heading_lines if section[1]}
    for line, section in heading_lines:
        if section[1] is None and section[0] in numbered_levels:
            warnings.append(issue('section_numbering', 'Unnumbered peer heading among numbered policy sections; retained without an invented number.', [line]))
    if not numbered_count:
        warnings.append(issue('section_numbering_absent', 'No reliable numbered policy sections detected; unnumbered headings remain usable.', []))
    # Join only obvious unfinished prose + lowercase continuation on adjacent pages.
    for left, right in zip(pages, pages[1:]):
        before, after = active(left), active(right)
        if not before or not after:
            continue
        a, b = before[-1], after[0]
        if a.is_code or b.is_code or heading(b.normalized_text) or is_table(a.normalized_text) or is_table(b.normalized_text):
            continue
        tail = a.normalized_text.rstrip('*_ ')
        unfinished = not re.search(r'[.!?:;]$', tail)
        lowercase = bool(re.match(r'^[a-z]', b.normalized_text))
        if unfinished and lowercase and not tail.endswith('-'):
            b.join_before = ' '
            b.actions.append('cross_page_continuation_joined')
            warnings.append(issue('cross_page_continuation', 'Unfinished prose joined to lowercase continuation on next physical page; words unchanged.', [a, b], 'corrected'))
        elif unfinished:
            warnings.append(issue('cross_page_uncertain', 'Possible unfinished clause at page boundary; retained without inferred text or word repair.', [a, b]))
    # Nonwhitespace lexical content must be identical after permitted marker/exclusion changes.
    for page in pages:
        for line in page.lines:
            if line.excluded:
                if not removable_boilerplate(boundary_key(line.original_text)):
                    raise DissertationError('Normalization invariant failed: unsafe exclusion.')
                continue
            old, new = line.original_text, line.normalized_text
            if 'explicit_bullet_normalized' in line.actions:
                old = re.sub(r'^[•●▪]', '-', old)
            if re.sub(r'\s', '', old) != re.sub(r'\s', '', new):
                raise DissertationError('Normalization invariant failed: lexical content changed.')
        page.normalized_content = render(active(page))
    if document.model_dump_json() != original_snapshot:
        raise DissertationError('Normalization mutated the source artifact.')
    normalized_content = render([line for page in pages for line in active(page)])
    status = 'review_required' if any(w.resolution == 'unresolved' for w in warnings) else (
        'normalized' if normalized_content != document.extracted_content else 'unchanged')
    report = NormalizationReport(
        document_id=document.document_id, source_content_hash=document.content_hash,
        source_path=document.source_path, original_extracted_content=document.extracted_content,
        normalized_content=normalized_content,
        pages=pages, normalization_status=status,
    )
    return report, QualityReport(document_id=document.document_id,
                                 source_content_hash=document.content_hash, warnings=warnings)
