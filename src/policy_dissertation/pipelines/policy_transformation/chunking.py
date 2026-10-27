"""Deterministic section-aware clause grouping, without semantic model inference."""
import hashlib
import json
import re

from policy_dissertation.errors import DissertationError
from .chunk_schemas import (
    ALGORITHM_VERSION, NormalizationReport, NormalizedLine, PolicyChunk,
    QualityReport, SectionContext,
)
from .normalization import active, issue, render
from .structure import BULLET, heading


def token_count(content: str) -> int:
    """UTF-8 byte count: conservative upper bound for ordinary byte-BPE text.

    Not a selected model's tokenizer, and does not budget future prompts or metadata.
    No tokenizer vocabulary, model, or network call is required.
    """
    return len(content.encode('utf-8'))


def build_chunks(report: NormalizationReport, quality: QualityReport, max_tokens: int = 2048) -> list[PolicyChunk]:
    if max_tokens <= 0:
        raise ValueError('max_tokens must be positive')
    # The most conservative unit is a whole list item, or an uninterrupted prose run.
    # Blank paragraphs within an item never cause a split of its exception/condition.
    sections: list[tuple[list[SectionContext], list[list[NormalizedLine]]]] = []
    stack: list[SectionContext] = []
    units: list[list[NormalizedLine]] = []
    unit: list[NormalizedLine] = []
    for line in [line for page in report.pages for line in active(page)]:
        detected = heading(line.normalized_text, line.is_code)
        if detected:
            if unit:
                units.append(unit)
                unit = []
            if units:
                sections.append((list(stack), units))
                units = []
            level, number, title = detected
            if number:
                # Explicit numeric ancestry wins over inconsistent PDF heading depth.
                while stack and stack[-1].section_id is not None:
                    if number.startswith(stack[-1].section_id + '.'):
                        break
                    stack.pop()
                while stack and stack[-1].section_id is None and stack[-1].level >= level:
                    stack.pop()
            else:
                while stack and stack[-1].level >= level:
                    stack.pop()
            stack.append(SectionContext(section_id=number, title=title, level=level,
                                        heading=line.normalized_text, source_span=line.source_span))
        else:
            marker = not line.is_code and BULLET.match(line.normalized_text)
            if marker and unit:
                # A marked exception must stay with the preceding requirement.
                continuation = re.match(r'^(?:[-+*•●▪]|G)\s+(?:unless|except|provided|only if|where|when)\b',
                                        line.normalized_text, re.IGNORECASE)
                if continuation:
                    quality.warnings.append(issue('conditional_list_continuation',
                                                   'Marked conditional text kept with the preceding clause; review dependency.', [line]))
                else:
                    if not re.search(r'[.!?;:]\s*[*_]*$', unit[-1].normalized_text):
                        # Independent short list items are common, so preserve rather than infer a missing sentence.
                        quality.warnings.append(issue('clause_boundary_uncertain',
                                                       'Previous list/prose unit lacks terminal punctuation; grouping may need review.', [unit[-1], line]))
                    units.append(unit)
                    unit = []
            unit.append(line)
    if unit:
        units.append(unit)
    if units:
        sections.append((list(stack), units))

    chunks = []
    for context, section_units in sections:
        groups: list[list[NormalizedLine]] = []
        group: list[NormalizedLine] = []
        for clause in section_units:
            if group and token_count(render(group + clause)) > max_tokens:
                groups.append(group)
                group = []
            group.extend(clause)
        if group:
            groups.append(group)
        for lines in groups:
            content = render(lines)
            count = token_count(content)
            oversized = count > max_tokens
            if oversized:
                quality.warnings.append(issue('oversized_clause',
                                               'Indivisible clause/prose run exceeds the configured byte-token budget; retained in full.', lines))
            spans = [line.source_span for line in lines]
            context_spans = [section.source_span for section in context]
            def relevant(warning):
                if not warning.source_spans:
                    return True
                return any(span in spans or span in context_spans for span in warning.source_spans)
            warnings = [warning for warning in quality.warnings if relevant(warning)]
            identity = json.dumps({
                'algorithm': ALGORITHM_VERSION, 'document': report.document_id,
                'source_hash': report.source_content_hash, 'content': content,
                'lines': [line.line_id for line in lines],
                'context': [section.model_dump() for section in context],
            }, sort_keys=True, ensure_ascii=False)
            status = 'review_required' if any(w.resolution == 'unresolved' for w in warnings) else (
                'normalized' if any(l.actions for l in lines) or any(w.resolution == 'corrected' for w in warnings) else 'unchanged')
            current = context[-1] if context else None
            chunks.append(PolicyChunk(
                chunk_id=hashlib.sha256(identity.encode('utf-8')).hexdigest(),
                document_id=report.document_id, source_content_hash=report.source_content_hash,
                source_path=report.source_path,
                section_id=current.section_id if current else None,
                section_title=current.title if current else 'Document preamble',
                parent_sections=context[:-1], section_heading=current.heading if current else None,
                section_source_spans=context_spans,
                page_numbers=sorted({s.page_number for s in spans if s.page_number is not None}),
                content=content, token_count=count, max_tokens=max_tokens, oversized=oversized,
                line_ids=[line.line_id for line in lines], source_spans=spans,
                normalization_status=status, quality_warnings=warnings,
            ))
    expected_lines = [line.line_id for page in report.pages for line in active(page)
                      if not heading(line.normalized_text, line.is_code)]
    actual_lines = [line_id for chunk in chunks for line_id in chunk.line_ids]
    if expected_lines != actual_lines or len(actual_lines) != len(set(actual_lines)):
        raise DissertationError('Chunk coverage invariant failed: missing, reordered, or duplicated source lines.')
    return chunks
