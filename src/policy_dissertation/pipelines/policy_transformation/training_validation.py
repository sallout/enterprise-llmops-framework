"""Fail-closed schema, provenance, extractive grounding, and model-assisted checks."""
from pydantic import ValidationError

from .data_io import digest
from .data_schemas import (
    AssistantMessage, Candidate, SystemMessage, TrainingRecord, UserMessage, ValidationOutcome,
)
from .generation import ModelCallError, compact

SYSTEM_INSTRUCTION = ('Answer corporate policy questions using the supplied policy scope and exceptions. '
                      'Preserve mandatory requirements, negations, conditions, thresholds, and approvals. '
                      'Do not substitute external legal advice for the policy.')


def validate_candidates(chunk, envelope, choices, context, gateway) -> ValidationOutcome:
    outcome = ValidationOutcome(chunk_id=chunk.chunk_id, generated_count=len(envelope.examples),
                                complete=True, accepted=[], rejected=[], errors=[])
    by_id = {item['passage_id']: item for item in choices}
    for index, raw in enumerate(envelope.examples):
        rejection = {'kind': 'example', 'chunk_id': chunk.chunk_id, 'document_id': chunk.document_id,
                     'candidate_index': index, 'candidate': gateway.scrub(raw), 'reasons': []}
        try:
            candidate = Candidate.model_validate(raw)
        except ValidationError:
            rejection['reasons'] = ['invalid_example_schema']
            outcome.rejected.append(rejection)
            continue
        reasons = []
        if candidate.document_id != chunk.document_id or candidate.chunk_id != chunk.chunk_id:
            reasons.append('invalid_source_reference')
        evidence = by_id.get(candidate.passage_id)
        if evidence is None:
            reasons.append('invalid_passage_reference')
        elif compact(candidate.response) != evidence['text']:
            reasons.append('answer_not_complete_verbatim_passage')
        if gateway.secret and gateway.secret in candidate.model_dump_json():
            reasons.append('credential_like_content')
        if reasons:
            rejection['reasons'] = reasons
            outcome.rejected.append(rejection)
            continue
        try:
            verdict = gateway.judge({
                'chunk': chunk.content, 'section': chunk.section_title,
                'policy_context': context, 'candidate': candidate.model_dump(),
                'evidence': evidence['text'],
            })
        except ModelCallError as exc:
            outcome.complete = False
            outcome.errors.append(exc.kind)
            rejection['reasons'] = ['grounding_verification_failed']
            outcome.rejected.append(rejection)
            continue
        for field in ('grounded', 'answers_question', 'preserves_conditions', 'scope_correct'):
            if not getattr(verdict, field):
                reasons.append(f'failed_{field}')
        if reasons:
            rejection['reasons'] = reasons
            rejection['verdict'] = gateway.scrub(verdict.model_dump())
            outcome.rejected.append(rejection)
            continue
        system = SYSTEM_INSTRUCTION + '\nPolicy: ' + context['policy_title']
        if context['scope']:
            system += '\nPolicy-wide scope:\n' + context['scope']
        if context['exceptions']:
            system += '\nPolicy-wide exceptions:\n' + context['exceptions']
        messages = (SystemMessage(content=system), UserMessage(content=candidate.instruction.strip()),
                    AssistantMessage(content=compact(candidate.response)))
        identity = digest([chunk.chunk_id, candidate.passage_id, [m.model_dump() for m in messages]])
        outcome.accepted.append(TrainingRecord(
            example_id=identity, messages=messages,
            metadata={
                'document_id': chunk.document_id, 'source_content_hash': chunk.source_content_hash,
                'source_path': chunk.source_path, 'chunk_id': chunk.chunk_id,
                'section_id': chunk.section_id, 'section_title': chunk.section_title,
                'parent_sections': [s.model_dump() for s in chunk.parent_sections],
                'page_numbers': chunk.page_numbers, 'source_spans': [s.model_dump() for s in chunk.source_spans],
                'passage_id': candidate.passage_id,
                'evidence_quote': chunk.content[evidence['start']:evidence['end']],
                'evidence_start': evidence['start'], 'evidence_end': evidence['end'],
                'context_references': context['references'],
                'quality_warnings': [w.model_dump() for w in chunk.quality_warnings],
                'validation': {'schema': 'passed', 'source_reference': 'passed',
                               'complete_extractive_passage': 'passed',
                               'grounding_judge': gateway.scrub(verdict.model_dump()),
                               'semantic_status': 'automated_checks_passed_not_human_verified'},
            },
        ))
    return outcome


def duplicate_key(record: TrainingRecord) -> str:
    # Deduplicate question/answer pairs globally, ignoring provenance/formatting differences.
    return digest([compact(record.messages[1].content).casefold(), compact(record.messages[2].content).casefold()])
