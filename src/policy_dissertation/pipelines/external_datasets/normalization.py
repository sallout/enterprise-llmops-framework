"""Dataset adapters and explicit non-generative filtering rules."""
import json
import re
from pydantic import ValidationError
from policy_dissertation.pipelines.policy_transformation.data_io import digest
from .schemas import VERSION, ExternalTrainingRecord

REASONING_FIELDS = {'reasoning', 'reasoning_content', 'analysis'}
MESSAGE_FIELDS = {'role', 'content', 'name', 'tool_call_id', 'tool_calls'}


class RejectedRecord(ValueError):
    pass


def normalize(raw, source, ordinal, settings):
    if not isinstance(raw, dict):
        raise RejectedRecord('unsupported_record_structure')
    try:
        json.dumps(raw, allow_nan=False)
    except (ValueError, TypeError):
        raise RejectedRecord('non_json_value') from None
    raw_hash = digest(raw)
    changes = []
    def clean_fields(fields, location):
        result = {}
        for key, value in fields.items():
            if key in REASONING_FIELDS:
                if value:
                    if settings.reasoning_policy == 'reject':
                        raise RejectedRecord('reasoning_trace')
                    changes.append({'location': location + '.' + key, 'action': 'remove_reasoning_field',
                                    'original_hash': digest(value)})
            else:
                result[key] = value
        return result
    if source['alias'] == 'opencode':
        if not isinstance(raw.get('input'), str) or not isinstance(raw.get('output'), str):
            raise RejectedRecord('missing_input_or_output')
        messages = [{'role': 'user', 'content': raw['input']}, {'role': 'assistant', 'content': raw['output']}]
        fields = {k: v for k, v in raw.items() if k not in ('input', 'output')}
    else:
        messages = raw.get('messages')
        if not isinstance(messages, list) or not messages:
            raise RejectedRecord('missing_messages')
        fields = {k: v for k, v in raw.items() if k != 'messages'}
    fields = clean_fields(fields, 'record')
    normalized, extras = [], {}
    for index, original in enumerate(messages):
        if not isinstance(original, dict):
            raise RejectedRecord('unsupported_message_structure')
        m = clean_fields(original, f'messages[{index}]')
        content = m.get('content')
        if content is not None and not isinstance(content, str):
            raise RejectedRecord('unsupported_nontext_content')
        if isinstance(content, str) and m.get('role') == 'assistant' and ('<think>' in content or '</think>' in content):
            if settings.reasoning_policy == 'reject':
                raise RejectedRecord('reasoning_trace')
            # Only closed leading marked blocks can be removed. Never guess a missing final answer.
            original_content = content
            while content.lstrip().startswith('<think>'):
                match = re.match(r'^\s*<think>((?:(?!<think>).)*?)</think>', content, re.DOTALL)
                if not match:
                    raise RejectedRecord('unclosed_or_nested_reasoning_trace')
                content = content[match.end():].lstrip('\r\n ')
            if '<think>' in content or '</think>' in content:
                raise RejectedRecord('ambiguous_reasoning_trace')
            m['content'] = content
            changes.append({'location': f'messages[{index}].content', 'action': 'strip_leading_think',
                            'original_hash': digest(original_content), 'result_hash': digest(content)})
        if m.get('role') == 'assistant' and not m.get('tool_calls') and isinstance(m.get('content'), str):
            if len(m['content'].strip()) < settings.min_assistant_chars:
                raise RejectedRecord('empty_or_short_answer')
        extra = {k: v for k, v in m.items() if k not in MESSAGE_FIELDS}
        if extra:
            extras[str(index)] = extra
        normalized.append({k: v for k, v in m.items() if k in MESSAGE_FIELDS})
    text = json.dumps(normalized, ensure_ascii=False, separators=(',', ':'))
    if any(token and token in text for token in settings.forbidden_tokens):
        raise RejectedRecord('incompatible_special_token')
    # Analysis channel tags aren't inferred from ordinary words or code identifiers.
    if any(m.get('role') == 'assistant' and isinstance(m.get('content'), str) and
           any(marker and marker in m['content'] for marker in settings.unsupported_reasoning_markers) for m in normalized):
        raise RejectedRecord('unsupported_reasoning_marker')
    token_count = len(text.encode('utf-8'))
    if token_count > settings.max_tokens:
        raise RejectedRecord('oversized_conversation')
    source_id = raw.get('id')
    upstream = isinstance(source_id, (str, int)) and not isinstance(source_id, bool) and str(source_id).strip()
    record_id = str(source_id) if upstream else f'row:{ordinal}'
    metadata = {k: source[k] for k in ('dataset', 'config', 'split', 'revision', 'license', 'license_url')}
    metadata.update(record_id=record_id, record_id_kind='upstream' if upstream else 'stream_ordinal',
                    stream_ordinal=ordinal, content_hash=raw_hash,
                    source_url=f"https://huggingface.co/datasets/{source['dataset']}/tree/{source['revision']}",
                    source_fields=fields, source_message_metadata=extras, transformations=changes,
                    token_count=token_count,
                    external_test_results_status='not_locally_verified' if any(raw.get(k) is not None for k in
                        ('unit_tests', 'tests_execution_status', 'average_test_score')) else 'not_provided')
    try:
        record = ExternalTrainingRecord(schema_version=VERSION,
            example_id=digest([VERSION, source['dataset'], source['config'], source['split'], source['revision'], record_id, raw_hash]),
            messages=normalized, metadata=metadata)
    except (ValidationError, ValueError, TypeError):
        raise RejectedRecord('invalid_conversation_schema') from None
    result = record.model_dump()
    result['messages'] = [m.model_dump(exclude_unset=True) for m in record.messages]
    return result
