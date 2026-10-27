"""Conservative credential/identifier filters and exact actual-template length checks."""
import re
import json
from .adapters import InvalidExample

SECRETS = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}|\bhf_[A-Za-z0-9]{24,}|\bgh[pousr]_[A-Za-z0-9]{24,}|\bAKIA[0-9A-Z]{16}\b')
ASSIGNMENT = re.compile(r'''(?i)\b(?:api_key|secret_key|password|access_token)\b["']?\s*[:=]\s*["']([^"'\n]{12,})["']''')
SENSITIVE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')


def scan_sensitive(text, settings, configured_secrets=()):
    if settings.reject_secrets:
        if any(value and len(value) >= 8 and value in text for value in configured_secrets) or SECRETS.search(text):
            raise InvalidExample('possible_secret')
        for match in ASSIGNMENT.finditer(text):
            value = match[1]
            if not re.search(r'(?i)example|dummy|fake|changeme|placeholder|your_|test|<|\$|\{', value) and len(set(value)) >= 10 and re.search(r'[A-Za-z]', value) and re.search(r'\d', value):
                raise InvalidExample('possible_hardcoded_credential')
    if settings.reject_sensitive_identifiers and SENSITIVE.search(text):
        raise InvalidExample('possible_sensitive_identifier')


def validate_tokens(record, tokenizer, maximum, settings):
    model_fields=json.dumps([record['messages'],record.get('tools')],ensure_ascii=False)
    if any(token in model_fields for token in tokenizer.special_tokens):
        raise InvalidExample('embedded_template_control_token')
    for message in record['messages']:
        content = message.get('content') or ''
        if not settings.allow_marked_reasoning and message['role'] == 'assistant' and any(t in content for t in ('<think>', '</think>', '<analysis>', '</analysis>')):
            raise InvalidExample('unwanted_marked_reasoning')
    try:
        rendered_hash = None
        if hasattr(tokenizer, 'analyze'):
            count, rendered_hash = tokenizer.analyze(record['messages'], record.get('tools'))
        else:
            count = tokenizer.measure(record['messages'], record.get('tools'))
    except Exception:
        raise InvalidExample('tokenizer_template_incompatible') from None
    if count < 1:
        raise InvalidExample('empty_token_sequence')
    context_limit = tokenizer.info.get('max_context_tokens') or maximum
    if count > min(maximum, context_limit):
        raise InvalidExample('oversized_conversation')
    record['metadata']['preparation'] = {'token_count': count, 'max_tokens': maximum,
        'tokenizer_model': tokenizer.info['model'], 'tokenizer_revision': tokenizer.info['revision'],
        'chat_template_hash': tokenizer.info['chat_template_hash'], 'add_generation_prompt': False,
        'enable_thinking': False, 'truncated': False}
    if rendered_hash:
        record['metadata']['preparation']['rendered_token_hash'] = rendered_hash
    return count
