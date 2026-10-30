"""Three pipeline-specific adapters; preserve original conversations and evidence."""
import re
from pathlib import PurePosixPath
from urllib.parse import urlparse
from policy_dissertation.pipelines.policy_transformation.data_io import digest
from policy_dissertation.pipelines.policy_transformation.data_schemas import TrainingRecord
from policy_dissertation.pipelines.repository_data.schemas import CodeTrainingRecord
from policy_dissertation.pipelines.external_datasets.schemas import ExternalTrainingRecord
from .schemas import UnifiedRecord, Verification

CODING = {'implementation', 'debugging', 'unit_tests', 'refactoring', 'secure_coding'}
HEX = re.compile(r'^[0-9a-f]{64}$')


class InvalidExample(ValueError):
    pass


def require(m, fields):
    if any(not isinstance(m.get(k), str) or not m[k].strip() for k in fields):
        raise InvalidExample('missing_source_reference')


def hash_fields(m, fields):
    if any(not isinstance(m.get(k), str) or not HEX.fullmatch(m[k]) for k in fields):
        raise InvalidExample('invalid_source_hash')


def adapt(raw, pipeline, reference):
    classes = {'pipeline1': TrainingRecord, 'pipeline2': CodeTrainingRecord, 'pipeline3': ExternalTrainingRecord}
    if raw.get('schema_version') == 'unified-qwen-sft-v1':
        unified = UnifiedRecord.model_validate(raw)
        m = unified.metadata
        origin = m.get('pipeline')
        if origin not in classes:
            raise InvalidExample('unsupported_unified_origin')
        versions = {'pipeline1': 'pipeline1-extractive-v1', 'pipeline2': 'repository-data-v1', 'pipeline3': 'external-datasets-v1'}
        legacy = {'schema_version': versions[origin], 'example_id': m.get('upstream_example_id'),
                  'messages': raw['messages'], 'metadata': m.get('provenance')}
        checked = adapt(legacy, origin, reference)
        for field in ('id', 'domain', 'source', 'verification', 'tools'):
            if raw.get(field) != checked.get(field):
                raise InvalidExample('unified_provenance_mismatch')
        # Recompute grouping and official holdouts from original provenance, never trust supplied labels.
        checked['metadata'].update({k: v for k, v in m.items() if k not in
            ('pipeline', 'upstream_example_id', 'provenance', 'input_reference', 'component_group', 'official_holdout', 'split', 'split_group', 'preparation')})
        split = m.get('training_split')
        if split is not None:
            if split not in ('train', 'validation', 'test'):
                raise InvalidExample('invalid_prepared_training_split')
            if checked['metadata'].get('official_holdout') and checked['metadata']['official_holdout'] != split:
                raise InvalidExample('prepared_split_conflicts_with_official_holdout')
            if split in ('validation', 'test'):
                checked['metadata']['official_holdout'] = split
        return checked
    pipeline = {'pipeline1-extractive-v1':'pipeline1', 'repository-data-v1':'pipeline2', 'external-datasets-v1':'pipeline3'}.get(raw.get('schema_version'), pipeline)
    original = classes[pipeline].model_validate(raw)
    m = raw['metadata']
    tools = None
    if pipeline == 'pipeline1':
        require(m, ('document_id', 'source_path', 'chunk_id', 'evidence_quote'))
        hash_fields(m, ('source_content_hash',))
        if re.sub(r'\s+', ' ', m['evidence_quote']).strip() != original.messages[-1].content:
            raise InvalidExample('policy_evidence_mismatch')
        if 'page_numbers' in m and (not isinstance(m['page_numbers'],list) or any(type(n) is not int or n<1 for n in m['page_numbers'])):
            raise InvalidExample('invalid_page_reference')
        domain, source = 'corporate_compliance', 'pipeline1'
        verification = Verification(status='not_applicable', evidence={'upstream_policy_validation': m.get('validation', {})})
        component = 'policy:' + m['document_id']
        official = None
    elif pipeline == 'pipeline2':
        require(m, ('repository_id', 'file_path', 'source_unit_id', 'task_type'))
        hash_fields(m, ('file_hash', 'snapshot_hash'))
        path = PurePosixPath(m['file_path'])
        if path.is_absolute() or '..' in path.parts or '\\' in m['file_path'] or type(m.get('start_line')) is not int or type(m.get('end_line')) is not int or m['start_line'] < 1 or m['end_line'] < m['start_line']:
            raise InvalidExample('invalid_repository_reference')
        execution = m.get('execution', {})
        status = 'not_applicable'
        tests = None
        if m['task_type'] in CODING:
            if not (m.get('verification_status') == execution.get('status') == 'verified' and
                    execution.get('executor') == 'docker' and execution.get('exit_code') == 0 and
                    isinstance(execution.get('tests_run'), int) and execution['tests_run'] > 0 and
                    all(execution.get(k) == 0 for k in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes')) and
                    re.fullmatch(r'sha256:[0-9a-f]{64}', execution.get('image_id') or '')):
                raise InvalidExample('repository_code_execution_not_verified')
            status = 'upstream_local_pass'
            extracted = re.findall(r'(?m)^Tests:\s*\n```[^\n]*\n(.*?)\n```', original.messages[-1].content, re.DOTALL)
            tests = extracted[0] if len(extracted) == 1 else None
        verification = Verification(status=status, unit_tests=tests,
            evidence={'upstream_execution': execution, 'trajectory': m.get('trajectory', []),
                      'grounding_review': m.get('grounding_review', {}),
                      'evidence_authenticity': 'inherited_upstream_evidence_not_reexecuted_in_pipeline4'})
        domain, source = 'coding', 'pipeline2'
        component = 'repository-file:' + digest([m['repository_id'], m['file_path']])
        official = None
    else:
        require(m, ('dataset', 'config', 'split', 'revision', 'record_id', 'source_url', 'license', 'license_url'))
        hash_fields(m, ('content_hash',))
        url=urlparse(m['source_url'])
        if url.scheme != 'https' or url.hostname != 'huggingface.co' or not url.path.startswith('/datasets/'+m['dataset']) or not re.fullmatch(r'[0-9a-f]{40,64}',m['revision']):
            raise InvalidExample('invalid_dataset_revision_or_url')
        fields = m.get('source_fields', {})
        external = {k: fields[k] for k in ('unit_tests', 'tests_execution_status', 'average_test_score', 'llm_judgement') if k in fields}
        verification = Verification(status='externally_reported' if external else 'unverified',
            unit_tests=external.get('unit_tests'), evidence={'original_external_test_metadata': external,
                'locally_verified': False})
        source = m['dataset']
        domain = ('coding' if source == 'nvidia/OpenCodeInstruct' else
                  {'math': 'reasoning', 'science': 'science', 'swe': 'coding', 'terminal_agent': 'agentic',
                   'conversational_agent': 'agentic', 'safety': 'safety'}.get(m['config'], 'general_dialogue'))
        tools = fields.get('tools')
        if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) or
            t.get('type') != 'function' or not isinstance(t.get('function'), dict) or not t['function'].get('name') for t in tools)):
            raise InvalidExample('unsupported_tool_definitions')
        if any(msg.get('tool_calls') for msg in raw['messages']) and not tools:
            raise InvalidExample('tool_definitions_missing')
        if tools:
            names = {t['function']['name'] for t in tools}
            if any(c['function']['name'] not in names for msg in raw['messages'] for c in (msg.get('tool_calls') or [])):
                raise InvalidExample('undefined_tool_reference')
        component = 'dataset-example:' + digest([source, m['record_id']] if m['record_id_kind'] == 'upstream'
                          else [source, m['config'], m['revision'], m['record_id']])
        labels = [m['split'], fields.get('official_split'), fields.get('source_split')]
        official = 'test' if fields.get('is_benchmark') is True or any(str(x).lower() in ('test', 'testing') for x in labels) else (
            'validation' if any(str(x).lower() in ('validation', 'valid', 'dev', 'eval', 'evaluation') for x in labels) else None)
    record = UnifiedRecord(id=digest([pipeline, original.example_id, raw['messages'], tools]),
        messages=raw['messages'], domain=domain, source=source, verification=verification,
        metadata={'pipeline': pipeline, 'upstream_example_id': original.example_id, 'provenance': m,
                  'input_reference': reference, 'component_group': component, 'official_holdout': official}, tools=tools)
    return record.export_dict()


def legacy_record(raw):
    """Recover the original validated record from a unified export without changing messages."""
    unified = UnifiedRecord.model_validate(raw)
    m = unified.metadata
    versions = {'pipeline1': 'pipeline1-extractive-v1', 'pipeline2': 'repository-data-v1', 'pipeline3': 'external-datasets-v1'}
    return {'schema_version': versions[m['pipeline']], 'example_id': m['upstream_example_id'],
            'messages': raw['messages'], 'metadata': m['provenance']}


def adapt_native(raw, entry, ordinal, settings):
    """Explicit NVIDIA-native adapter reuses Pipeline 3's non-generative normalization."""
    specification = entry.get('source_spec', {})
    alias = specification.get('adapter', 'auto')
    if alias == 'auto':
        return raw
    from policy_dissertation.pipelines.external_datasets.normalization import normalize, RejectedRecord
    from policy_dissertation.pipelines.external_datasets.sources import DATASETS, LICENSE_URLS
    if specification.get('type') != 'huggingface' or specification.get('repo_id') != DATASETS[alias]:
        raise InvalidExample('native_adapter_requires_matching_pinned_nvidia_source')
    license_fields = entry.get('license_metadata', {})
    license_name = license_fields.get('license_name') if license_fields.get('license') == 'other' else license_fields.get('license')
    if license_name not in LICENSE_URLS:
        raise InvalidExample('unsupported_native_dataset_license')
    context = {'alias': alias, 'dataset': specification['repo_id'],
               'config': specification.get('config') or ('chat' if alias == 'nemotron' else 'train'),
               'split': specification.get('split', 'train'), 'revision': entry['resolved_revision'],
               'license': license_name, 'license_url': LICENSE_URLS[license_name]}
    try:
        return normalize(raw, context, ordinal, settings.pipeline3)
    except RejectedRecord as exc:
        raise InvalidExample('native_normalization:' + str(exc)) from None
