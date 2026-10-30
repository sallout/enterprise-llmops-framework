"""Small official base benchmarks; no answers or tests in model-facing prompts."""
import json
import random
from pathlib import Path
from huggingface_hub import HfApi
from datasets import load_dataset
from policy_dissertation.errors import DissertationError
from policy_dissertation.data.schemas import UnifiedRecord
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json
from .schemas import Case

VERSION = 'model-evaluation-suite-v1'
SOURCES = {'humaneval': ('openai/openai_humaneval', 'openai_humaneval', 'mit'),
           'mbpp': ('google-research-datasets/mbpp', 'sanitized', 'cc-by-4.0')}


def official_rows(name, token=None):
    repo, config, license_name = SOURCES[name]
    revision = HfApi(token=token).dataset_info(repo).sha
    rows = load_dataset(repo, config, split='test', revision=revision, token=token)
    source = {'dataset': repo, 'configuration': config, 'split': 'test', 'revision': revision,
              'license': license_name, 'variant': 'base tests, not EvalPlus extended tests'}
    return list(rows), source


def adapt(name, row, source):
    if name == 'humaneval':
        prompt = row['prompt']
        reference = prompt + row['canonical_solution']
        tests = row['test'] + '\ncheck(' + row['entry_point'] + ')\n'
        instruction = 'Complete the following Python function. Return the full implementation, including its signature and required imports. Return Python code only.\n\n' + prompt
    else:
        reference = row['code']
        tests = '\n'.join(row.get('test_imports', []) + row['test_list']) + '\n'
        # Function names are specified by visible assertions; these are standard
        # MBPP prompt examples, not additional hidden test generation.
        instruction = ('Write Python code for this task. Return the full implementation and required imports, with no explanation.\n\n'
                       + row['prompt'] + '\n\nRequired behaviour:\n' + '\n'.join(row['test_list']))
    if not reference.strip() or not tests.strip():
        raise DissertationError('Benchmark record lacks a reference or tests.')
    return Case(id=name + ':' + str(row['task_id']), suite=name,
                messages=[{'role': 'user', 'content': instruction}], reference=reference, tests=tests,
                source=source | {'record_id': str(row['task_id'])})


def prepare_suite(output, *, benchmarks=('humaneval', 'mbpp'), max_tasks=20,
                  enterprise=None, enterprise_limit=10, seed=42, token=None, provider=None):
    """Limit applies per benchmark; zero selects every available task. Never overwrite."""
    output = Path(output)
    if output.exists():
        raise DissertationError('Suite already exists. Reuse it for comparisons or choose a new suite path.')
    if max_tasks < 0 or enterprise_limit < 0 or not set(benchmarks) <= SOURCES.keys():
        raise DissertationError('Invalid benchmark or sample limit.')
    cases = []
    for name in dict.fromkeys(benchmarks):
        rows, source = (provider or official_rows)(name, token)
        rows = sorted(rows, key=lambda r: str(r['task_id']))
        random.Random(seed).shuffle(rows)
        for row in rows[:max_tasks or None]:
            cases.append(adapt(name, row, source))
    if enterprise:
        root = Path(enterprise).resolve()
        if output.resolve().is_relative_to(root):
            raise DissertationError('Frozen suite must be written outside the enterprise source directory.')
        # test.jsonl only: no training.jsonl union, no silent train fallback.
        path = root / 'test.jsonl'
        if not path.is_file():
            raise DissertationError('Prepared enterprise directory must contain test.jsonl; train fallback is forbidden.')
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest.get('training_ready') is not True:
            raise DissertationError('Enterprise manifest is not prepared for training/evaluation.')
        ids = {}
        for split in ('train', 'validation', 'test'):
            split_path = root/(split+'.jsonl')
            if manifest.get('output_hashes', {}).get(split+'.jsonl') != file_hash(split_path):
                raise DissertationError('Enterprise split checksum mismatch.')
            ids[split] = {json.loads(line)['id'] for line in split_path.read_text().splitlines() if line.strip()}
        if ids['test'] & (ids['train'] | ids['validation']):
            raise DissertationError('Enterprise test IDs overlap other splits.')
        records = [UnifiedRecord.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
        random.Random(seed).shuffle(records)
        for row in records[:enterprise_limit or None]:
            messages = row.export_dict()['messages']
            if messages[-1]['role'] != 'assistant' or not isinstance(messages[-1].get('content'), str):
                raise DissertationError('Enterprise test target must end in a textual assistant answer.')
            if row.tools or any(m['role'] == 'tool' or m.get('tool_calls') for m in messages):
                raise DissertationError('Enterprise tool execution evaluation is not implemented; provide text-only held-out tasks.')
            cases.append(Case(id='enterprise:' + row.id, suite='enterprise', messages=messages[:-1],
                reference=messages[-1]['content'], source={'file': str(path), 'sha256': file_hash(path),
                    'record_id': row.id, 'metadata': row.metadata, 'domain': row.domain,
                    'assessment': 'original held-out prompt; answer overlap only; human semantic review and assessment of available policy context required'}))
    if not cases or len({c.id for c in cases}) != len(cases):
        raise DissertationError('Evaluation suite is empty or has duplicate task IDs.')
    payload = {'schema_version': VERSION, 'seed': seed, 'max_tasks_per_benchmark': max_tasks,
               'enterprise_limit': enterprise_limit, 'cases': [c.model_dump() for c in cases],
               'benchmark_selection': 'seeded shuffle of official test splits; fixed selection for all models',
               'limitations': ['Base HumanEval/MBPP only; not HumanEval+/MBPP+.',
                  'Enterprise answer overlap is not semantic correctness.',
                  'Benchmark pretraining contamination remains unknown.']}
    payload['digest'] = digest(payload)
    dump_json(output, payload)
    return payload


def read_suite(path):
    payload = json.loads(Path(path).read_text())
    value = dict(payload); expected = value.pop('digest', None)
    if value.get('schema_version') != VERSION or digest(value) != expected:
        raise DissertationError('Frozen suite checksum/schema mismatch.')
    cases = [Case.model_validate(c) for c in payload['cases']]
    if not cases or len({c.id for c in cases}) != len(cases):
        raise DissertationError('Invalid frozen evaluation cases.')
    return payload, cases
