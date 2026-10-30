"""Synthetic transport/API fixtures; no upload or runtime-verification evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.data.adapters import adapt, legacy_record, InvalidExample
from policy_dissertation.data.schemas import UnifiedRecord
from policy_dissertation.data.sources import Source, read_config, materialize
from policy_dissertation.data.hub import publish, protect_exports
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.dataset_preparation.pipeline import run_pipeline4
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from test_pipeline4 import policy, repository, external, FakeTokenizer


@pytest.mark.parametrize('origin,raw', [('pipeline1', policy()), ('pipeline2', repository()), ('pipeline3', external())])
def test_shared_roundtrip_and_idempotent_adapters(origin, raw):
    r = adapt(raw, origin, {'line': 1})
    UnifiedRecord.model_validate(r)
    assert legacy_record(r) == raw
    again = adapt(r, 'arbitrary-source-name', {'line': 2})
    assert again['id'] == r['id']
    assert again['messages'] == raw['messages']
    assert again['verification'] == r['verification']
    assert again['metadata']['input_reference'] == {'line': 2}


@pytest.mark.parametrize('field,value', [('verification', {'status': 'upstream_local_pass'}), ('source', 'invented'), ('id', 'c'*64)])
def test_unified_tampering_rejected(field, value):
    r = adapt(external(), 'pipeline3', {})
    r[field] = value
    with pytest.raises((InvalidExample, ValueError)):
        adapt(r, 'pipeline3', {})


def export(tmp_path):
    records = [adapt(policy(i), 'pipeline1', {}) for i in range(6)]
    tmp_path.mkdir(exist_ok=True)
    p = tmp_path/'training.jsonl'
    p.write_text(''.join(json.dumps(r)+'\n' for r in records))
    (tmp_path/'report.json').write_text(json.dumps({'status': 'COMPLETED', 'errors': [], 'accepted_examples': 6,
        'output_hashes': {'training.jsonl': file_hash(p)}}))
    return records


class MockHub:
    def __init__(self, private=True, fail=False):
        self.private, self.fail, self.uploads, self.creates = private, fail, [], []
    def create_repo(self, *args, **kwargs):self.creates.append((args, kwargs))
    def dataset_info(self, *args, **kwargs):return SimpleNamespace(private=self.private, sha='a'*40)
    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        if self.fail:raise RuntimeError('private provider response must not appear')
        directory = Path(kwargs['folder_path'])
        assert {p.name for p in directory.iterdir()} == {'training.jsonl', 'manifest.json', 'README.md'}
        assert kwargs['allow_patterns'] == ['training.jsonl', 'manifest.json', 'README.md']
        assert all(UnifiedRecord.model_validate_json(l) for l in (directory/'training.jsonl').read_text().splitlines())
        return SimpleNamespace(oid='b'*40)


def test_private_publish_allowlist_and_revision(tmp_path):
    export(tmp_path)
    (tmp_path/'rejected.jsonl').write_text('never upload')
    (tmp_path/'.env').write_text('never upload either')
    hub = MockHub()
    result = publish(tmp_path, 'owner/policies', Settings(_env_file=None, hf_token='synthetic-token'), api=hub)
    assert result['revision'] == 'b'*40 and result['private']
    assert hub.creates[0][1]['private'] is True
    assert json.loads((tmp_path/'manifest.json').read_text())['publication'] == result


@pytest.mark.parametrize('mode', ['public', 'missing_token', 'failed_run', 'checksum'])
def test_publication_fail_closed(tmp_path, mode):
    export(tmp_path)
    settings = Settings(_env_file=None, hf_token=None if mode == 'missing_token' else 'synthetic-token')
    hub = MockHub(private=mode != 'public')
    if mode == 'failed_run':
        report = json.loads((tmp_path/'report.json').read_text());report['status'] = 'FAILED'
        (tmp_path/'report.json').write_text(json.dumps(report))
    if mode == 'checksum':(tmp_path/'training.jsonl').write_text('changed')
    with pytest.raises(DissertationError):publish(tmp_path, 'owner/policies', settings, api=hub)
    assert not hub.uploads


def test_upload_failure_preserves_local_and_scrubs_error(tmp_path, monkeypatch):
    import policy_dissertation.data.hub as module
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    export(tmp_path)
    original = (tmp_path/'training.jsonl').read_bytes()
    hub = MockHub(fail=True)
    with pytest.raises(DissertationError, match='bounded retries') as error:
        publish(tmp_path, 'owner/policies', Settings(_env_file=None, hf_token='synthetic-token'), api=hub)
    assert len(hub.uploads) == 3
    assert 'private provider' not in str(error.value)
    assert (tmp_path/'training.jsonl').read_bytes() == original


def test_yaml_relative_paths_are_config_relative(tmp_path, monkeypatch):
    config = tmp_path/'datasets.yaml'
    config.write_text('sources:\n- name: policies\n  type: local\n  path: inputs/policies.jsonl\n  weight: 1\noutput:\n  directory: prepared\n')
    monkeypatch.chdir('/')
    parsed = read_config(config)
    assert parsed.sources[0].path == str(tmp_path/'inputs/policies.jsonl')
    assert parsed.output.directory == str(tmp_path/'prepared')


@pytest.mark.parametrize('text', ['sources: []', 'sources: [{name: x, type: local, path: a, weight: .nan}]',
    'sources: [{name: x, type: local, path: a, weight: 1}, {name: x, type: local, path: b, weight: 1}]'])
def test_bad_configuration_rejected(tmp_path, text):
    p = tmp_path/'bad.yaml';p.write_text(text)
    with pytest.raises(DissertationError):read_config(p)


def test_local_directory_and_standalone_file(tmp_path):
    export(tmp_path)
    settings = Settings(_env_file=None)
    for path in (tmp_path, tmp_path/'training.jsonl'):
        result = materialize(Source(name='local', type='local', path=str(path), weight=1), tmp_path/'cache', settings)
        assert result['upstream_accepted'] == 6
        assert result['training_hash'] == file_hash(tmp_path/'training.jsonl')
    with pytest.raises(DissertationError, match='checksum'):
        materialize(Source(name='local', type='local', path=str(tmp_path), weight=1, sha256='a'*64), tmp_path/'cache', settings)


def test_remote_pin_stream_cache_and_verification(tmp_path, monkeypatch):
    import huggingface_hub, datasets
    hub = MockHub()
    hub.list_repo_files = lambda *a, **k: ['data.jsonl']
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **k: hub)
    calls = []
    def stream(*a, **k):
        calls.append(k)
        return iter([external(i) for i in range(6)])
    monkeypatch.setattr(datasets, 'load_dataset', stream)
    source = Source(name='remote', type='huggingface', repo_id='owner/data', weight=1)
    settings = Settings(_env_file=None, hf_token='synthetic-token')
    first = materialize(source, tmp_path, settings)
    second = materialize(source, tmp_path, settings)
    assert first == second and len(calls) == 1
    assert calls[0]['revision'] == 'a'*40 and calls[0]['streaming'] is True
    assert calls[0]['token'] == 'synthetic-token'
    assert 'synthetic-token' not in json.dumps(first)


def test_mixed_sources_preserve_evidence_dedupe_and_isolate_splits(tmp_path, monkeypatch):
    import huggingface_hub, datasets
    local = tmp_path/'policies';export(local)
    hub = MockHub();hub.list_repo_files = lambda *a, **k: ['data.jsonl']
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **k: hub)
    # Same policy appears through local and remote sources; retain exactly one copy.
    monkeypatch.setattr(datasets, 'load_dataset', lambda *a, **k: iter([policy(0)]+[external(i) for i in range(10)]))
    sources = [Source(name='policies', type='local', path=str(local), weight=.5),
               Source(name='external', type='huggingface', repo_id='owner/data', weight=.5)]
    settings = Settings(_env_file=None, pipeline4={'mix_ratios': {'policies': .5, 'external': .5}})
    r = run_pipeline4(tmp_path/'unused', tmp_path/'out', settings, overwrite=True, sources=sources, tokenizer=FakeTokenizer(), research_report=tmp_path/'research.md')
    assert r['status'] == 'COMPLETED_WITH_WARNINGS' and r['input_records'] == 17
    assert r['duplicates_removed'] == 1 and not r['errors']
    assert r['splitting']['component_leakage'] == r['splitting']['near_pair_leakage'] == 0
    assert r['inputs'][1]['resolved_revision'] == 'a'*40
    output = [json.loads(line) for name in ('train', 'validation', 'test') for line in (tmp_path/f'out/{name}.jsonl').read_text().splitlines()]
    assert len({x['id'] for x in output}) == len(output)
    for row in output:
        UnifiedRecord.model_validate(row)
        assert row['verification']['executed_in_pipeline4'] is False
        assert row['metadata']['dataset_source']['name'] in ('policies', 'external')
    first = {n:(tmp_path/f'out/{n}.jsonl').read_bytes() for n in ('train','validation','test')}
    again = run_pipeline4(tmp_path/'unused', tmp_path/'out', settings, overwrite=True, sources=sources, tokenizer=FakeTokenizer(), research_report=tmp_path/'research.md')
    assert again['resumed_stages'] == ['load','deduplication']
    assert all((tmp_path/f'out/{n}.jsonl').read_bytes() == text for n,text in first.items())


def test_existing_exports_require_explicit_cli_overwrite(tmp_path):
    export(tmp_path)
    with pytest.raises(DissertationError, match='--overwrite'):protect_exports(tmp_path)
    protect_exports(tmp_path, True)
    result = CliRunner().invoke(app, ['pip1_data', '--output', str(tmp_path)])
    assert result.exit_code == 1 and '--overwrite' in result.output


def test_yaml_cli_with_mock_tokenizer(tmp_path, monkeypatch):
    import policy_dissertation.pipelines.dataset_preparation.pipeline as pipeline
    local = tmp_path/'policies';export(local)
    monkeypatch.setattr(pipeline, 'load_tokenizer', lambda *a: FakeTokenizer())
    config = tmp_path/'datasets.yaml'
    config.write_text('sources:\n- name: policies\n  type: local\n  path: policies\n  weight: 1\noutput:\n  directory: final\n  max_tokens: 4096\n')
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ['pip4_prepare', '--config', str(config)])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path/'final/report.json').read_text())
    assert report['accepted_examples'] == 6
    assert not report['training_ready']  # Fake tokenizer cannot certify training readiness.


def test_published_snapshot_manifest_is_checked(tmp_path, monkeypatch):
    import huggingface_hub
    data = tmp_path/'published';export(data)
    manifest = data/'manifest.json'
    manifest.write_text(json.dumps({'accepted_examples': 6, 'output_hashes': {'training.jsonl': file_hash(data/'training.jsonl')}}))
    hub = MockHub();hub.list_repo_files = lambda *a, **k: ['training.jsonl','manifest.json']
    monkeypatch.setattr(huggingface_hub, 'HfApi', lambda **k: hub)
    calls = []
    def download(repo, name, **kwargs):
        calls.append(kwargs)
        return str(data/name)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    source = Source(name='published',type='huggingface',repo_id='owner/data',weight=1)
    entry = materialize(source,tmp_path/'cache',Settings(_env_file=None))
    assert entry['training_hash'] == file_hash(data/'training.jsonl')
    assert all(c['revision'] == 'a'*40 for c in calls)
    manifest.write_text(json.dumps({'output_hashes':{'training.jsonl':'c'*64}}))
    with pytest.raises(DissertationError,match='checksum'):
        materialize(source,tmp_path/'other-cache',Settings(_env_file=None))


def test_native_nvidia_adapter_preserves_external_tests_and_license():
    from policy_dissertation.data.adapters import adapt_native
    raw = {'id':'original-7','input':'What is the sum of one and two?', 'output':'The sum is three.',
           'unit_tests':['assert add(1,2)==3'],'tests_execution_status':'passed'}
    entry = {'source_spec': {'adapter':'opencode','type':'huggingface','repo_id':'nvidia/OpenCodeInstruct','config':'train'},
             'resolved_revision':'a'*40,'license_metadata':{'license':'cc-by-4.0'}}
    normalized = adapt_native(raw,entry,7,Settings(_env_file=None))
    unified = adapt(normalized,'remote',{})
    assert unified['verification']['status'] == 'externally_reported'
    assert unified['verification']['unit_tests'] == raw['unit_tests']
    assert unified['verification']['evidence']['locally_verified'] is False
    assert unified['metadata']['provenance']['license'] == 'cc-by-4.0'
    entry['source_spec']['repo_id'] = 'untrusted/other'
    with pytest.raises(InvalidExample):adapt_native(raw,entry,7,Settings(_env_file=None))


def test_remote_record_cap_is_not_an_extra_prefetch(tmp_path,monkeypatch):
    import huggingface_hub,datasets
    hub=MockHub();hub.list_repo_files=lambda *a,**k:['data.jsonl']
    monkeypatch.setattr(huggingface_hub,'HfApi',lambda **k:hub)
    reads=[]
    def rows():
        for i in range(20):
            reads.append(i)
            yield external(i)
    monkeypatch.setattr(datasets,'load_dataset',lambda *a,**k:rows())
    s=Source(name='bounded',type='huggingface',repo_id='owner/data',weight=1,max_records=2)
    entry=materialize(s,tmp_path,Settings(_env_file=None))
    assert reads==[0,1] and entry['upstream_accepted']==2


def test_direct_preparation_api_does_not_overwrite(tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'training.jsonl').write_text('existing')
    with pytest.raises(DissertationError,match='--overwrite'):
        run_pipeline4(tmp_path/'source',out,Settings(_env_file=None),tokenizer=FakeTokenizer())
    assert (out/'training.jsonl').read_text()=='existing'


def test_pipeline3_filters_incompatible_tool_records_before_commit(tmp_path):
    from test_pipeline3 import FakeProvider, chat
    from policy_dissertation.pipelines.external_datasets.pipeline import run_pipeline3
    bad = chat()
    bad['messages'] = [{'role':'user','content':'Find a value'},
       {'role':'assistant','content':None,'tool_calls':[{'id':'x','type':'function','function':{'name':'lookup','arguments':'{}'}}]},
       {'role':'tool','content':'found','tool_call_id':'x'}, {'role':'assistant','content':'The value was found.'}]
    provider = FakeProvider(rows={'nemotron':[bad,chat()]})
    result = run_pipeline3(tmp_path/'p3',Settings(_env_file=None,pipeline3={'shuffle_buffer':1}),1,
                          selected=('nemotron',), provider=provider,research_report=tmp_path/'r.md')
    assert result['accepted_examples']==1
    assert result['rejection_reasons']['incompatible_unified_schema']==1


def test_explicit_subset_publication_keeps_failed_producer_report(tmp_path):
    export(tmp_path)
    path=tmp_path/'report.json'
    report=json.loads(path.read_text());report.update(status='FAILED',errors=['mandatory_unit_processing_incomplete'],failed_units=4,pending_units=0)
    path.write_text(json.dumps(report));before=path.read_bytes()
    hub=MockHub()
    base_upload=hub.upload_folder
    def upload(**kwargs):
        stage=Path(kwargs['folder_path'])
        manifest=json.loads((stage/'manifest.json').read_text())
        assert manifest['source_run']['accepted_subset_only'] is True
        assert manifest['source_run']['status']=='FAILED'
        return base_upload(**kwargs)
    hub.upload_folder=upload
    result=publish(tmp_path,'owner/accepted',Settings(_env_file=None,hf_token='synthetic-token'),api=hub,allow_incomplete_run=True)
    assert result['accepted_subset_only'] is True and result['source_run_status']=='FAILED'
    assert path.read_bytes()==before


@pytest.mark.parametrize('mode',['checksum','count','invalid_record'])
def test_subset_mode_still_enforces_record_validation(tmp_path,mode):
    export(tmp_path)
    path=tmp_path/'report.json';report=json.loads(path.read_text())
    report.update(status='FAILED',errors=['mandatory_unit_processing_incomplete'])
    if mode=='count':report['accepted_examples']=99
    if mode=='checksum':report['output_hashes']['training.jsonl']='a'*64
    if mode=='invalid_record':
        data=tmp_path/'training.jsonl';rows=data.read_text().splitlines();raw=json.loads(rows[0]);raw['verification']['status']='upstream_local_pass'
        rows[0]=json.dumps(raw);data.write_text('\n'.join(rows)+'\n')
        report['output_hashes']['training.jsonl']=file_hash(data)
    path.write_text(json.dumps(report));hub=MockHub()
    with pytest.raises((DissertationError,ValueError)):
        publish(tmp_path,'owner/accepted',Settings(_env_file=None,hf_token='synthetic-token'),api=hub,allow_incomplete_run=True)
    assert not hub.uploads and not hub.creates


def test_standalone_publish_cli_does_not_generate(tmp_path,monkeypatch):
    import policy_dissertation.data.hub as module
    export(tmp_path);calls=[]
    def fake_publish(path,repo,settings,**kwargs):
        calls.append(kwargs)
        return {'repo_id':repo,'revision':'a'*40,'accepted_subset_only':True}
    monkeypatch.setattr(module,'publish',fake_publish)
    result=CliRunner().invoke(app,['publish_dataset','--input',str(tmp_path),'--repo-id','owner/data','--allow-incomplete-run'])
    assert result.exit_code==0 and 'source run remains' in result.output
    assert calls==[{'allow_incomplete_run':True}]
