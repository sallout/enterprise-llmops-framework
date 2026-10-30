"""Native source-plan preparation/replay; mocked streams are not live Hub evidence."""
import copy
import json
from pathlib import Path
import sqlite3
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.data.training_inputs import TrainingInputGate,index_enterprise
from policy_dissertation.pipelines.enterprise.pipeline import run_enterprise
from policy_dissertation.pipelines.enterprise.preparation import prepare_enterprise
from policy_dissertation.pipelines.native_datasets.pipeline import run_native_datasets
from policy_dissertation.pipelines.native_datasets.replay import iter_prepared_native
from policy_dissertation.errors import DissertationError
from test_pipeline3 import FakeProvider,chat,code
from test_pipeline4 import FakeTokenizer
from test_enterprise import dataset


class SimulatedTokenizer(FakeTokenizer):
    # Synthetic dependency injection only; no real tokenizer certification is claimed.
    info=FakeTokenizer.info | {'validated':True}


@pytest.fixture
def settings():
    return Settings(_env_file=None,pipeline3={'shuffle_buffer':1,'max_attempts':2})


@pytest.fixture
def enterprise(tmp_path,settings):
    old=tmp_path/'old';dataset(old,count=12)
    source=tmp_path/'enterprise'
    run_enterprise(source,settings,code_dataset=old,research_report=tmp_path/'research.md')
    ready=tmp_path/'ready'
    result=prepare_enterprise(source,ready,settings,tokenizer=SimulatedTokenizer())
    assert result['training_ready']
    return ready


def run(tmp_path,enterprise,settings,provider=None,**kwargs):
    return run_native_datasets(enterprise,tmp_path/'native',settings,max_samples=kwargs.pop('max_samples',4),
        tokenizer=kwargs.pop('tokenizer',SimulatedTokenizer()),provider=provider or FakeProvider(),
        research_report=tmp_path/'research.md',**kwargs)


def gate(tmp_path,enterprise,settings):
    index=TrainingInputGate(tmp_path/'replay-gate.sqlite3')
    index_enterprise(enterprise,settings,SimulatedTokenizer(),4096,index)
    index.seal_holdouts()
    return index


def test_total_cap_schemas_licenses_no_native_jsonl_and_repeat_resume(tmp_path,enterprise,settings):
    p=FakeProvider()
    r=run(tmp_path,enterprise,settings,p)
    assert r['accepted_examples']==4 and r['records_processed']==4 and r['data_prepared']
    assert {s['alias']:s['accepted'] for s in r['source_statistics']}=={'nemotron':2,'opencode':2}
    assert not list((tmp_path/'native').rglob('*.jsonl'))
    assert not r['training_started'] and r['llm_calls']==r['locally_executed_external_tests']==0
    plan=json.loads((tmp_path/'native/source_plan.json').read_text())
    assert all(s['license'] and s['revision']==p.revision for s in plan['sources'])
    p2=FakeProvider()
    again=run(tmp_path,enterprise,settings,p2)
    assert again['resumed_decisions']==4 and again['decoded_rows_read_this_run']==0 and not p2.streams
    assert json.loads((tmp_path/'native/source_plan.json').read_text())['decisions_digest']==plan['decisions_digest']


def test_committed_plan_replays_original_tests_and_does_not_reexecute(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings)
    index=gate(tmp_path,enterprise,settings)
    try:
        rows=list(iter_prepared_native(tmp_path/'native',settings,index,tokenizer=SimulatedTokenizer(),provider=FakeProvider()))
        assert len(rows)==len({r['id'] for r in rows})==4
        external=[r for r in rows if r['source']=='nvidia/OpenCodeInstruct']
        assert all(r['verification']['status']=='externally_reported' for r in external)
        assert all(r['verification']['unit_tests']==code()['unit_tests'] for r in external)
        assert all(r['verification']['evidence']['locally_verified'] is False for r in external)
    finally:index.close()


def test_retry_replays_committed_prefix_without_duplicate_decisions(tmp_path,enterprise,settings,monkeypatch):
    import policy_dissertation.pipelines.native_datasets.pipeline as module
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    class Flaky(FakeProvider):
        failed=False
        def stream(self,source):
            if source['alias']=='nemotron' and not self.failed:
                self.failed=True
                def rows():
                    yield chat(0)
                    raise ConnectionError('provider-secret-must-not-appear')
                return rows()
            return super().stream(source)
    r=run(tmp_path,enterprise,settings,Flaky())
    assert r['accepted_examples']==4 and r['stream_failures_this_run']==1 and r['replayed_rows_this_run']==1
    assert 'provider-secret' not in (tmp_path/'native/report.json').read_text()


def test_interruption_recovers_saved_decisions(tmp_path,enterprise,settings):
    class Interrupted(FakeProvider):
        def stream(self,source):
            def rows():
                yield chat()
                raise KeyboardInterrupt()
            return rows()
    first=run(tmp_path,enterprise,settings,Interrupted())
    assert first['status']=='FAILED' and first['accepted_examples']==1
    assert not json.loads((tmp_path/'native/source_plan.json').read_text())['data_prepared']
    result=run(tmp_path,enterprise,settings,FakeProvider())
    assert result['accepted_examples']==4 and result['resumed_decisions']==1 and result['data_prepared']


def test_changed_or_shortened_replay_fails_closed(tmp_path,enterprise,settings):
    class Interrupted(FakeProvider):
        def stream(self,source):
            def rows():
                yield chat()
                raise KeyboardInterrupt()
            return rows()
    run(tmp_path,enterprise,settings,Interrupted())
    changed=FakeProvider(rows={'nemotron':[],'opencode':[code(i) for i in range(5)]})
    r=run(tmp_path,enterprise,settings,changed)
    assert r['status']=='FAILED' and not r['data_prepared']
    assert any('ended before committed' in e for e in r['errors'])


def test_malformed_duplicates_and_caps_have_accurate_counts(tmp_path,enterprise,settings):
    rows=[{'messages':[]},chat(),chat(),chat(1)]
    r=run(tmp_path,enterprise,settings,FakeProvider(rows={'nemotron':rows}),selected=('nemotron',),max_samples=2)
    assert r['records_processed']==4 and r['accepted_examples']==2 and r['rejected_examples']==2
    assert r['rejection_reasons']=={'missing_messages':1,'duplicate_conversation':1}


def test_candidate_shortfall_is_reported_not_invented(tmp_path,enterprise,settings):
    r=run(tmp_path,enterprise,settings,max_records_per_dataset=1)
    assert r['accepted_examples']==2 and r['status']=='COMPLETED_WITH_WARNINGS'
    assert all(s['stop_reason']=='candidate_limit' for s in r['source_statistics'])


def test_all_invalid_source_is_failure_even_if_other_source_valid(tmp_path,enterprise,settings):
    p=FakeProvider(rows={'nemotron':[{'messages':[]}],'opencode':[code(i) for i in range(5)]})
    r=run(tmp_path,enterprise,settings,p)
    assert r['status']=='FAILED' and 'nemotron:no_accepted_examples' in r['errors']


def test_fake_tokenizer_never_certifies_plan(tmp_path,enterprise,settings):
    r=run(tmp_path,enterprise,settings,tokenizer=FakeTokenizer())
    assert not r['data_prepared'] and 'actual_tokenizer_not_validated' in r['warnings']


def test_replay_requires_correct_enterprise_gate_and_unmodified_rows(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings)
    index=TrainingInputGate(tmp_path/'empty.sqlite3');index.seal_holdouts()
    try:
        with pytest.raises(DissertationError,match='same enterprise'):
            list(iter_prepared_native(tmp_path/'native',settings,index,tokenizer=SimulatedTokenizer(),provider=FakeProvider()))
    finally:index.close()
    index=gate(tmp_path,enterprise,settings)
    changed=FakeProvider();changed.rows['nemotron'][0]=chat(99)
    try:
        with pytest.raises(DissertationError,match='content/order changed'):
            list(iter_prepared_native(tmp_path/'native',settings,index,tokenizer=SimulatedTokenizer(),provider=changed))
    finally:index.close()


def test_sqlite_corruption_cannot_be_replayed(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings)
    plan=json.loads((tmp_path/'native/source_plan.json').read_text())
    db=tmp_path/'native'/plan['decision_database']
    with db.open('ab') as stream:stream.write(b'tampered')
    index=gate(tmp_path,enterprise,settings)
    try:
        with pytest.raises(DissertationError,match='checksum'):
            list(iter_prepared_native(tmp_path/'native',settings,index,tokenizer=SimulatedTokenizer(),provider=FakeProvider()))
    finally:index.close()


def test_native_output_cannot_overwrite_existing_data(tmp_path,enterprise,settings):
    folder=tmp_path/'native';folder.mkdir();(folder/'training.jsonl').write_text('preserved')
    with pytest.raises(DissertationError):run(tmp_path,enterprise,settings)
    assert (folder/'training.jsonl').read_text()=='preserved'


def test_cli_and_invalid_options(tmp_path,enterprise,settings,monkeypatch):
    import policy_dissertation.pipelines.native_datasets.pipeline as module
    monkeypatch.setattr(module,'load_tokenizer',lambda *a:SimulatedTokenizer())
    monkeypatch.setattr(module,'HubProvider',lambda *a:FakeProvider())
    monkeypatch.chdir(tmp_path)
    r=CliRunner().invoke(app,['nvidia_data','--enterprise',str(enterprise),'--output',str(tmp_path/'native'),
        '--max-samples','4','--shuffle-buffer','1','--max-records-per-dataset','4'])
    assert r.exit_code==0,r.output
    assert 'Accepted: 4/4' in r.output
    invalid=CliRunner().invoke(app,['nvidia_data','--enterprise',str(enterprise),'--allocation','nemotron=wrong'])
    assert invalid.exit_code==1


def test_published_enterprise_splits_do_not_require_local_generation_report(tmp_path,enterprise,settings):
    (enterprise/'report.json').unlink()
    r=run(tmp_path,enterprise,settings)
    assert r['accepted_examples']==4 and r['data_prepared']


def test_cross_source_holdout_duplicates_are_rejected_not_assigned_to_train(tmp_path,enterprise,settings):
    heldout=json.loads((enterprise/'validation.jsonl').read_text().splitlines()[0])
    p=FakeProvider(rows={'nemotron':[{'messages':heldout['messages']},chat(1),chat(2)],
                         'opencode':[code(i) for i in range(4)]})
    r=run(tmp_path,enterprise,settings,p)
    assert r['accepted_examples']==4 and r['rejection_reasons']['cross_source_split_leakage']==1
    assert r['data_prepared']


def test_closed_reasoning_is_counted_but_not_persisted_as_content(tmp_path,enterprise,settings):
    raw=chat();raw['messages'][-1]['content']='<think>hidden-private-trace</think>Answer 0.'
    r=run(tmp_path,enterprise,settings,FakeProvider(rows={'nemotron':[raw]}),selected=('nemotron',),max_samples=1)
    assert r['normalization_actions']=={'strip_leading_think':1}
    plan=json.loads((tmp_path/'native/source_plan.json').read_text())
    with sqlite3.connect(tmp_path/'native'/plan['decision_database']) as db:
        assert 'hidden-private-trace' not in '\n'.join(db.iterdump())


def test_metadata_failure_and_research_report_failure_never_claim_prepared(tmp_path,enterprise,settings):
    class Denied(FakeProvider):
        def inspect(self,*a,**k):raise PermissionError('credential-must-not-be-printed')
    r=run(tmp_path,enterprise,settings,Denied())
    assert r['status']=='FAILED' and not r['data_prepared']
    assert 'credential-must-not' not in (tmp_path/'native/report.json').read_text()
    # Existing dataset/enterprise inputs remain valid; failed research output invalidates readiness.
    result=run_native_datasets(enterprise,tmp_path/'second',settings,max_samples=4,tokenizer=SimulatedTokenizer(),
        provider=FakeProvider(),research_report=tmp_path)
    assert result['status']=='FAILED' and 'research_report_update_failed' in result['errors']
    assert json.loads((tmp_path/'second/source_plan.json').read_text())['data_prepared'] is False


def test_resume_rejects_changed_row_before_committed_prefix(tmp_path,enterprise,settings):
    class Interrupted(FakeProvider):
        def stream(self,source):
            def rows():
                yield chat()
                raise KeyboardInterrupt()
            return rows()
    run(tmp_path,enterprise,settings,Interrupted())
    p=FakeProvider();p.rows['nemotron'][0]=chat(999)
    r=run(tmp_path,enterprise,settings,p)
    assert r['status']=='FAILED' and any('replay changed' in e for e in r['errors'])


def test_zero_allocation_requires_no_data_stream(tmp_path,enterprise,settings):
    p=FakeProvider()
    r=run(tmp_path,enterprise,settings,p,max_samples=2,allocation={'nemotron':0,'opencode':2})
    assert r['accepted_examples']==2
    assert all(s['alias']=='opencode' for s in p.streams)


def test_finalized_checkpoint_tampering_is_not_silently_resumed(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings)
    plan=json.loads((tmp_path/'native/source_plan.json').read_text())
    with (tmp_path/'native'/plan['decision_database']).open('ab') as stream:stream.write(b'changed')
    r=run(tmp_path,enterprise,settings)
    assert r['status']=='FAILED' and not r['data_prepared']
    assert any('checkpoint file changed' in e for e in r['errors'])


def test_raw_byte_guard_excludes_record_without_persisting_body(tmp_path,enterprise,settings):
    settings.pipeline3.max_record_bytes=1024
    raw=chat();raw['messages'][-1]['content']='x'*1500
    r=run(tmp_path,enterprise,settings,FakeProvider(rows={'nemotron':[raw,chat(1)]}),selected=('nemotron',),max_samples=1)
    assert r['accepted_examples']==1 and r['rejection_reasons']['oversized_source_record']==1


def test_plan_pins_enterprise_tokenizer_and_does_not_mutate_caller_settings(tmp_path,enterprise,settings):
    settings.pipeline4.tokenizer_revision='main'
    run(tmp_path,enterprise,settings)
    plan=json.loads((tmp_path/'native/source_plan.json').read_text())
    assert plan['quality']['tokenizer_revision']=='mock-only'
    assert settings.pipeline4.tokenizer_revision=='main'


def test_replay_rejects_weakened_cross_source_threshold(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings)
    index=TrainingInputGate(tmp_path/'weak-gate.sqlite3',1.0)
    index_enterprise(enterprise,settings,SimulatedTokenizer(),4096,index);index.seal_holdouts()
    try:
        with pytest.raises(DissertationError,match='threshold'):
            list(iter_prepared_native(tmp_path/'native',settings,index,tokenizer=SimulatedTokenizer(),provider=FakeProvider()))
    finally:index.close()


def test_resume_does_not_ignore_revoked_enterprise_readiness(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings)
    path=enterprise/'manifest.json';manifest=json.loads(path.read_text());manifest['training_ready']=False
    path.write_text(json.dumps(manifest))
    with pytest.raises(DissertationError,match='not prepared'):
        run(tmp_path,enterprise,settings)
