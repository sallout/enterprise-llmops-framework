import json
import pytest
from test_native_datasets import enterprise, settings, SimulatedTokenizer, run, gate
from test_pipeline3 import FakeProvider, chat
from policy_dissertation.native_finalization import finalize_native_selection
from policy_dissertation.pipelines.native_datasets.replay import iter_prepared_native
from policy_dissertation.errors import DissertationError


class Interrupted(FakeProvider):
    def stream(self,source):
        yield chat(0)
        raise KeyboardInterrupt()


def test_interrupted_selection_frozen_without_source_replay(tmp_path,enterprise,settings):
    result=run(tmp_path,enterprise,settings,Interrupted(),selected=('nemotron',))
    assert result['status']=='FAILED' and result['accepted_examples']==1
    before=(tmp_path/'native/report.json').read_bytes()
    class Offline(FakeProvider):
        def stream(self,*args):raise AssertionError('Finalization must not stream')
    final=tmp_path/'partial'
    r=finalize_native_selection(tmp_path/'native',enterprise,final,settings,tokenizer=SimulatedTokenizer(),provider=Offline())
    assert r['data_prepared'] and r['accepted_examples']==1 and r['partial_selection']
    assert (tmp_path/'native/report.json').read_bytes()==before
    index=gate(tmp_path,enterprise,settings)
    try:
        rows=list(iter_prepared_native(final,settings,index,tokenizer=SimulatedTokenizer(),provider=FakeProvider()))
        assert len(rows)==1
    finally:index.close()


def test_changed_settings_cannot_approve_old_checkpoint(tmp_path,enterprise,settings):
    run(tmp_path,enterprise,settings,Interrupted(),selected=('nemotron',))
    with pytest.raises(DissertationError,match='fingerprint mismatch'):
        finalize_native_selection(tmp_path/'native',enterprise,tmp_path/'partial',settings,max_tokens=999,tokenizer=SimulatedTokenizer(),provider=FakeProvider())
    assert not json.loads((tmp_path/'partial/source_plan.json').read_text())['data_prepared']
