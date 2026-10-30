"""Interactive navigation/configuration tests; no teacher API, upload or training calls."""
from io import StringIO
import json
from pathlib import Path
import stat
import pytest
from rich.console import Console
from typer.testing import CliRunner

from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.interface.dashboard import Dashboard,path_value,MENU
from policy_dissertation.interface.environment import save_environment,environment_status
from test_enterprise import dataset


class Scripted(Dashboard):
    def __init__(self,choices=(),answers=(),confirmations=(),executor=None):
        self.out=StringIO();self.commands=[]
        super().__init__(console=Console(file=self.out,force_terminal=False,width=110),
                         executor=executor or (lambda args:self.commands.append(args) or 0))
        self.choices=iter(choices);self.answers=iter(answers);self.confirmations=iter(confirmations)
    def choose(self,*args):return next(self.choices)
    def ask(self,*args,**kwargs):return next(self.answers)
    def confirm(self,*args,**kwargs):return next(self.confirmations)


def test_no_argument_launcher_and_explicit_menu_have_branding(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    for args in ([],['menu']):
        r=CliRunner().invoke(app,args,input='0\n')
        assert r.exit_code==0,r.output
        assert 'Bridging the Cusem' in r.output and 'By Mohammed Sallout' in r.output
        assert 'https://www.linkedin.com/in/msallout/' in r.output
        for item in MENU:assert item.label in r.output


def test_unsupported_stages_cannot_start_work(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    ui=Scripted(choices=['security','prompt_injection','customization','exit'])
    ui.run()
    assert not ui.commands and ui.out.getvalue().count('Not implemented yet')==3
    assert not Path('.llmops').exists()


def test_cancelled_preview_never_invokes_pipeline(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs=tmp_path/'docs';dataset(docs,origin='pipeline1')
    code=tmp_path/'code';dataset(code)
    ui=Scripted(choices=['full','existing','existing'],answers=[str(docs),str(code),'4048',str(tmp_path/'output')],
                confirmations=[False,False,False])
    ui.extraction()
    assert not ui.commands and not (tmp_path/'output').exists()


def test_import_only_extraction_reuses_coding_without_llm(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs=tmp_path/'docs';dataset(docs,origin='pipeline1')
    code=tmp_path/'code';dataset(code)
    before=(code/'training.jsonl').read_bytes()
    ui=Scripted(choices=['full','existing','existing'],answers=[str(docs),str(code),'4048',str(tmp_path/'output')],
                confirmations=[False,False,True])
    ui.extraction()
    args=ui.commands[0]
    assert args[0]=='enterprise_data' and '--code-dataset' in args and '--documents-dataset' in args
    assert '--code' not in args and '--documents' not in args and '--skip-preparation' in args
    assert (code/'training.jsonl').read_bytes()==before
    history=Path('.llmops/last_run.json').read_text()
    assert 'api_key' not in history and 'token' not in history


def test_small_document_extraction_forwards_test_limit(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'.env').write_text('MODEL__BASE_URL=https://example.invalid/v1\nMODEL__API_KEY=fake-secret\nMODEL__MODEL_NAME=fake-model\n')
    docs=tmp_path/'docs';docs.mkdir()
    ui=Scripted(choices=['test','folder','skip'],answers=[str(docs),'4048',str(tmp_path/'out')],
                confirmations=[False,False,True])
    ui.extraction()
    args=ui.commands[0]
    assert args[args.index('--limit-documents')+1]=='1'
    assert '--code' not in args and '--code-dataset' not in args
    assert 'fake-secret' not in ui.out.getvalue()


def test_generation_without_llm_configuration_is_blocked(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in ('MODEL__API_KEY','MODEL__BASE_URL','MODEL__MODEL_NAME'):monkeypatch.delenv(key,raising=False)
    docs=tmp_path/'docs';docs.mkdir()
    ui=Scripted(choices=['full','folder','skip'],answers=[str(docs)])
    with pytest.raises(DissertationError,match='Env setup'):ui.extraction()
    assert not ui.commands


def test_wrong_branch_import_is_detected_before_running(tmp_path):
    code=tmp_path/'code';dataset(code)
    docs=tmp_path/'docs';dataset(docs,origin='pipeline1')
    ui=Scripted(answers=[str(code),str(docs)])
    p,_=ui.dataset('Documents','pipeline1',str(code))
    assert p==docs and 'mixed or belongs' in ui.out.getvalue() and not ui.commands


def test_existing_output_requires_explicit_resume_choice(tmp_path):
    (tmp_path/'report.json').write_text('{}')
    ui=Scripted(choices=['resume'],answers=[str(tmp_path)])
    p,overwrite=ui.output(str(tmp_path));assert p==tmp_path and overwrite
    ui=Scripted(choices=['resume'],answers=[str(tmp_path)])
    p,overwrite=ui.output(str(tmp_path),resumable=True);assert not overwrite


def test_external_installation_has_bounded_defaults_and_no_training(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    ready=tmp_path/'ready';ready.mkdir()
    ui=Scripted(choices=['test','both'],answers=[str(ready),str(tmp_path/'native'),'20','4096','30','42','2'],
                confirmations=[False,True])
    ui.external_installation()
    args=ui.commands[0]
    assert args[0]=='nvidia_data'
    assert args[args.index('--max-samples')+1]=='20'
    assert args.count('--dataset')==2 and '--max-records-per-dataset' in args
    assert '--max-tokens' in args


def test_external_validation_uses_read_only_validation_action(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan=tmp_path/'plan';plan.mkdir();ready=tmp_path/'ready';ready.mkdir()
    ui=Scripted(answers=[str(plan),str(ready),str(tmp_path/'validation')],confirmations=[False,True])
    ui.external_validation()
    assert ui.commands[0][0]=='nvidia_validate' and '--limit-records' not in ui.commands[0]
    assert not (plan/'report.json').exists()


def test_failed_executor_is_not_reported_as_success(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    ui=Scripted(executor=lambda _:1)
    assert ui.execute(['enterprise_prepare'],tmp_path/'out')==1
    assert 'Operation failed' in ui.out.getvalue()
    assert json.loads(Path('.llmops/last_run.json').read_text())['exit_code']==1


def test_keyboard_menu_backend_uses_arrows_and_disables_unimplemented_options(monkeypatch):
    import policy_dissertation.interface.dashboard as module
    ui=Scripted();ui.terminal=True
    captured={}
    class Answer:
        def unsafe_ask(self):return 'exit'
    def select(*a,**k):captured.update(k);return Answer()
    monkeypatch.setattr(module.questionary,'select',select)
    assert Dashboard.choose(ui,'Menu',MENU)=='exit'
    assert captured['use_shortcuts'] is True
    assert all(choice.disabled for choice in captured['choices'] if choice.value in ('security','prompt_injection','customization'))
    assert any(choice.value=='evaluation' and not choice.disabled for choice in captured['choices'])


def test_secret_entry_refuses_non_tty_to_avoid_echo():
    with pytest.raises(DissertationError,match='interactive terminal'):Scripted().secret('API key')


def test_environment_presence_never_displays_secret_values(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    secret='super-private-test-value'
    (tmp_path/'.env').write_text('MODEL__API_KEY='+secret+'\n')
    ui=Scripted(choices=['status']);ui.environment()
    assert secret not in ui.out.getvalue() and 'not verified' in ui.out.getvalue()


def test_env_save_backups_permissions_and_unrelated_settings(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    original='# Important comment\nCUSTOM_SETTING=keep-me\nMODEL__API_KEY=old-private\n'
    path=tmp_path/'.env';path.write_text(original)
    result=save_environment(path,{'MODEL__API_KEY':'new-private-secret','HF_TOKEN':'fake-hf-secret'})
    assert Path(result['backup']).read_text()==original
    assert stat.S_IMODE(path.stat().st_mode)==0o600
    assert stat.S_IMODE(Path(result['backup']).stat().st_mode)==0o600
    assert '# Important comment' in path.read_text() and 'CUSTOM_SETTING=keep-me' in path.read_text()
    s=Settings(_env_file=path)
    assert s.model.api_key.get_secret_value()=='new-private-secret'
    assert 'new-private-secret' not in json.dumps(result)


@pytest.mark.parametrize('changes',[
    {'UNKNOWN_SECRET':'value'}, {'MODEL__API_KEY':'bad\nvalue'}, {'MODEL__BASE_URL':'not-a-url'},
    {'HF_TOKEN':'${EXPOSE_VARIABLE}'}, {'LOG_LEVEL':'bad-level'}])
def test_bad_env_changes_preserve_existing_file(tmp_path,changes):
    path=tmp_path/'.env';path.write_text('HF_TOKEN=original\n')
    before=path.read_bytes()
    with pytest.raises(DissertationError):save_environment(path,changes)
    assert path.read_bytes()==before and not list(tmp_path.glob('.env.backup-*'))


def test_env_save_is_not_hidden_by_process_overrides(tmp_path,monkeypatch):
    monkeypatch.setenv('LOG_LEVEL','invalid-active-value')
    result=save_environment(tmp_path/'.env',{'LOG_LEVEL':'INFO'})
    assert result['process_overrides_present']
    assert 'invalid-active-value' not in json.dumps(result)


def test_path_input_supports_spaces_quoted_and_dragged_paths(tmp_path):
    p=tmp_path/'folder with spaces';p.mkdir()
    assert path_value('"'+str(p)+'"')==p
    assert path_value(str(p).replace(' ','\\ '))==p


def test_invalid_menu_input_and_eof_do_not_start_jobs(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    r=CliRunner().invoke(app,[],input='99\n0\n')
    assert r.exit_code==0 and not Path('.llmops').exists()
    r=CliRunner().invoke(app,[],input='')
    assert r.exit_code==0


def test_menu_can_execute_existing_prepare_command_without_recursion(tmp_path,monkeypatch):
    from test_native_datasets import SimulatedTokenizer
    from policy_dissertation.pipelines.enterprise.pipeline import run_enterprise
    import policy_dissertation.pipelines.dataset_preparation.pipeline as module
    old=tmp_path/'old';dataset(old,count=8)
    accepted=tmp_path/'accepted'
    run_enterprise(accepted,Settings(_env_file=None),code_dataset=old,research_report=tmp_path/'research.md')
    monkeypatch.setattr(module,'load_tokenizer',lambda *a:SimulatedTokenizer())
    monkeypatch.chdir(tmp_path)
    inputs='2\n1\n'+str(accepted)+'\n'+str(tmp_path/'ready')+'\n4096\n42\ny\n0\n'
    result=CliRunner().invoke(app,[],input=inputs)
    assert result.exit_code==0,result.output
    report=json.loads((tmp_path/'ready/report.json').read_text())
    assert report['accepted_examples']==8 and report['training_ready']
    assert json.loads(Path('.llmops/last_run.json').read_text())['exit_code']==0


def test_successful_outputs_become_defaults_without_credentials(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    ready=tmp_path/'custom-ready';ready.mkdir()
    ui=Scripted()
    ui.execute(['enterprise_prepare'],ready)
    assert ui.suggested('enterprise_prepare','fallback')==str(ready)
    state=Path('.llmops/outputs.json').read_text()
    assert 'api_key' not in state and 'hf_token' not in state


def test_history_failure_does_not_turn_successful_processing_into_failure(tmp_path,monkeypatch):
    import policy_dissertation.interface.dashboard as module
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module,'dump_json',lambda *a:(_ for _ in ()).throw(OSError()))
    ui=Scripted()
    assert ui.execute(['enterprise_prepare'],tmp_path/'out')==0
    assert 'result above is unchanged' in ui.out.getvalue()


def test_tuning_defaults_has_bounds_and_does_not_process_data(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    ui=Scripted(choices=['generation'],answers=['4096','16384','','','3',''],confirmations=[True])
    ui.tune_environment()
    settings=Settings()
    assert settings.pipeline1.response_max_tokens==4096 and settings.pipeline2.response_max_tokens==16384
    assert not ui.commands
    assert 'invalidate checkpoints' in ui.out.getvalue()


def test_back_keyword_cancels_a_form_without_running_work():
    from policy_dissertation.interface.dashboard import Cancelled
    class Input:
        def unsafe_ask(self):return ':back'
    import unittest.mock
    ui=Scripted();ui.terminal=True
    with unittest.mock.patch('policy_dissertation.interface.dashboard.questionary.text',return_value=Input()):
        with pytest.raises(Cancelled):Dashboard.ask(ui,'Path')
    assert not ui.commands


def test_logging_setup_can_repair_invalid_dotenv_without_printing_value(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'.env').write_text('LOG_LEVEL=private-invalid-level\n')
    ui=Scripted(choices=['logging','INFO'],confirmations=[True])
    ui.tune_environment()
    assert Settings().log_level=='INFO'
    assert 'private-invalid-level' not in ui.out.getvalue()


def test_terminal_menu_uses_temporary_screen_and_waits_before_return(monkeypatch):
    from contextlib import contextmanager
    ui=Scripted(choices=['enterprise_validation','exit']);ui.terminal=True
    events=[]
    @contextmanager
    def screen(**kwargs):
        assert kwargs=={'hide_cursor':False}
        events.append('enter')
        try:yield
        finally:events.append('restore')
    monkeypatch.setattr(ui.console,'screen',screen)
    monkeypatch.setattr(ui,'enterprise_validation',lambda:events.append('result'))
    monkeypatch.setattr(ui,'pause',lambda:events.append('wait'))
    ui.run()
    assert events==['enter','result','wait','restore']


def test_keyboard_page_redraw_clears_completed_page(monkeypatch):
    import policy_dissertation.interface.dashboard as module
    ui=Scripted();ui.terminal=True;events=[]
    monkeypatch.setattr(ui.console,'clear',lambda:events.append('clear'))
    monkeypatch.setattr(ui,'header',lambda:events.append('header'))
    class Answer:
        def unsafe_ask(self):return 'exit'
    monkeypatch.setattr(module.questionary,'select',lambda *a,**k:Answer())
    assert Dashboard.choose(ui,'Menu',MENU)=='exit'
    assert events==['clear','header']


def test_redirected_menu_never_clears_screen_or_waits(monkeypatch):
    ui=Scripted(choices=['exit'])
    def forbidden(*a,**k):raise AssertionError('No terminal controls in piped output')
    monkeypatch.setattr(ui.console,'clear',forbidden)
    monkeypatch.setattr(ui.console,'screen',forbidden)
    ui.run()
    assert 'Goodbye' in ui.out.getvalue()
