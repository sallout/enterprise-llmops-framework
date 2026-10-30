"""Keyboard-driven terminal front end; commands remain the execution authority."""
from dataclasses import dataclass
from contextlib import nullcontext
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import sys
import time
from collections.abc import Callable

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt,Confirm
from rich.table import Table
from rich.text import Text
from pydantic import ValidationError

from policy_dissertation.config import Settings,InterfaceSettings
from policy_dissertation.errors import DissertationError
from policy_dissertation.interface.environment import environment_status,save_environment
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json


@dataclass(frozen=True)
class Option:
    key:str
    label:str
    enabled:bool=True


MENU=(Option('enterprise','Enterprise data extraction'),Option('enterprise_validation','Enterprise data validation'),
      Option('external','External dataset installation'),Option('external_validation','External dataset validation'),
      Option('fine_tuning','Model compression & fine-tuning'),Option('security','Security check',False),
      Option('prompt_injection','Prompt injection check',False),Option('customization','Model customization',False),
      Option('environment','Env setup'),Option('exit','Exit'))


class Cancelled(Exception):pass


def execute_command(arguments):
    # No shell execution, interpolation, or separate copies of pipeline logic.
    from typer.main import get_command
    from policy_dissertation.cli import app
    result=get_command(app).main(args=arguments,prog_name='llmops',standalone_mode=False)
    return result if type(result) is int else 0


def path_value(value):
    value=value.strip()
    if len(value)>1 and value[0]==value[-1] and value[0] in ('"',"'"):value=value[1:-1]
    direct=Path(value).expanduser()
    if direct.exists():return direct
    try:parts=shlex.split(value)
    except ValueError:parts=[]
    if len(parts)==1 and Path(parts[0]).expanduser().exists():return Path(parts[0]).expanduser()
    return direct


class Dashboard:
    def __init__(self,*,console=None,executor:Callable|None=None):
        self.console=console or Console(highlight=False)
        self.executor=executor or execute_command
        self.terminal=bool(self.console.is_terminal and sys.stdin.isatty())

    def header(self):
        try:branding=Settings().ui
        except (ValueError,ValidationError):branding=InterfaceSettings()
        text=Text(branding.title,style='bold cyan')
        text.append('\nBy '+branding.author,style='not bold default')
        url=str(branding.linkedin_url)
        text.append('\n'+url,style='link '+url)
        self.console.print(Panel(text,border_style='cyan',subtitle='Data preparation workspace · no training is started automatically'))

    def choose(self,title,options):
        options=list(options)
        if self.terminal:
            self.console.clear()
            self.header()
            choices=[questionary.Choice(o.label,value=o.key,disabled=None if o.enabled else 'Not implemented',
                       shortcut_key='0' if o.key=='exit' else str(i+1)) for i,o in enumerate(options)]
            return questionary.select(title,choices=choices,use_shortcuts=True,use_indicator=True,
                use_jk_keys=False,pointer='›',instruction='Arrow keys / number, then Enter · Ctrl+C cancels').unsafe_ask()
        for i,o in enumerate(options):
            key='0' if o.key=='exit' else str(i+1)
            self.console.print(Text(f'[{key}] {o.label}'+(' — not implemented' if not o.enabled else ''),style='default' if o.enabled else 'dim'))
        keys={('0' if o.key=='exit' else str(i+1)):o.key for i,o in enumerate(options)}
        value=Prompt.ask(title,choices=list(keys),show_choices=False,console=self.console)
        return keys[value]

    def ask(self,title,default=''):
        value=questionary.text(title,default=str(default)).unsafe_ask().strip() if self.terminal else Prompt.ask(title,default=str(default),console=self.console).strip()
        if value==':back':raise Cancelled()
        return value

    def secret(self,title):
        if not self.terminal:
            raise DissertationError('Hidden credential entry requires an interactive terminal. Run llmops directly, without piping input.')
        value=questionary.password(title).unsafe_ask().strip()
        if value==':back':raise Cancelled()
        return value

    def confirm(self,title,default=False):
        if self.terminal:return bool(questionary.confirm(title,default=default).unsafe_ask())
        return Confirm.ask(title,default=default,console=self.console)

    def path(self,title,*,default='',directory=False,optional=False):
        while True:
            if self.terminal:
                value=questionary.path(title,default=str(default),only_directories=directory).unsafe_ask().strip()
            else:value=self.ask(title,default)
            if value==':back':raise Cancelled()
            if not value and optional:return None
            if not value:
                self.console.print('A path is required. Type :back to cancel.',style='yellow');continue
            p=path_value(value)
            if not p.exists() or (directory and not p.is_dir()):
                self.console.print('Path does not exist or is not the required directory.',style='yellow');continue
            return p

    def integer(self,title,default,*,optional=False,minimum=1,maximum=None):
        while True:
            value=self.ask(title,'' if default is None else str(default))
            if value==':back':raise Cancelled()
            if optional and not value:return None
            try:
                n=int(value)
                if n<minimum or (maximum is not None and n>maximum):raise ValueError()
                return n
            except ValueError:self.console.print('Enter a valid integer within the allowed range.',style='yellow')

    def output(self,default,*,resumable=False):
        while True:
            value=self.ask('Output directory',default)
            if value==':back':raise Cancelled()
            if not value:self.console.print('Output directory is required.',style='yellow');continue
            p=path_value(value)
            if p.exists() and not p.is_dir():self.console.print('Choose a directory path.',style='yellow');continue
            populated=p.exists() and any((p/name).exists() for name in ('report.json','training.jsonl','source_plan.json','manifest.json'))
            if populated:
                action=self.choose('Existing output found',[
                    Option('new','Choose a new output folder'),
                    Option('resume','Resume/rewrite exports explicitly (source inputs stay protected)'),
                    Option('back','Back')])
                if action=='new':continue
                if action=='back':raise Cancelled()
                # Native preparation owns fingerprinted recovery; other commands need --overwrite.
                return p,not resumable
            return p,False

    def dataset(self,label,origin,default):
        while True:
            p=self.path(label,default=default)
            file=p/'training.jsonl' if p.is_dir() else p
            report=file.parent/'report.json'
            if not file.is_file() or not report.is_file():
                self.console.print('Choose accepted JSONL with its adjacent report.json.',style='yellow');continue
            try:
                kinds=set()
                with file.open(encoding='utf-8') as stream:
                    for line in stream:
                        if len(line.encode())>16777216:raise ValueError()
                        row=json.loads(line)
                        kind=row.get('metadata',{}).get('pipeline') if row.get('schema_version')=='unified-qwen-sft-v1' else {
                            'pipeline1-extractive-v1':'pipeline1','repository-data-v1':'pipeline2'}.get(row.get('schema_version'))
                        kinds.add(kind)
                if kinds!={origin}:
                    self.console.print('This dataset is mixed or belongs to the other branch. Use Enterprise data validation for an already combined dataset.',style='yellow');continue
                producer=json.loads(report.read_text())
            except (OSError,ValueError,TypeError,AttributeError):
                self.console.print('Could not identify this accepted dataset. Choose a valid branch export.',style='yellow');continue
            incomplete=(producer.get('status') not in ('COMPLETED','COMPLETED_WITH_WARNINGS') or
                        bool(producer.get('errors')) or producer.get('source_run_complete') is False)
            if incomplete and not self.confirm('The producer is failed/incomplete. Reuse only its validated accepted subset, preserving that warning?'):
                raise Cancelled()
            return p,incomplete

    def preview(self,title,values):
        table=Table(show_header=False,box=None,padding=(0,2))
        for key,value in values.items():table.add_row(Text(str(key),style='cyan'),Text(str(value)))
        self.console.print(Panel(table,title=title,border_style='blue'))
        return self.confirm('Run this operation now?',False)

    def suggested(self,operation,fallback):
        try:
            outputs=json.loads(Path('.llmops/outputs.json').read_text())
            value=outputs.get(operation) or (outputs.get('download_dataset') if operation=='enterprise_prepare' else None)
            if isinstance(value,str) and Path(value).is_dir():return value
        except (OSError,ValueError,AttributeError):pass
        if operation=='nvidia_data':
            # Discover top-level native plans only; never scan raw dataset bodies/caches.
            candidates=list(Path('artifacts').glob('*/source_plan.json')) if Path('artifacts').exists() else []
            for path in sorted(candidates,key=lambda p:p.stat().st_mtime,reverse=True):
                try:
                    plan=json.loads(path.read_text())
                    if plan.get('schema_version')=='native-nvidia-selection-v1' and plan.get('data_prepared'):
                        return str(path.parent)
                except (OSError,ValueError,AttributeError):continue
        return fallback

    def summary(self,output):
        path=Path(output)/'report.json'
        if not path.is_file():return
        try:r=json.loads(path.read_text())
        except (OSError,ValueError):return
        allowed=('status','accepted_examples','rejected_examples','records_processed','records_checked',
                 'duplicates_removed','training_ready','data_prepared','full_plan_validated','source_run_complete','resumed_decisions',
                 'final_model_ready','original_parameters','pruned_parameters','evaluation_pending')
        table=Table(show_header=False,box=None)
        for k in allowed:
            value=r.get(k)
            if type(value) in (int,bool) or (k=='status' and value in ('COMPLETED','COMPLETED_WITH_WARNINGS','FAILED','RUNNING','PREFLIGHT_PASSED','INTERRUPTED')):
                table.add_row(k.replace('_',' ').capitalize(),str(value))
        splits=r.get('splits')
        if isinstance(splits,dict):
            for key in ('train','validation','test'):
                if type(splits.get(key)) is int:table.add_row(key.capitalize()+' examples',str(splits[key]))
        self.console.print(table)
        self.console.print(Text('Report: '+str(path.resolve()),style='dim'))

    def execute(self,args,output):
        started=time.monotonic()
        try:code=self.executor(args)
        except KeyboardInterrupt:
            code=130;self.console.print('Interrupted. Committed checkpoints are preserved.',style='yellow')
        except Exception:
            code=1;self.console.print('Operation failed. Review configuration and the output report; no raw provider errors are displayed.',style='red')
        code=code if type(code) is int else 0
        warning=False
        if code==0 and args[0]!='publish_dataset':
            try:warning=json.loads((Path(output)/'report.json').read_text()).get('status')=='COMPLETED_WITH_WARNINGS'
            except (OSError,ValueError,AttributeError):pass
        self.console.print(('Operation completed with warnings' if warning else 'Operation completed') if code==0 else f'Operation failed / interrupted (exit {code})',
                           style='yellow' if warning else 'green' if code==0 else 'red')
        if args[0]=='publish_dataset':
            if code==0:self.console.print('Publication completed; the original producer status remains unchanged.')
        else:self.summary(output)
        # Never persist credentials or typed environment values in menu history.
        try:
            dump_json(Path('.llmops/last_run.json'),{'operation':args[0],'output_directory':str(Path(output).resolve()),
                'exit_code':code,'duration_seconds':round(time.monotonic()-started,2),
                'finished_at':datetime.now(timezone.utc).isoformat()})
            if code==0:
                try:outputs=json.loads(Path('.llmops/outputs.json').read_text())
                except (OSError,ValueError):outputs={}
                allowed={'enterprise_data','enterprise_prepare','nvidia_data','nvidia_validate','publish_dataset','download_dataset','compress_model'}
                outputs={k:v for k,v in outputs.items() if k in allowed and isinstance(v,str)} if isinstance(outputs,dict) else {}
                outputs[args[0]]=str(Path(output).resolve())
                dump_json(Path('.llmops/outputs.json'),outputs)
        except OSError:
            self.console.print('Operation result above is unchanged; menu history could not be saved.',style='yellow')
        return code

    def extraction(self):
        mode=self.choose('Enterprise extraction scope',[
            Option('test','Small local test (recommended)'),Option('full','Full selected sources'),Option('custom','Custom generation limits'),Option('back','Back')])
        if mode=='back':raise Cancelled()
        args=['enterprise_data'];raw=False;incomplete=False;selected={}
        for label,flag,origin,folder,existing in (
            ('Document','documents','pipeline1','dumy_policies','artifacts/pipeline1'),
            ('Code','code','pipeline2','code_repos','artifacts/pipeline2')):
            options=[Option('existing','Reuse accepted '+label.lower()+' dataset (no generation)'),
                     Option('folder','Read '+label.lower()+' source folder'),Option('skip','Skip '+label.lower()+' input')]
            choice=self.choose(label+' input',options)
            if choice=='skip':continue
            if choice=='existing':
                p,partial=self.dataset('Accepted '+label.lower()+' dataset path',origin,existing)
                args+=['--'+flag+'-dataset',str(p)];incomplete|=partial
                selected[label]=str(p)+' (existing accepted records)'
            else:
                p=self.path('Document folder [PDF, MD, TXT]' if flag=='documents' else 'Code repository folder',default=folder,directory=True)
                args+=['--'+flag,str(p)];raw=True;selected[label]=str(p)+' (new generation)'
                n=1 if flag=='documents' else 2
                limit=n if mode=='test' else self.integer(label+' generation limit (blank = all)',None,optional=True) if mode=='custom' else None
                if limit:args+=['--limit-documents' if flag=='documents' else '--limit-units',str(limit)]
        if not selected:raise DissertationError('Select at least one document or code input.')
        if raw:
            status=environment_status()
            if not all(status.get(k) for k in ('configuration_valid','llm_endpoint','llm_api_key','llm_model')):
                raise DissertationError('New generation requires LLM endpoint, API key and model name. Open Env setup first.')
        maximum=self.integer('Max source tokens (4048 suggested; not the model response budget)',4048)
        output,overwrite=self.output('artifacts/enterprise_test' if mode=='test' else 'artifacts/enterprise_extracted')
        args+=['--output',str(output),'--max-tokens',str(maximum)]
        if incomplete:args.append('--allow-incomplete-import')
        if overwrite:args.append('--overwrite')
        prepare=self.confirm('Also prepare final train/validation/test splits now? (Otherwise use Enterprise data validation)',False)
        if prepare:
            token_limit=self.integer('Exact Qwen conversation token limit',4096)
            seed=self.integer('Split seed',42,minimum=-2147483648)
            args+=['--training-max-tokens',str(token_limit),'--seed',str(seed)]
        else:args.append('--skip-preparation')
        if self.confirm('Configure repository task types?',False):
            from policy_dissertation.pipelines.repository_data.schemas import TASK_TYPES
            value=self.ask('Comma-separated task types',','.join(TASK_TYPES))
            kinds=[v.strip() for v in value.split(',') if v.strip()]
            if not kinds or any(v not in TASK_TYPES for v in kinds):raise DissertationError('Unsupported repository task type.')
            args+=['--task-types',','.join(kinds)]
        if self.preview('Enterprise extraction',selected|{'Source token bound':maximum,'Output':output,
            'Preparation':'Included' if prepare else 'Separate validation step','LLM API calls':'Yes, for new generation' if raw else 'No; imports only',
            'GPU / model training':'Not used'}):self.execute(args,output)

    def enterprise_validation(self):
        action=self.choose('Enterprise data validation',[
            Option('prepare','Validate accepted enterprise data and prepare splits'),
            Option('publish','Publish an existing accepted/prepared dataset privately to Hugging Face'),
            Option('download','Download an existing enterprise dataset from Hugging Face'),
            Option('inspect','Inspect a local report summary'),Option('back','Back')])
        if action=='back':raise Cancelled()
        if action=='download':
            repo=self.ask('Hugging Face dataset repository (owner/name)')
            revision=self.ask('Dataset revision (branch/tag/commit)', 'main')
            output,overwrite=self.output('artifacts/enterprise_ready')
            args=['download_dataset','--repo-id',repo,'--output',str(output),'--revision',revision]
            if overwrite:args.append('--overwrite')
            if self.preview('Restore existing enterprise dataset',{'Repository':repo,'Revision':revision,'Output':output,
                'Checks':'Checksums, schema, counts and split membership','Generation / LLM calls':'None'}):self.execute(args,output)
            return
        default=self.suggested('enterprise_prepare','artifacts/enterprise_ready') if action=='publish' else self.suggested('enterprise_data','artifacts/enterprise_combined')
        source=self.path('Enterprise artifact directory',default=default,directory=True)
        if action=='inspect':self.summary(source);return
        if action=='publish':
            if not environment_status().get('hugging_face_token'):raise DissertationError('Configure HF_TOKEN in Env setup before publishing.')
            repo=self.ask('Private Hugging Face repository (owner/name)')
            if repo.count('/')!=1 or any(v in repo for v in ('YOUR_USERNAME','OWNER/')):
                raise DissertationError('Enter your actual owner/dataset repository name.')
            args=['publish_dataset','--input',str(source),'--repo-id',repo]
            if self.confirm('Allow publishing only the accepted subset if source runs are incomplete?',False):args.append('--allow-incomplete-run')
            if self.preview('Private dataset publication',{'Input':source,'Repository':repo,'Uploads':'Accepted records + split files/card/manifest only','New generation':'None'}):
                self.execute(args,source)
            return
        output,overwrite=self.output('artifacts/enterprise_ready')
        maximum=self.integer('Exact Qwen conversation token limit',4096)
        seed=self.integer('Split seed',42,minimum=-2147483648)
        args=['enterprise_prepare','--input',str(source),'--output',str(output),'--max-tokens',str(maximum),'--seed',str(seed)]
        if overwrite:args.append('--overwrite')
        if self.preview('Enterprise validation / splitting',{'Input':source,'Output':output,'Exact Qwen token bound':maximum,
            'New generation / LLM calls':'None','GPU training':'Not started'}):self.execute(args,output)

    def external_installation(self):
        mode=self.choose('External preparation scope',[
            Option('test','Small local test: 20 accepted examples'),Option('server','Server-sized sample: 10000 accepted examples'),
            Option('custom','Custom sample size'),Option('back','Back')])
        if mode=='back':raise Cancelled()
        enterprise=self.path('Prepared enterprise directory',default=self.suggested('enterprise_prepare','artifacts/enterprise_ready'),directory=True)
        output,_=self.output('artifacts/nvidia_local' if mode=='test' else 'artifacts/nvidia',resumable=True)
        cached={}
        try:cached=json.loads((output/'source_plan.json').read_text()) if (output/'source_plan.json').exists() else {}
        except (OSError,ValueError):pass
        which=self.choose('NVIDIA datasets',[
            Option('both','Both NVIDIA datasets'),Option('nemotron','Nemotron only'),Option('opencode','OpenCodeInstruct only')])
        aliases=['nemotron','opencode'] if which=='both' else [which]
        cap=self.integer('Total accepted examples across selected datasets',cached.get('accepted_examples') or (20 if mode=='test' else 10000))
        maximum=self.integer('Exact Qwen conversation token limit',cached.get('max_tokens',4096))
        candidates=self.integer('Candidate limit per dataset (blank = automatic)',30 if mode=='test' else None,optional=True)
        seed=self.integer('Sampling seed',cached.get('seed',42),minimum=-2147483648)
        buffer=self.integer('Prefix shuffle buffer (not whole-corpus uniform sampling)',cached.get('normalization',{}).get('shuffle_buffer',2 if mode=='test' else 16),maximum=128)
        args=['nvidia_data','--enterprise',str(enterprise),'--output',str(output),'--max-samples',str(cap),
              '--max-tokens',str(maximum),'--seed',str(seed),'--shuffle-buffer',str(buffer)]
        for alias in aliases:args+=['--dataset',alias]
        if candidates:args+=['--max-records-per-dataset',str(candidates)]
        if self.confirm('Configure dataset revisions, configurations and allocations?',False):
            remaining=cap
            for i,alias in enumerate(aliases):
                configuration=self.ask(alias+' configuration','chat' if alias=='nemotron' else 'train')
                args+=['--'+alias+'-config',configuration]
                pin=self.ask(alias+' immutable revision (blank = reuse first cached pin)')
                if pin:args+=['--revision',alias+'='+pin]
                n=self.integer(alias+' accepted allocation',remaining//(len(aliases)-i),minimum=0)
                remaining-=n;args+=['--allocation',alias+'='+str(n)]
            if remaining!=0:raise DissertationError('Dataset allocations must sum to the total accepted cap.')
        if self.preview('External dataset installation',{'Enterprise splits':enterprise,'Sources':', '.join(aliases),'Accepted cap':cap,
            'Output':output,'Storage':'Native source plan + fingerprints; no full NVIDIA JSONL copy',
            'Resume':'Automatic for matching source/settings fingerprint','LLM / GPU':'Not used'}):self.execute(args,output)

    def external_validation(self):
        plan=self.path('Native source-plan directory',default=self.suggested('nvidia_data','artifacts/nvidia_local'),directory=True)
        enterprise=self.path('Prepared enterprise directory',default=self.suggested('enterprise_prepare','artifacts/enterprise_ready'),directory=True)
        output,overwrite=self.output('artifacts/nvidia_validation')
        limit=None
        if self.confirm('Validate only a bounded sample instead of every approved record?',False):
            limit=self.integer('Maximum approved records to check',20)
        args=['nvidia_validate','--plan',str(plan),'--enterprise',str(enterprise),'--output',str(output)]
        if limit:args+=['--limit-records',str(limit)]
        if overwrite:args.append('--overwrite')
        if self.preview('External dataset validation',{'Source plan':plan,'Enterprise splits':enterprise,'Output':output,
            'Scope':f'First {limit} approved records only' if limit else 'All approved source-plan records',
            'Checks':'Source/ID/token hashes, schema, original tests metadata and fresh leakage gate',
            'LLM / GPU training':'Not used'}):self.execute(args,output)

    def environment(self):
        action=self.choose('Environment setup',[
            Option('status','Check configuration presence (no endpoint calls)'),Option('edit','Create/update .env securely'),
            Option('tune','Tune pipeline response budgets, timeouts and retries'),Option('back','Back')])
        if action=='back':raise Cancelled()
        status=environment_status()
        self.preview_status(status)
        if action=='status':return
        if action=='tune':
            self.tune_environment();return
        self.console.print('Existing values stay unchanged when an answer is blank. Existing secrets are never displayed.',style='dim')
        changes={}
        for key,label,hidden in (
            ('MODEL__BASE_URL','LLM base URL (hidden; blank keeps existing)',True),
            ('MODEL__MODEL_NAME','LLM model name (blank keeps existing)',False),
            ('MODEL__API_KEY','LLM API key (hidden; blank keeps existing)',True),
            ('HF_TOKEN','Hugging Face token (hidden; blank keeps existing)',True)):
            value=self.secret(label) if hidden else self.ask(label)
            if value:changes[key]=value
        if self.confirm('Change console branding?',False):
            for key,label in (('UI__TITLE','Console title'),('UI__AUTHOR','Author name'),('UI__LINKEDIN_URL','LinkedIn URL')):
                value=self.ask(label+' (blank keeps existing)')
                if value:changes[key]=value
        if not changes:self.console.print('No changes requested.');return
        self.console.print(Text('Keys to update: '+', '.join(sorted(changes)),style='cyan'))
        if not self.confirm('Save .env now? An existing file will be backed up with owner-only permissions.',False):raise Cancelled()
        result=save_environment(Path('.env'),changes)
        self.console.print('Environment saved. Secret values remain hidden.',style='green')
        if result['backup']:self.console.print(Text('Private backup: '+result['backup'],style='dim'))
        if result['process_overrides_present']:self.console.print('Active process environment variables override .env; adjust those overrides if needed.',style='yellow')

    def tune_environment(self):
        group=self.choose('Pipeline defaults',[
            Option('generation','Enterprise generation (response budget is separate from source tokens)'),
            Option('external','External streaming retries/timeouts/candidate scan'),
            Option('logging','Logging level'),Option('back','Back')])
        if group=='back':raise Cancelled()
        if group=='logging':
            level=self.choose('Logging level',[Option(v,v) for v in ('INFO','WARNING','ERROR','DEBUG','CRITICAL')])
            if self.preview('Save logging preference',{'LOG_LEVEL':level}):
                result=save_environment(Path('.env'),{'LOG_LEVEL':level})
                self.console.print('Logging preference saved.',style='green')
                if result['process_overrides_present']:self.console.print('An active environment variable overrides this file preference.',style='yellow')
            return
        if group=='generation':
            fields=(('PIPELINE1__RESPONSE_MAX_TOKENS','Document LLM response budget',256,32768),
                    ('PIPELINE2__RESPONSE_MAX_TOKENS','Code LLM response budget',256,32768),
                    ('PIPELINE1__REQUEST_TIMEOUT','Document endpoint timeout seconds',1,600),
                    ('PIPELINE2__REQUEST_TIMEOUT','Code endpoint timeout seconds',1,600),
                    ('PIPELINE1__MAX_ATTEMPTS','Document bounded attempts',1,5),
                    ('PIPELINE2__MAX_ATTEMPTS','Code bounded attempts',1,5))
        else:
            fields=(('PIPELINE3__MAX_ATTEMPTS','External bounded attempts',1,5),
                    ('PIPELINE3__REQUEST_TIMEOUT','External request timeout seconds',1,120),
                    ('PIPELINE3__SCAN_MULTIPLIER','Candidate scan multiplier',1,100))
        changes={}
        for key,label,minimum,maximum in fields:
            value=self.integer(label+' (blank keeps existing)',None,optional=True,minimum=minimum,maximum=maximum)
            if value is not None:changes[key]=str(value)
        if not changes:self.console.print('No changes requested.');return
        self.console.print('Changed generation defaults can invalidate checkpoints. Reuse accepted code datasets to avoid regeneration.',style='yellow')
        if self.preview('Save pipeline defaults (no data processing)',changes):
            result=save_environment(Path('.env'),changes)
            self.console.print('Pipeline defaults saved; no API calls were made.',style='green')
            if result['process_overrides_present']:self.console.print('Active environment variables may override the saved defaults.',style='yellow')
            if result['backup']:self.console.print(Text('Private backup: '+result['backup'],style='dim'))

    def compression(self):
        action=self.choose('Linux / NVIDIA CUDA model compression',[
            Option('check','Check GPU environment only — no model download'),
            Option('run','Prune, recover/train and export 4-bit model'),Option('back','Back')])
        if action=='back':raise Cancelled()
        enterprise=self.path('Prepared enterprise directory',default=self.suggested('enterprise_prepare','artifacts/enterprise_ready'),directory=True)
        native=self.path('Prepared NVIDIA source-plan directory',default=self.suggested('nvidia_data','artifacts/nvidia'),directory=True)
        output,_=self.output('artifacts/compression',resumable=True)
        args=['compress_model','--enterprise',str(enterprise),'--native-plan',str(native),'--output',str(output)]
        if (output/'checkpoint.json').exists() or (output/'report.json').exists():args.append('--resume')
        if action=='check':args.append('--check-only')
        else:
            value=self.ask('MLP channel pruning fraction (0 < value <= 0.20; not total parameter reduction)','0.10')
            try:ratio=float(value)
            except ValueError:raise DissertationError('Enter a valid pruning fraction.') from None
            if not 0<ratio<=.2:raise DissertationError('Pruning fraction must be greater than 0 and at most 0.20.')
            steps=self.integer('Recovery optimizer steps (10 for GPU smoke test; not efficacy evidence)',200)
            samples=self.integer('Training-only calibration examples',128,maximum=2048)
            maximum=self.integer('Exact maximum conversation tokens (no truncation)',4096,minimum=32,maximum=32768)
            args+=['--prune-ratio',str(ratio),'--max-steps',str(steps),'--calibration-samples',str(samples),'--max-tokens',str(maximum)]
            if self.confirm('Allow validated accepted subsets with incomplete enterprise source runs?',False):args.append('--allow-incomplete-sources')
        if self.preview('Model compression experiment',{'Enterprise':enterprise,'Native source plan':native,'Output':output,
            'Scope':'Environment checks only' if action=='check' else 'MLP pruning → BF16 LoRA recovery → NF4 4-bit export',
            'Hardware':'Linux + NVIDIA CUDA / BF16; no macOS or CPU fallback',
            'Memory':'BF16 pruning/recovery needs substantially more VRAM than 4-bit inference',
            'Evaluation':'Technical smoke checks only; final quality/security evaluation remains pending'}):self.execute(args,output)

    def preview_status(self,status):
        table=Table(show_header=False,box=None)
        for key,value in status.items():table.add_row(key.replace('_',' ').capitalize(),'Present / valid' if value else 'Missing / invalid')
        self.console.print(table)
        self.console.print('Presence checks only: endpoint reachability and token permissions are not verified.',style='dim')

    def pause(self):
        if self.terminal:
            try:questionary.text('Press Enter to return to the main menu').unsafe_ask()
            except (EOFError,KeyboardInterrupt):pass

    def run(self):
        # Alternate screen keeps completed pages out of the normal terminal history.
        # Plain/piped output remains append-only for automation and log capture.
        with self.console.screen(hide_cursor=False) if self.terminal else nullcontext():
            self.run_loop()

    def run_loop(self):
        dispatch={'enterprise':self.extraction,'enterprise_validation':self.enterprise_validation,
                  'external':self.external_installation,'external_validation':self.external_validation,'environment':self.environment,
                  'fine_tuning':self.compression}
        while True:
            if not self.terminal:self.header()
            try:choice=self.choose('Select an option to start',MENU)
            except (EOFError,KeyboardInterrupt):break
            if choice in ('exit',None):break
            if choice not in dispatch:
                self.console.print('Not implemented yet. No training, security scan, prompt attack or customization was started.',style='yellow')
                self.pause()
                continue
            try:dispatch[choice]()
            except Cancelled:self.console.print('Cancelled. No new operation was started.',style='yellow')
            except EOFError:break
            except KeyboardInterrupt:self.console.print('Cancelled. Existing data/checkpoints are preserved.',style='yellow')
            except (DissertationError,ValidationError) as exc:
                message=str(exc) if isinstance(exc,DissertationError) else 'Invalid configuration. Review Env setup; values are hidden.'
                self.console.print(Text(message,style='red'))
            except Exception:self.console.print('Could not complete this action. Review configuration and output reports; private errors are hidden.',style='red')
            self.pause()
        self.console.print('Goodbye.')


def run_dashboard():Dashboard().run()
