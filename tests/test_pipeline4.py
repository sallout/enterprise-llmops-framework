"""Synthetic preparation fixtures; fake token lengths never certify the actual Qwen tokenizer."""
import copy
import hashlib
import json
import random
from pathlib import Path
import pytest
from typer.testing import CliRunner
from policy_dissertation.cli import app
from policy_dissertation.config import Settings
from policy_dissertation.pipelines.dataset_preparation.adapters import adapt, InvalidExample
from policy_dissertation.pipelines.dataset_preparation.deduplication import candidates
from policy_dissertation.pipelines.dataset_preparation.pipeline import run_pipeline4
from policy_dissertation.pipelines.dataset_preparation.schemas import UnifiedRecord
from policy_dissertation.pipelines.dataset_preparation.selection import mix_counts
from policy_dissertation.pipelines.dataset_preparation.storage import PreparationStore
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash


class FakeTokenizer:
    info={'model':'Qwen/Qwen3-14B','revision':'mock-only','chat_template_hash':'mock-template', 'validated':False}
    special_tokens={'<|im_start|>','<|im_end|>','<|endoftext|>'}
    calls=0
    def measure(self,messages,tools=None):
        self.calls+=1
        return len(json.dumps([messages,tools],ensure_ascii=False))//4+10


def policy(i=0):
    answer=f'Employees MUST request approval for policy clause {i} before transferring restricted records.'
    return {'schema_version':'pipeline1-extractive-v1','example_id':f'policy-{i}',
        'messages':[{'role':'system','content':f'Corporate policy document {i//2}; follow its complete requirements.'},
                    {'role':'user','content':f'Explain the approval requirement for policy clause {i}.'},
                    {'role':'assistant','content':answer}],
        'metadata':{'document_id':f'document-{i//2}','source_path':f'dumy_policies/policy-{i//2}.txt',
                    'chunk_id':f'chunk-{i}','source_content_hash':'a'*64,'evidence_quote':answer,
                    'section_id':str(i),'section_title':'Transfers','page_numbers':[1],'validation':{'schema':'passed'}}}


def repository(i=0,coding=False):
    data={'schema_version':'repository-data-v1','example_id':f'repository-{i}',
        'messages':[{'role':'system','content':f'Analyze source symbol helper_{i} in its supplied repository context.'},
                    {'role':'user','content':f'Explain how helper_{i} returns the transformed numeric result.'},
                    {'role':'assistant','content':f'Helper {i} applies the documented transformation to its input and returns the numeric result.'}],
        'metadata':{'repository_id':'repository-A','repository_name':'synthetic','file_path':f'src/helper_{i}.py',
                    'source_unit_id':f'unit-{i}','task_type':'explanation','file_hash':'b'*64,'snapshot_hash':'c'*64,
                    'start_line':1,'end_line':3,'verification_status':'unverified',
                    'execution':{'status':'unverified','reason':'runtime_not_applicable_to_knowledge_task'},
                    'source_quotes':['return value + 1'], 'trajectory':[]}}
    if coding:
        data['metadata'].update(task_type='implementation',verification_status='verified',
            execution={'status':'verified','executor':'docker','exit_code':0,'tests_run':2,'failures':0,'errors':0,
                       'skipped':0,'expected_failures':0,'unexpected_successes':0,'image_id':'sha256:'+'f'*64})
        data['messages'][-1]['content']+='\n\nTests:\n```python\nassert helper(1) == 2\n```'
    return data


def external(i=0,split='train'):
    return {'schema_version':'external-datasets-v1','example_id':f'external-{i}',
        'messages':[{'role':'user','content':f'Write a short explanation of unrelated concept number {i}.'},
                    {'role':'assistant','content':f'Concept {i} is described using a concise self-contained factual explanation.'}],
        'metadata':{'dataset':'nvidia/OpenCodeInstruct','config':'train','split':split,'revision':'d'*40,
            'record_id':f'upstream-{i}','record_id_kind':'upstream','stream_ordinal':i,'content_hash':'e'*64,
            'source_url':'https://huggingface.co/datasets/nvidia/OpenCodeInstruct','license':'cc-by-4.0',
            'license_url':'https://creativecommons.org/licenses/by/4.0/','source_fields':{'unit_tests':'["assert True"]',
            'tests_execution_status':'["pass"]','average_test_score':1.0},'source_message_metadata':{},
            'transformations':[],'token_count':100,'token_method':'utf8_serialized_messages_bytes_v1',
            'locally_executed':False,'semantic_status':'structural_checks_only','external_test_results_status':'not_locally_verified'}}


def write_source(root,pipeline,rows,status='COMPLETED'):
    p=root/pipeline;p.mkdir(parents=True,exist_ok=True)
    (p/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (p/'report.json').write_text(json.dumps({'status':status,'accepted_examples':len(rows),'errors':[],
        'output_hashes':{'training.jsonl':file_hash(p/'training.jsonl')}}))


@pytest.fixture
def settings():
    return Settings(_env_file=None, model={'api_key':'configured-private-token-123456'})


@pytest.fixture
def inputs(tmp_path):
    p=tmp_path/'inputs'
    for name,fn in [('pipeline1',policy),('pipeline2',repository),('pipeline3',external)]:
        write_source(p,name,[fn(i) for i in range(10)])
    return p


def run(inputs,tmp_path,settings,**kwargs):
    return run_pipeline4(inputs,tmp_path/'out',settings,overwrite=True,tokenizer=kwargs.pop('tokenizer',FakeTokenizer()),
                         research_report=tmp_path/'report.md',**kwargs)


def exports(tmp_path):
    return {name:[json.loads(l) for l in (tmp_path/f'out/{name}.jsonl').read_text().splitlines()]
            for name in ('train','validation','test','rejected')}


@pytest.mark.parametrize('pipeline,raw,domain,status',[
    ('pipeline1',policy(),'corporate_compliance','not_applicable'),
    ('pipeline2',repository(coding=True),'coding','upstream_local_pass'),
    ('pipeline3',external(),'coding','externally_reported')])
def test_source_adapters_keep_content_provenance_and_verification(pipeline,raw,domain,status):
    original=copy.deepcopy(raw)
    r=adapt(raw,pipeline,{'line':1})
    assert r['messages']==raw['messages'] and r['metadata']['provenance']==raw['metadata']
    assert r['domain']==domain and r['verification']['status']==status
    assert r['verification']['executed_in_pipeline4'] is False
    assert raw==original
    UnifiedRecord.model_validate(r)
    if pipeline=='pipeline3':
        assert r['verification']['unit_tests']=='["assert True"]'
        assert r['verification']['evidence']['locally_verified'] is False
    if pipeline=='pipeline2':assert r['verification']['unit_tests']=='assert helper(1) == 2'


def test_tool_conversation_preserved_with_definitions():
    raw=external()
    raw['metadata']['source_fields']['tools']=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object'}}}]
    raw['messages'][1]={'role':'assistant','content':None,'tool_calls':[{'id':'call','type':'function',
        'function':{'name':'lookup','arguments':'{"key":"x"}'}}]}
    raw['messages'] += [{'role':'tool','content':'result','tool_call_id':'call'},
                        {'role':'assistant','content':'Here is the original result.'}]
    r=adapt(raw,'pipeline3',{})
    assert r['messages']==raw['messages'] and r['tools']==raw['metadata']['source_fields']['tools']


@pytest.mark.parametrize('mode',['empty','invalid_role','unfinished','orphan_tool','missing_reference','bad_hash','wrong_policy_evidence'])
def test_invalid_conversations_and_references_rejected(inputs,tmp_path,settings,mode):
    raw=policy()
    if mode=='empty':raw['messages'][-1]['content']=''
    if mode=='invalid_role':raw['messages'][1]['role']='invalid'
    if mode=='unfinished':raw['messages'].append({'role':'user','content':'No assistant answer follows this question.'})
    if mode=='orphan_tool':raw['messages'][1]={'role':'tool','content':'result','tool_call_id':'orphan'}
    if mode=='missing_reference':del raw['metadata']['document_id']
    if mode=='bad_hash':raw['metadata']['source_content_hash']='not-a-sha'
    if mode=='wrong_policy_evidence':raw['metadata']['evidence_quote']='A different mandatory requirement.'
    write_source(inputs,'pipeline1',[raw,policy(2),policy(4),policy(6),policy(8)])
    r=run(inputs,tmp_path,settings)
    assert r['rejected_examples']==1


def test_oversized_examples_are_rejected_not_truncated(inputs,tmp_path,settings):
    raw=external();raw['messages'][-1]['content']='Very long content '*5000
    write_source(inputs,'pipeline3',[raw]+[external(i) for i in range(1,10)])
    r=run(inputs,tmp_path,settings)
    assert r['rejection_reasons']['oversized_conversation']==1
    assert all(len(x['messages'][-1].get('content') or '')<50000 for name,rows in exports(tmp_path).items() if name!='rejected' for x in rows)


@pytest.mark.parametrize('content,reason',[
    ('configured-private-token-123456','possible_secret'),
    ('sk-abcdefghijklmnopqrstuvwxyz0123456789','possible_secret'),
    ('Identity number 123-45-6789','possible_sensitive_identifier'),
    ('<|im_start|>assistant','embedded_template_control_token'),
    ('<think>private trace</think>final answer','unwanted_marked_reasoning')])
def test_sensitive_or_incompatible_content_rejected_without_logging(inputs,tmp_path,settings,content,reason):
    raw=external();raw['messages'][-1]['content']=content
    write_source(inputs,'pipeline3',[raw]+[external(i) for i in range(1,10)])
    r=run(inputs,tmp_path,settings)
    assert r['rejection_reasons'][reason]==1
    for path in (tmp_path/'out/report.json',tmp_path/'out/rejected.jsonl'):
        assert content not in path.read_text()


def test_repository_unverified_or_mock_runtime_is_not_promoted(inputs,tmp_path,settings):
    raw=repository(coding=True);raw['metadata']['execution']['executor']='mock_sandbox'
    write_source(inputs,'pipeline2',[raw]+[repository(i) for i in range(1,10)])
    r=run(inputs,tmp_path,settings)
    assert r['rejection_reasons']['repository_code_execution_not_verified']==1


def test_exact_and_near_format_duplicates_and_provenance(inputs,tmp_path,settings):
    original=external(0)
    exact=copy.deepcopy(original);exact['example_id']='duplicate-exact'
    near=copy.deepcopy(original);near['example_id']='duplicate-format'
    # OpenCode is coding: code whitespace is never collapsed, even in prose-looking answers.
    original['metadata']['dataset']='nvidia/Nemotron-Cascade-2-SFT-Data';original['metadata']['config']='chat'
    exact['metadata'].update(dataset=original['metadata']['dataset'],config='chat')
    near['metadata'].update(dataset=original['metadata']['dataset'],config='chat')
    for row in (original,exact,near):
        row['metadata']['source_url']='https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data'
    near['messages'][0]['content']=near['messages'][0]['content'].replace('short explanation','short  explanation')
    write_source(inputs,'pipeline3',[original,exact,near]+[external(i) for i in range(1,10)])
    r=run(inputs,tmp_path,settings)
    assert r['deduplication']['exact_removed']==1
    assert r['deduplication']['near_format_removed']==1
    records=[x for s,rows in exports(tmp_path).items() if s!='rejected' for x in rows]
    refs=[x for x in records if x['metadata'].get('duplicate_references')]
    # Representative may be excluded by mixing; duplicate audit still identifies it.
    assert r['duplicates_removed']==2
    assert sum(x['disposition']=='duplicate' for x in exports(tmp_path)['rejected'])==2


def test_similar_mandatory_rules_are_retained_but_grouped(inputs,tmp_path,settings):
    a=policy(0);b=policy(2)
    context=' '.join(f'Operational workflow clause {i} preserves condition identifier value_{i} and approval stage stage_{i}.' for i in range(50))
    for r in (a,b):r['messages'][0]['content']=context
    b['messages'][-1]['content']=a['messages'][-1]['content'].replace('MUST request','MUST NOT request')
    b['metadata']['evidence_quote']=b['messages'][-1]['content']
    write_source(inputs,'pipeline1',[a,b]+[policy(i) for i in range(4,20)])
    r=run(inputs,tmp_path,settings)
    assert r['deduplication']['near_pairs_detected']>0
    assert not r['deduplication']['near_format_removed']
    assert r['splitting']['near_pair_leakage']==0
    records=[x for s,rows in exports(tmp_path).items() if s!='rejected' for x in rows]
    roots={x['metadata']['split_group']:set() for x in records}
    for x in records:roots[x['metadata']['split_group']].add(x['metadata']['split'])
    assert all(len(s)==1 for s in roots.values())


def test_mixing_without_replacement_and_deterministic_splits(inputs,tmp_path,settings):
    first=run(inputs,tmp_path,settings)
    assert first['mixing']['selected']=={'pipeline1':10,'pipeline2':8,'pipeline3':7}
    assert first['accepted_examples']==25
    assert first['training_ready'] is False # Fake token counts cannot certify Qwen compatibility.
    before={p:Path(tmp_path/f'out/{p}.jsonl').read_bytes() for p in ('train','validation','test','rejected')}
    second=run(inputs,tmp_path,settings)
    assert second['resumed_stages']==['load','deduplication']
    assert before=={p:Path(tmp_path/f'out/{p}.jsonl').read_bytes() for p in before}
    assert first['output_hashes']==second['output_hashes']
    rows=exports(tmp_path);ids=[];groups={}
    for name in ('train','validation','test'):
        assert rows[name]
        for x in rows[name]:
            ids.append(x['id']);groups.setdefault(x['metadata']['split_group'],set()).add(name)
    assert len(ids)==len(set(ids))==25 and all(len(g)==1 for g in groups.values())


def test_official_holdouts_and_source_components_never_leak(inputs,tmp_path,settings):
    write_source(inputs,'pipeline3',[external(i,'test' if i<3 else 'validation' if i<5 else 'train') for i in range(10)])
    r=run(inputs,tmp_path,settings)
    assert r['splitting']['forced_official_groups']>0
    docs={}
    for name,rows in exports(tmp_path).items():
        if name=='rejected':continue
        for x in rows:
            official=x['metadata'].get('official_holdout')
            if official:assert name==official
            if x['metadata']['pipeline']=='pipeline1':
                docs.setdefault(x['metadata']['provenance']['document_id'],set()).add(name)
    assert all(len(v)==1 for v in docs.values())


def test_failed_or_missing_mandatory_inputs_fail_closed(inputs,tmp_path,settings):
    (inputs/'pipeline1/training.jsonl').unlink()
    r=run(inputs,tmp_path,settings)
    assert r['status']=='FAILED' and not r['training_ready'] and r['accepted_examples']==0
    assert 'pipeline1:missing_training_jsonl' in r['input_issues']
    assert all((tmp_path/'out'/name).exists() for name in ('train.jsonl','validation.jsonl','test.jsonl','rejected.jsonl','report.json','manifest.json'))
    partial=run(inputs,tmp_path/'partial',settings,allow_partial_sources=True)
    assert partial['accepted_examples']>0 and partial['training_ready'] is False


def test_checksum_failure_skips_untrusted_source_even_in_partial_mode(inputs,tmp_path,settings):
    p=inputs/'pipeline1/training.jsonl';p.write_text(p.read_text()+'{}\n')
    r=run(inputs,tmp_path,settings,allow_partial_sources=True)
    assert 'pipeline1:training_checksum_mismatch' in r['input_issues']
    assert r['distribution']['pipeline'].get('pipeline1',0)==0


def test_interruption_resumes_only_committed_validation(inputs,tmp_path,settings):
    class Interrupted(FakeTokenizer):
        def measure(self,messages,tools=None):
            if self.calls==1:raise KeyboardInterrupt()
            return super().measure(messages,tools)
    failed=run(inputs,tmp_path,settings,tokenizer=Interrupted())
    assert failed['status']=='FAILED'
    resumed=run(inputs,tmp_path,settings)
    assert resumed['input_records']==30 and resumed['accepted_examples']==25
    assert len({x['id'] for s,rows in exports(tmp_path).items() if s!='rejected' for x in rows})==25


def test_prefix_filter_has_no_misses_against_brute_force_jaccard(tmp_path):
    s=PreparationStore(tmp_path/'prefix.sqlite3');rng=random.Random(2)
    sets=[]
    for i in range(30):
        vals={hashlib.sha256(str(x).encode()).digest() for x in range(50)}
        for x in rng.sample(range(50),rng.randrange(0,5)):vals.remove(hashlib.sha256(str(x).encode()).digest())
        values=sorted(vals);prefix=candidates(s,values,.9)
        actual={r[0] for r in s.db.execute('SELECT rid FROM candidates')}
        expected={j for j,other in enumerate(sets) if len(vals&other)/len(vals|other)>=.9}
        assert expected<=actual
        s.db.executemany('INSERT INTO prefixes VALUES (?,?,?)',((v,i,len(vals)) for v in prefix));sets.append(vals)
    s.close()


def test_tokenizer_failure_never_claims_readiness(inputs,tmp_path,settings,monkeypatch):
    import policy_dissertation.pipelines.dataset_preparation.pipeline as module
    from policy_dissertation.errors import DissertationError
    monkeypatch.setattr(module,'load_tokenizer',lambda *args: (_ for _ in ()).throw(DissertationError('Actual tokenizer unavailable.')))
    r=run_pipeline4(inputs,tmp_path/'out',settings,research_report=tmp_path/'report.md')
    assert r['status']=='FAILED' and not r['training_ready']


def test_cli_and_export_contract(inputs,tmp_path,settings,monkeypatch):
    import policy_dissertation.pipelines.dataset_preparation.pipeline as module
    monkeypatch.setattr('policy_dissertation.cli.Settings',lambda:settings)
    monkeypatch.setattr(module,'load_tokenizer',lambda *args:FakeTokenizer())
    monkeypatch.chdir(tmp_path)
    result=CliRunner().invoke(app,['pip4_prepare','--input',str(inputs),'--output',str(tmp_path/'out'),
        '--max-tokens','4096','--seed','7'])
    assert result.exit_code==0,result.output
    assert 'Exported examples: 25' in result.output
    for name,rows in exports(tmp_path).items():
        if name!='rejected':
            for x in rows:
                UnifiedRecord.model_validate(x)
                assert 'verification' in x and 'provenance' in x['metadata']
                assert x['metadata']['preparation']['truncated'] is False
                assert x['metadata']['preparation']['token_count']<=4096
    invalid=CliRunner().invoke(app,['pip4_prepare','--mix-ratio','pipeline1=nan'])
    assert invalid.exit_code!=0


def test_insufficient_groups_are_not_training_ready(inputs,tmp_path,settings):
    for p,fn in [('pipeline1',policy),('pipeline2',repository),('pipeline3',external)]:write_source(inputs,p,[fn()])
    settings.pipeline4.split_ratios={'train':.9,'validation':.05,'test':.05}
    r=run(inputs,tmp_path,settings,max_examples=1)
    assert r['status']=='FAILED' and not r['training_ready']


def test_ratio_math_does_not_oversample():
    counts,total=mix_counts({'pipeline1':40,'pipeline2':30,'pipeline3':1000000},{'pipeline1':.4,'pipeline2':.3,'pipeline3':.3})
    assert counts=={'pipeline1':40,'pipeline2':30,'pipeline3':30} and total==100


def test_cross_source_exact_copy_moves_policy_group_into_official_test(inputs,tmp_path,settings):
    copy_record=external(99,'test')
    copy_record['messages']=copy.deepcopy(policy(0)['messages'])
    write_source(inputs,'pipeline3',[copy_record]+[external(i) for i in range(10)])
    r=run(inputs,tmp_path,settings)
    assert r['duplicates_removed']==1
    policy_zero=[x for name,rows in exports(tmp_path).items() if name!='rejected' for x in rows
                 if x['metadata']['upstream_example_id']=='policy-0']
    assert len(policy_zero)==1 and policy_zero[0]['metadata']['split']=='test'
    assert policy_zero[0]['metadata']['duplicate_references'][0]['official_holdout']=='test'
    assert r['splitting']['exact_leakage']==0


def test_bad_json_and_bounded_oversized_line_do_not_stop_siblings(inputs,tmp_path,settings):
    p=inputs/'pipeline3/training.jsonl'
    p.write_bytes(b'not-json\n'+b'x'*5000+b'\n'+b''.join((json.dumps(external(i))+'\n').encode() for i in range(10)))
    upstream={'status':'COMPLETED','accepted_examples':12,'errors':[], 'output_hashes':{'training.jsonl':file_hash(p)}}
    (p.parent/'report.json').write_text(json.dumps(upstream))
    settings.pipeline4.max_line_bytes=4096
    r=run(inputs,tmp_path,settings)
    assert r['input_records']==32
    assert r['rejection_reasons']['invalid_json_or_source_schema']==1
    assert r['rejection_reasons']['oversized_jsonl_record']==1
    assert r['accepted_examples']>0


def test_model_context_limit_is_independent_of_requested_limit(inputs,tmp_path,settings):
    class Limited(FakeTokenizer):
        info=FakeTokenizer.info|{'max_context_tokens':20}
    r=run(inputs,tmp_path,settings,tokenizer=Limited(),max_tokens=4096)
    assert r['rejection_reasons']['oversized_conversation']==30
    assert not r['training_ready'] and r['status']=='FAILED'


def test_training_output_cannot_overwrite_upstream(inputs,settings):
    from policy_dissertation.errors import DissertationError
    with pytest.raises(DissertationError):run_pipeline4(inputs,inputs/'pipeline3',settings,tokenizer=FakeTokenizer())


def test_inputs_changed_during_preparation_are_not_ready(inputs,tmp_path,settings):
    class Changing(FakeTokenizer):
        def measure(self,messages,tools=None):
            if self.calls==0:
                path=inputs/'pipeline3/report.json'
                data=json.loads(path.read_text());data['changed']=True;path.write_text(json.dumps(data))
            return super().measure(messages,tools)
    r=run(inputs,tmp_path,settings,tokenizer=Changing())
    assert r['status']=='FAILED'
    assert 'pipeline3:input_changed_during_preparation' in r['errors']
    assert not json.loads((tmp_path/'out/manifest.json').read_text())['training_ready']


def test_research_report_failure_invalidates_manifest_readiness(inputs,tmp_path,settings):
    class SimulatedVerified(FakeTokenizer):
        # Artificial flag exercises the failure path only; not actual tokenizer evidence.
        info=FakeTokenizer.info|{'validated':True}
    blocked=tmp_path/'directory-not-a-report';blocked.mkdir()
    r=run_pipeline4(inputs,tmp_path/'out',settings,tokenizer=SimulatedVerified(),research_report=blocked)
    assert r['status']=='FAILED' and r['training_ready'] is False
    manifest=json.loads((tmp_path/'out/manifest.json').read_text())
    assert manifest['training_ready'] is False and manifest['status']=='FAILED'


def test_real_cached_qwen_template_and_tool_token_count():
    from policy_dissertation.pipelines.dataset_preparation.tokenization import QwenTokenizer
    from policy_dissertation.pipelines.policy_transformation.data_io import digest
    folder=Path(__file__).resolve().parents[1]/'artifacts/pipeline4'
    if not list((folder/'_tokenizer').glob('*.json')):
        pytest.skip('Actual tokenizer assets unavailable locally; never download in unit tests.')
    settings=Settings(_env_file=None);settings.pipeline4.local_files_only=True
    tokenizer=QwenTokenizer(folder,settings.pipeline4)
    tools=[{'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'key':{'type':'string'}}}}}]
    messages=[{'role':'user','content':'Look up x.'},
        {'role':'assistant','content':None,'tool_calls':[{'id':'call1','type':'function','function':{'name':'lookup','arguments':'{"key":"x"}'}}]},
        {'role':'tool','tool_call_id':'call1','content':'value'}, {'role':'assistant','content':'The value is available.'}]
    ids=tokenizer.tokenizer.apply_chat_template(messages,tools=tools,tokenize=True,add_generation_prompt=False,
        enable_thinking=False,truncation=False,return_dict=False)
    measured,key=tokenizer.analyze(messages,tools)
    assert measured==len(ids) and key==digest(ids)
    assert tokenizer.info['validated'] is True and tokenizer.info['model']=='Qwen/Qwen3-14B'
    assert tokenizer.measure(messages,tools)>tokenizer.measure(messages)


def test_resolved_symlink_output_cannot_overwrite_source_report(inputs,tmp_path,settings):
    from policy_dissertation.errors import DissertationError
    link=tmp_path/'linked-output';link.symlink_to(inputs/'pipeline3',target_is_directory=True)
    original=(inputs/'pipeline3/report.json').read_bytes()
    with pytest.raises(DissertationError):run_pipeline4(inputs,link,settings,tokenizer=FakeTokenizer())
    assert (inputs/'pipeline3/report.json').read_bytes()==original
