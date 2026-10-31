"""Profile a frozen native selection and select independent skill-aligned benchmarks.

This module is outside preparation adapter hashes; it never changes a training plan.
Labels are transparent heuristics, not semantic annotations or model outcomes.
"""
from collections import Counter, defaultdict
import ast
import copy
import heapq
import json
from pathlib import Path
import random
import re
import sqlite3

from policy_dissertation.config import Pipeline3Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from policy_dissertation.pipelines.external_datasets.sources import HubProvider
from policy_dissertation.pipelines.external_datasets.sampling import sampled
from policy_dissertation.pipelines.native_datasets.pipeline import decision_digest, VERSION as NATIVE_VERSION
from policy_dissertation.pipelines.model_evaluation.schemas import EvaluationConfig
from policy_dissertation.pipelines.model_evaluation.suites import official_rows, adapt, read_suite, VERSION
from policy_dissertation.pipelines.model_evaluation.execution import extract_code, DockerExecutor
from policy_dissertation.pipelines.model_evaluation.contamination import grams
from policy_dissertation.pipelines.model_evaluation.scoring import words
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json, atomic_text

# Multi-label tags: totals may exceed sample size. Regex matches are evidence, not
# proof of a task's principal skill; prompts are used to avoid answer-only inflation.
RULES = {
    'sequences':r'\b(list|lists|array|arrays|tuple|tuples|sequence|sequences|subarray|subsequence)\b',
    'strings':r'\b(string|strings|substring|character|characters|palindrome|anagram|text)\b',
    'mappings_sets':r'\b(dictionary|dictionaries|hashmap|hash map|set|sets|counter|frequency|frequencies)\b',
    'sorting_search':r'\b(sort|sorted|sorting|binary search|searching|bisect|heap|priority queue)\b',
    'numeric_math':r'\b(integer|integers|number|numbers|prime|divisor|factorial|fibonacci|sum|product|gcd|lcm|digit|digits)\b',
    'recursion_dp':r'\b(recurs\w*|dynamic programming|memoiz\w*|backtrack\w*)\b',
    'graphs_trees':r'\b(graph|graphs|tree|trees|node|nodes|vertex|vertices|breadth.first|depth.first)\b',
    'matrices':r'\b(matrix|matrices|2d|two.dimensional|grid|grids)\b',
    'regex_parsing':r'\b(regex|regular expression|parse|parser|parsing|tokeniz\w*)\b',
    'files_serialization':r'\b(file|files|directory|csv|json|serialize|serialization)\b',
    'oop':r'\b(class|classes|object.oriented|inheritance|method|methods)\b',
    'exceptions_validation':r'\b(exception|exceptions|raise|error handling|validate|validation|invalid input)\b',
    'concurrency':r'\b(async|asynchronous|thread|threads|threading|concurrency|multiprocessing)\b',
}


def skills(text):
    return sorted(tag for tag,pattern in RULES.items() if re.search(pattern,text,re.I)) or ['unclassified']


def language(prompt,answer):
    names=re.findall(r'```([\w+#-]+)',answer)
    aliases={'py':'python','js':'javascript','ts':'typescript','c++':'cpp','c#':'csharp'}
    labels={aliases.get(n.lower(),n.lower()) for n in names if n.lower() not in ('text','json','bash','sh','')}
    if len(labels)==1:return next(iter(labels)), 'code_fence'
    for name in ('python','javascript','typescript','java','kotlin','cpp','rust','go','sql'):
        if re.search(r'\b'+name+r'\b',prompt,re.I):return name,'prompt_keyword'
    code=extract_code(answer)
    if code:
        try:
            tree=ast.parse(code)
            if any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for n in ast.walk(tree)):
                return 'python','python_ast'
        except (SyntaxError,ValueError,RecursionError):pass
    return 'unknown','no_reliable_marker'


def function_hashes(text):
    code=extract_code(text)
    if not code:return set()
    try:tree=ast.parse(code)
    except (SyntaxError,ValueError,RecursionError):return set()
    result=set()
    for item in ast.walk(tree):
        if not isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
        node=copy.deepcopy(item)
        for child in ast.walk(node):
            if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                if child.body and isinstance(child.body[0],ast.Expr) and isinstance(child.body[0].value,ast.Constant) and isinstance(child.body[0].value.value,str):
                    child.body=child.body[1:]
        node.name='_entry'
        if sum(1 for _ in ast.walk(node))>=15:
            result.add(digest(ast.dump(node,include_attributes=False)))
    return result


class OverlapScreen:
    """Conservative explicit matching against all accepted raw source records."""
    def __init__(self,cases):
        self.targets={};self.index=defaultdict(set);self.code=defaultdict(set);self.matches={}
        for case in cases:
            for kind,text in (('prompt',case.source.get('task_prompt',case.messages[-1]['content'])),('reference',case.reference),('tests',case.tests)):
                values=grams(words(text))
                if values:
                    key=(case.id,kind);self.targets[key]=values
                    for gram in values:self.index[gram].add(key)
            for value in function_hashes(case.reference):self.code[value].add(case.id)

    def check(self,record_id,prompt,answer,tests=''):
        hits=Counter(key for gram in grams(words(prompt+'\n'+answer+'\n'+tests)) for key in self.index.get(gram,()))
        for key,n in hits.items():
            fraction=n/len(self.targets[key])
            if fraction>=.8:
                self.matches.setdefault(key[0],{'method':'13_word_containment_at_least_0.8','field':key[1],
                    'coverage':fraction,'training_record_id':record_id})
        for value in function_hashes(answer):
            for case_id in self.code.get(value,()):
                self.matches.setdefault(case_id,{'method':'python_function_ast_match','training_record_id':record_id})


def select_cases(cases,profile,limit,seed,retention_fraction=.2):
    rng=random.Random(seed);pool=list(cases);rng.shuffle(pool)
    counts=profile['skill_counts'];supported=[tag for tag,n in sorted(counts.items(),key=lambda x:(-x[1],x[0])) if tag!='unclassified' and n>=max(3,profile['sample_size']*.02)]
    aligned_target=limit-round(limit*retention_fraction)
    chosen=[];used=set()
    while len(chosen)<aligned_target:
        before=len(chosen)
        for tag in supported:
            candidate=next((c for c in pool if c.id not in used and tag in c.source['skill_tags']),None)
            if candidate:
                candidate=candidate.model_copy(deep=True)
                candidate.source.update(selection_role='skill_aligned',primary_skill=tag)
                chosen.append(candidate);used.add(candidate.id)
            if len(chosen)>=aligned_target:break
        if len(chosen)==before:break
    for case in pool:
        if len(chosen)>=limit:break
        if case.id not in used:
            candidate=case.model_copy(deep=True);candidate.source.update(selection_role='general_retention')
            chosen.append(candidate);used.add(candidate.id)
    return chosen


def verify_distinct_references(cases):
    """Guard exact/AST clones within the evaluation pool before splitting it."""
    seen_text={};seen_functions={}
    for case in cases:
        text=digest(words(case.reference))
        if text in seen_text:
            raise DissertationError('Duplicate evaluation references require reselection before freezing.')
        seen_text[text]=case.id
        hashes=function_hashes(case.reference)
        if any(h in seen_functions for h in hashes):
            raise DissertationError('Function-identical evaluation tasks require reselection before freezing.')
        for h in hashes:seen_functions[h]=case.id


def write_suite(path,cases,seed,profile_hash,selection):
    if not cases:raise DissertationError('No eligible independent tasks remain.')
    payload={'schema_version':VERSION,'seed':seed,'cases':[c.model_dump() for c in cases],
             'benchmark_selection':selection,'profile_hash':profile_hash,
             'limitations':['Python base HumanEval/MBPP only; not full debugging/refactoring/multilingual coverage.',
                 'Heuristic skills; labels require review.',
                 'Raw-source exact/near-text and function AST screening is not semantic independence proof.',
                 'Pretraining contamination and external data beyond this frozen selection remain unknown.']}
    payload['digest']=digest(payload);dump_json(path,payload)
    return payload['digest']


def prepare_skill_evaluation(plan_directory,enterprise,output,settings,*,sample_size=500,max_tasks=100,seed=42,final_tasks=None,
                              excluded_suites=(),provider=None,benchmark_provider=None,executor=None,progress=print):
    if final_tasks is not None and not 1<=final_tasks<max_tasks:
        raise DissertationError('Final task count must be positive and smaller than the total task count.')
    root,enterprise,output=(Path(p).resolve() for p in (plan_directory,enterprise,output))
    if sample_size<1 or max_tasks<10:raise DissertationError('Use at least one profile row and ten tasks for disjoint development/final suites.')
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in (root,enterprise)):
        raise DissertationError('Analysis output must be separate from source data.')
    if output.exists() and any(output.iterdir()):raise DissertationError('Analysis output exists; choose a new directory.')
    output.mkdir(parents=True,exist_ok=True)
    report={'status':'RUNNING','training_started':False,'llm_calls':0,'errors':[],
            'sample_size_requested':sample_size,'tasks_requested':max_tasks,'seed':seed}
    db=None
    try:
        plan_path=root/'source_plan.json';plan_initial_hash=file_hash(plan_path);plan=json.loads(plan_path.read_text())
        if plan.get('schema_version')!=NATIVE_VERSION or not plan.get('data_prepared'):
            raise DissertationError('Use a validated prepared native source plan.')
        if any(s['alias']!='opencode' for s in plan['sources']):
            raise DissertationError('This profiler currently supports the frozen OpenCodeInstruct selection only.')
        for name,expected in plan['enterprise_split_hashes'].items():
            if file_hash(enterprise/(name+'.jsonl'))!=expected:
                raise DissertationError('Enterprise split hashes differ from the selected training plan.')
        dbpath=(root/plan['decision_database']).resolve()
        if not dbpath.is_relative_to(root) or file_hash(dbpath)!=plan['decision_database_hash']:
            raise DissertationError('Native decision database path/hash mismatch.')
        db=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True)
        if decision_digest(db)!=plan['decisions_digest']:raise DissertationError('Native decision digest mismatch.')
        progress('[1/4] Selecting reproducible profile sample and loading official independent test candidates')
        heap=[];total=0;token_total=0
        for text,checksum in db.execute('SELECT data,checksum FROM native_decisions ORDER BY source,position'):
            row=json.loads(text)
            if digest(row)!=checksum:raise DissertationError('Decision checksum mismatch.')
            if not row['accepted']:continue
            total+=1;token_total+=row['token_count']
            rank=int(digest([seed,row['id']]),16)
            value=(-rank,row['id'])
            if len(heap)<sample_size:heapq.heappush(heap,value)
            elif value>heap[0]:heapq.heapreplace(heap,value)
        if total!=plan['accepted_examples']:raise DissertationError('Approved native row count mismatch.')
        sample_ids={i for _,i in heap}
        excluded=set()
        for path in excluded_suites:
            _,prior=read_suite(path);excluded.update(c.id for c in prior)
        candidates=[]
        token=settings.hf_token.get_secret_value() if settings.hf_token else None
        for name in ('humaneval','mbpp'):
            rows,source=(benchmark_provider or official_rows)(name,token)
            for row in rows:
                case=adapt(name,row,source)
                case.source.update(task_prompt=row['prompt'],skill_tags=skills(row['prompt']))
                if case.id not in excluded:candidates.append(case)
        audit=OverlapScreen(candidates);samples=[];language_counts=Counter();skill_counts=Counter();type_counts=Counter()
        configured=Pipeline3Settings.model_validate(plan['normalization'])
        provider=provider or HubProvider(configured,token)
        accepted_checked=0;decisions_checked=0;rows_read=0
        def read(*args):
            nonlocal rows_read
            rows_read+=1
        progress('[2/4] Replaying pinned raw identities and screening candidates against every accepted training row')
        for source in plan['sources']:
            stream=sampled(provider.stream(source),plan['seed'],configured,read)
            try:
                for position,text in db.execute('SELECT position,data FROM native_decisions WHERE source=? ORDER BY position',(source['alias'],)):
                    expected=json.loads(text)
                    item=next(stream)
                    if item['raw_hash']!=expected['raw_hash'] or item['ordinal']!=expected['ordinal']:
                        raise DissertationError('Pinned training source row/order changed.')
                    decisions_checked+=1
                    if not expected['accepted']:continue
                    raw=item['raw'];prompt,answer=raw['input'],raw['output']
                    tests=raw.get('unit_tests','');tests=tests if isinstance(tests,str) else json.dumps(tests)
                    audit.check(expected['id'],prompt,answer,tests)
                    accepted_checked+=1
                    if expected['id'] in sample_ids:
                        lang,evidence=language(prompt,answer);tags=skills(prompt)
                        kind='debugging' if re.search(r'\b(debug|fix the bug|fix a bug|buggy|repair)\b',prompt,re.I) else 'refactoring' if re.search(r'\brefactor\w*\b',prompt,re.I) else 'unit_test_generation' if re.search(r'\b(write|generate|create)\b.{0,35}\b(unit tests|test cases|tests)\b',prompt,re.I) else 'implementation_or_other'
                        language_counts[lang]+=1;skill_counts.update(tags);type_counts[kind]+=1
                        samples.append({'id':expected['id'],'source_record_id':raw.get('id'),
                            'source':source['dataset'],'revision':source['revision'],'ordinal':expected['ordinal'],
                            'raw_hash':item['raw_hash'],'token_count':expected['token_count'],
                            'language':lang,'language_evidence':evidence,'skills':tags,'task_type':kind,
                            'prompt':prompt,'answer':answer,'external_tests_status':raw.get('tests_execution_status'),
                            'external_tests_locally_verified':False})
                    if accepted_checked%5000==0:progress(f'Raw identity/overlap audit: {accepted_checked}/{total} accepted rows')
            finally:stream.close()
        for line in (enterprise/'train.jsonl').open(encoding='utf-8'):
            if not line.strip():continue
            row=json.loads(line);messages=row['messages']
            prompt='\n'.join(m['content'] for m in messages if m.get('role')!='assistant' and isinstance(m.get('content'),str))
            answer='\n'.join(m['content'] for m in messages if m.get('role')=='assistant' and isinstance(m.get('content'),str))
            audit.check(row['id'],prompt,answer)
        if accepted_checked!=total or len(samples)!=len(sample_ids):raise DissertationError('Incomplete source replay or profile sample.')
        profile={'sample_size':len(samples),'approved_selection_size':total,'approved_conversation_tokens':token_total,
                 'sampling':'bottom-k SHA256(seed, accepted record id), uniform-hash sample of this approved selection; not of all OpenCodeInstruct',
                 'language_counts':dict(language_counts),'skill_counts':dict(skill_counts),'task_type_counts':dict(type_counts),
                 'labels':'overlapping prompt-regex skills, code-fence/prompt/AST language evidence; not semantic annotation',
                 'python_sample_proportion':language_counts['python']/len(samples)}
        if profile['python_sample_proportion']<.5:raise DissertationError('Python-only benchmark alignment is insufficient for this sample; manual multilingual evaluation design required.')
        atomic_text(output/'profile_sample.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(samples,key=lambda r:r['id'])))
        dump_json(output/'profile.json',profile)
        dump_json(output/'overlap.json',{'native_accepted_rows_checked':accepted_checked,'native_decisions_identity_checked':decisions_checked,
            'enterprise_training_checked':True,'candidates_excluded':audit.matches,
            'prior_suite_ids_excluded':sorted(excluded),'method':'13-word >=80% containment and docstring-free function AST identity',
            'limitations':'Not semantic clone detection; short text, renamed code, paraphrases and base-model pretraining overlap unresolved.'})
        eligible=[c for c in candidates if c.id not in audit.matches]
        unique=[];seen_text=set();seen_code=set();duplicate_ids=[]
        for case in eligible:
            text=digest(words(case.reference));code=function_hashes(case.reference)
            if text in seen_text or code & seen_code:
                duplicate_ids.append(case.id);continue
            unique.append(case);seen_text.add(text);seen_code.update(code)
        selected=select_cases(unique,profile,max_tasks,seed)
        verify_distinct_references(selected)
        progress('[3/4] Checking selected official reference solutions against their tests inside Docker')
        executor=executor or DockerExecutor(EvaluationConfig())
        report['test_runtime']=executor.preflight()
        verified=[];bad=[]
        for i,case in enumerate(selected,1):
            reference=executor.run(case.reference,case.tests)
            if reference['status']=='unverified':raise DissertationError('Docker reference verification unavailable; no suite approved.')
            if reference['status']!='passed':bad.append({'id':case.id,'result':reference});continue
            # Require tests to distinguish a wrong implementation, not just a crashing harness.
            mutated=ast.parse(case.reference)
            for node in ast.walk(mutated):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    node.body=[ast.Return(value=ast.Constant(value=None))]
            mutant=ast.unparse(ast.fix_missing_locations(mutated))
            control=executor.run(mutant,case.tests)
            if control['status']=='unverified':raise DissertationError('Negative-control execution unavailable.')
            if control['status']!='failed':
                bad.append({'id':case.id,'result':{'reason':'return_none_mutant_not_rejected'}})
                continue
            case.source['local_reference_test_status']='passed'
            verified.append(case)
            if i%10==0:progress(f'Reference validation: {i}/{len(selected)}')
        rng=random.Random(seed);rng.shuffle(verified)
        final_count=final_tasks if final_tasks is not None else max(1,round(len(verified)*.2))
        if len(verified)<=final_count:
            raise DissertationError('Not enough verified tasks for the requested final count and a separate development set.')
        final=verified[:final_count];development=verified[final_count:]
        progress('[4/4] Freezing disjoint development and final suites before any model results')
        selection='Pre-result skill-aligned selection plus general Python retention; prior development task IDs and detected training overlap excluded.'
        suites={}
        for name,values in (('development_suite',development),('final_suite',final)):
            suites[name]=write_suite(output/(name+'.json'),values,seed,digest(profile),selection)
        if file_hash(plan_path)!=plan_initial_hash or file_hash(dbpath)!=plan['decision_database_hash']:
            raise DissertationError('Source changed during analysis.')
        report.update(status='COMPLETED_WITH_WARNINGS',native_plan_run_id=plan['run_id'],native_plan_hash=file_hash(plan_path),
            profile=profile,native_accepted_rows_checked=accepted_checked,decoded_rows_read=rows_read,
            candidate_count=len(candidates),overlap_excluded=len(audit.matches),reference_failures=bad,
            duplicate_reference_candidates_excluded=duplicate_ids,requested_final_tasks=final_tasks,
            development_tasks=len(development),final_tasks=len(final),suite_digests=suites,
            selected_skill_counts=dict(Counter(tag for c in verified for tag in c.source['skill_tags'])),
            selected_benchmark_counts=dict(Counter(c.suite for c in verified)),
            selected_role_counts=dict(Counter(c.source['selection_role'] for c in verified)),
            warnings=['Heuristic profile labels need review.','No semantic independence or pretraining-contamination guarantee.',
                'Base Python implementation tests do not cover every observed task type.','Keep final suite untouched until experiment choices are frozen.'])
        if len(verified)<max_tasks:report['warnings'].append('Requested task count not reached; actual verified count reported.')
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='FAILED',errors=[str(exc) if isinstance(exc,DissertationError) else 'skill_preparation_failed:'+type(exc).__name__])
        for name in ('development_suite.json','final_suite.json'):
            (output/name).unlink(missing_ok=True)
    finally:
        if db:db.close()
        dump_json(output/'report.json',report)
    return report
