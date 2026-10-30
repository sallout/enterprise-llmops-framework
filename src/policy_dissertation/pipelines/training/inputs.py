"""Verified input reuse and exact assistant-only token supervision, never holdout training."""
import json
import random
from collections import Counter
from pathlib import Path
from policy_dissertation.errors import DissertationError
from policy_dissertation.data.training_inputs import TrainingInputGate,index_enterprise
from policy_dissertation.pipelines.native_datasets.replay import iter_prepared_native
from policy_dissertation.pipelines.policy_transformation.data_io import digest,dump_json
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash


def encode_record(record, tokenizer, maximum):
    options={'add_generation_prompt':False,'enable_thinking':False}
    if record.get('tools'):options['tools']=record['tools']
    text=tokenizer.apply_chat_template(record['messages'],tokenize=False,**options)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
    ids=encoded['input_ids']
    expected=tokenizer.apply_chat_template(record['messages'],tokenize=True,**options)
    if hasattr(expected,'keys'):expected=expected['input_ids']
    if expected and isinstance(expected[0],list):expected=expected[0]
    if ids!=expected:raise DissertationError('Training rendering differs from the pinned conversation tokenization.')
    if len(ids)>maximum:raise DissertationError('Training record exceeds max-tokens; silent truncation is prohibited.')
    # Qwen role delimiters cannot occur in validated user/source content. Include
    # assistant text/tool calls and its end token, excluding all user/tool responses.
    spans=[];cursor=0;start_marker='<|im_start|>assistant\n';end_marker='<|im_end|>'
    while True:
        start=text.find(start_marker,cursor)
        if start<0:break
        start+=len(start_marker);end=text.find(end_marker,start)
        if end<0:raise DissertationError('Unterminated assistant message in Qwen rendering.')
        spans.append((start,end+len(end_marker)));cursor=end+len(end_marker)
    labels=[token if end>start and any(start>=a and end<=b for a,b in spans) else -100
            for token,(start,end) in zip(ids,encoded['offset_mapping'],strict=True)]
    if not spans or not any(label!=-100 for label in labels[1:]):
        raise DissertationError('Training record has no supervised assistant tokens.')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels,'record_id':record['id']}


def snapshot_inputs(enterprise,native):
    enterprise=Path(enterprise).resolve();native=Path(native).resolve()
    plan=json.loads((native/'source_plan.json').read_text())
    db=(native/plan['decision_database']).resolve()
    if not db.is_relative_to(native):raise DissertationError('Native checkpoint escapes its source directory.')
    paths=[enterprise/name for name in ('manifest.json','train.jsonl','validation.jsonl','test.jsonl')]
    if (enterprise/'report.json').exists():paths.append(enterprise/'report.json')
    paths += [native/'source_plan.json',db]
    return {str(p):file_hash(p) for p in paths}


def prepare_inputs(enterprise,native,output,settings,config,tokenizer,*,provider=None,progress=lambda _:None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():raise DissertationError('Use the committed input stage or a fresh cache directory.')
    plan=json.loads((Path(native)/'source_plan.json').read_text())
    from policy_dissertation.config import Pipeline3Settings,Pipeline4Settings
    configured=settings.model_copy(deep=True)
    configured.pipeline3=Pipeline3Settings.model_validate(plan['normalization'])
    configured.pipeline4=Pipeline4Settings.model_validate(plan['quality'])
    if not plan.get('data_prepared') or plan.get('status')=='FAILED':raise DissertationError('NVIDIA source plan is not prepared.')
    if config.max_tokens>plan['max_tokens']:progress('Training token bound exceeds source-plan bound; source records retain their original checks.')
    gate=TrainingInputGate(output/'fingerprints.sqlite3',configured.pipeline4.near_threshold)
    counts=Counter();reservoirs={};seen=Counter();rng=random.Random(config.seed)
    warnings=[]
    try:
        index_enterprise(enterprise,configured,tokenizer,config.max_tokens,gate)
        incomplete=any(s.get('source_run_complete') is not True for s in gate.sources)
        if incomplete and not config.allow_incomplete_sources:
            raise DissertationError('Enterprise source run is incomplete; explicitly allow its accepted subset or supply completed sources.')
        if incomplete:warnings.append('accepted_subset_from_incomplete_enterprise_sources')
        gate.seal_holdouts()
        def rows():
            with (Path(enterprise)/'train.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():yield json.loads(line),'enterprise'
            for row in iter_prepared_native(native,configured,gate,tokenizer=tokenizer,provider=provider):
                yield row,'external'
        with (output/'tokens.jsonl').open('w',encoding='utf-8') as stream, (output/'provenance.jsonl').open('w',encoding='utf-8') as audit:
            for raw,source in rows():
                encoded=encode_record(raw,tokenizer.tokenizer,config.max_tokens)
                group=raw['metadata'].get('pipeline','unknown') if source=='enterprise' else raw.get('source','external')
                bucket=reservoirs.setdefault(group,[]);seen[group]+=1
                if len(bucket)<config.calibration_samples:bucket.append(encoded)
                else:
                    position=rng.randrange(seen[group])
                    if position<len(bucket):bucket[position]=encoded
                copies=config.enterprise_repeats if source=='enterprise' else 1
                for _ in range(copies):stream.write(json.dumps(encoded)+'\n')
                audit.write(json.dumps({'record_id':raw['id'],'source':raw['source'],'domain':raw['domain'],
                                        'metadata':raw['metadata'],'verification':raw['verification'],'copies':copies},ensure_ascii=False)+'\n')
                counts[source]+=1;counts['training_rows']+=copies
                if sum(counts[k] for k in ('enterprise','external'))%100==0:progress('Validated '+str(counts['training_rows'])+' training rows')
        if not counts['enterprise'] or not counts['external']:raise DissertationError('Both enterprise training examples and approved NVIDIA examples are required.')
        # Round-robin from bounded per-source reservoirs avoids calibration being
        # entirely dominated by the largest source. This is not semantic balancing.
        calibration=[];groups=sorted(reservoirs)
        for values in reservoirs.values():rng.shuffle(values)
        for i in range(config.calibration_samples):
            available=[g for g in groups if reservoirs[g]]
            if not available:break
            calibration.append(reservoirs[available[i%len(available)]].pop())
        dump_json(output/'calibration.json',calibration)
        if len(calibration)<config.calibration_samples:warnings.append('calibration_smaller_than_requested')
        if counts['external']<100:warnings.append('small_external_selection_smoke_test_not_training_efficacy_evidence')
        result={'counts':dict(counts),'calibration_records':len(calibration),'calibration_groups_seen':dict(seen),
                'warnings':warnings,'tokenizer':tokenizer.info,'holdouts_used_for_training':False,
                'supervision':'all assistant spans; user/system/tool responses masked; no truncation',
                'files':{name:file_hash(output/name) for name in ('tokens.jsonl','calibration.json','provenance.jsonl')}}
        dump_json(output/'complete.json',result);return result
    finally:gate.close()
