"""Embedded enterprise finalization using the established preparation engine."""
import json
import shutil
from pathlib import Path
from policy_dissertation.data.hub import protect_exports
from policy_dissertation.data.sources import Source
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json, atomic_text
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash, write_jsonl
from policy_dissertation.pipelines.dataset_preparation.pipeline import run_pipeline4


def prepare_enterprise(source, output, settings, *, maximum=4096, seed=42, overwrite=False,
                       tokenizer=None, progress=lambda text:None, research_report=None, in_place=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_file():source=source.parent
    if source==output and not in_place:
        raise DissertationError('Choose a new prepared output directory; existing source data is preserved.')
    if source!=output and (source.is_relative_to(output) or output.is_relative_to(source)):
        raise DissertationError('Prepared output and source directories must not overlap.')
    original=json.loads((source/'report.json').read_text())
    if original.get('pipeline') != 'enterprise' or original.get('status') not in ('COMPLETED','COMPLETED_WITH_WARNINGS') or original.get('errors'):
        raise DissertationError('Enterprise preparation requires a successful accepted enterprise export.')
    training=source/'training.jsonl'
    if file_hash(training)!=original.get('output_hashes',{}).get('training.jsonl'):
        raise DissertationError('Enterprise input checksum mismatch.')
    with output_lock(output):
        if not in_place:protect_exports(output,overwrite)
        elif not overwrite and any((output/(s+'.jsonl')).exists() for s in ('train','validation','test')):
            raise DissertationError('Prepared splits already exist; explicit --overwrite is required.')
        # An immutable accepted snapshot avoids report-hash cycles when root readiness is updated.
        snapshot_report={'status':original['status'],'errors':[], 'accepted_examples':original['accepted_examples'],
            'source_run_complete':original.get('source_run_complete',True),
            'output_hashes':{'training.jsonl':file_hash(training)}}
        snapshot=output/'_accepted'/digest(snapshot_report)
        snapshot.mkdir(parents=True,exist_ok=True)
        if not (snapshot/'training.jsonl').exists():shutil.copyfile(training,snapshot/'training.jsonl')
        if file_hash(snapshot/'training.jsonl')!=snapshot_report['output_hashes']['training.jsonl']:
            raise DissertationError('Accepted snapshot checksum mismatch.')
        dump_json(snapshot/'report.json',snapshot_report)
        configured=settings.model_copy(deep=True)
        configured.pipeline4.mix_ratios={'enterprise':1.0}
        prepared=run_pipeline4(output/'_unused_legacy_inputs',output/'_preparation',configured,maximum,
            seed=seed,overwrite=True,sources=[Source(name='enterprise',type='local',path=str(snapshot/'training.jsonl'),weight=1)],
            tokenizer=tokenizer,progress=progress,research_report=output/'preparation-research.md')
        result=dict(original)
        result['output_directory']=str(output)
        result['generation_accepted_examples']=original['accepted_examples']
        result['preparation']={k:prepared.get(k) for k in ('status','training_ready','input_records','validated_examples',
            'rejected_examples','rejection_reasons','duplicates_removed','splits','splitting','tokenizer','token_statistics','distribution','errors','warnings')}
        result['training_ready']=bool(prepared['training_ready'])
        result['cross_source_audit']='required at training-input loading against all selected external holdouts; not performed by enterprise-only preparation'
        if prepared['status']=='FAILED':
            result['status']='FAILED';result['errors']=list(original.get('errors',[]))+['enterprise_preparation_failed']
            # Fail closed, but do not destroy the accepted source if tokenization/splitting fails.
            if source!=output:shutil.copyfile(training,output/'training.jsonl')
        else:
            result['status']='COMPLETED_WITH_WARNINGS' if prepared['status']=='COMPLETED_WITH_WARNINGS' or original['status']=='COMPLETED_WITH_WARNINGS' else 'COMPLETED'
            for split in ('train','validation','test'):
                def records(split=split):
                    with (output/'_preparation'/(split+'.jsonl')).open() as stream:
                        for line in stream:
                            row=json.loads(line)
                            row['metadata']['training_split']=split
                            yield row
                write_jsonl(output/(split+'.jsonl'),records())
            def accepted():
                for split in ('train','validation','test'):
                    with (output/(split+'.jsonl')).open() as stream:
                        for line in stream:yield json.loads(line)
            write_jsonl(output/'training.jsonl',accepted())
            result['generation_domain_counts']=original.get('domain_counts',{})
            result['generation_verification_counts']=original.get('verification_counts',{})
            result['domain_counts']=prepared.get('distribution',{}).get('domain',{})
            result['verification_counts']=prepared.get('distribution',{}).get('status',{})
            result['accepted_examples']=prepared['accepted_examples']
            result['splits']=prepared['splits']
        if source!=output and (source/'rejected.jsonl').exists():shutil.copyfile(source/'rejected.jsonl',output/'rejected.jsonl')
        shutil.copyfile(output/'_preparation'/'rejected.jsonl',output/'preparation_rejected.jsonl')
        result['output_hashes']={p.name:file_hash(p) for p in (output/n for n in
            ('training.jsonl','train.jsonl','validation.jsonl','test.jsonl','rejected.jsonl','preparation_rejected.jsonl')) if p.exists()}
        manifest={'pipeline':'enterprise','schema_version':'unified-qwen-sft-v1','status':result['status'],
            'accepted_examples':result['accepted_examples'],'training_ready':result['training_ready'],
            'source_run_complete':original.get('source_run_complete',True),'output_hashes':result['output_hashes'],
            'splits':result.get('splits',{}),'tokenizer':prepared['tokenizer'],
            'accepted_snapshot':str(snapshot),'cross_source_audit':result['cross_source_audit'],
            'license':json.loads((source/'manifest.json').read_text()).get('license','Original source restrictions apply; no license inferred.') if (source/'manifest.json').exists() else 'unspecified'}
        if research_report:
            path=Path(research_report)
            marker=digest([str(output),result['output_hashes'],maximum,seed])
            start,end=f'<!-- enterprise-preparation:{marker} -->',f'<!-- /enterprise-preparation:{marker} -->'
            block=(f"{start}\n## Embedded enterprise preparation\n\n"
                f"Status: {result['status']}; enterprise-only training ready: {result['training_ready']}. "
                f"Accepted input: {original['accepted_examples']}; prepared: {result['accepted_examples']}; "
                f"preparation rejected: {prepared['rejected_examples']}; duplicates: {prepared['duplicates_removed']}; "
                f"splits: {json.dumps(result.get('splits',{}))}. Exact Qwen limit: {maximum}. "
                f"Cross-source external audit remains required; no training started.\n{end}")
            try:
                text=path.read_text() if path.exists() else '# Research Report\n'
                if start in text and end in text:
                    a,b=text.index(start),text.index(end)+len(end);text=text[:a]+block+text[b:]
                else:text=text.rstrip()+'\n\n'+block+'\n'
                atomic_text(path,text)
            except OSError:
                result['status']='FAILED';result['training_ready']=False
                result.setdefault('errors',[]).append('research_report_update_failed')
                manifest.update(status='FAILED',training_ready=False)
        dump_json(output/'manifest.json',manifest)
        dump_json(output/'report.json',result)
        return result
