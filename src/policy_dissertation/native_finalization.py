"""Publish an explicitly bounded copy of committed native decisions, without streaming.

Kept outside adapter-hashed packages so adding this command does not invalidate
existing preparation fingerprints or silently change normalization.
"""
import json
import shutil
import sqlite3
from pathlib import Path

from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.native_datasets.pipeline import NativePipeline, VERSION
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json, digest
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash


def finalize_native_selection(source, enterprise, output, settings, *, max_tokens=4096, seed=42,
                              tokenizer=None, provider=None, progress=print):
    source, enterprise, output = (Path(p).resolve() for p in (source,enterprise,output))
    if not source.is_dir():
        raise DissertationError('Native checkpoint input does not exist.')
    if any(output==p or output.is_relative_to(p) or p.is_relative_to(output) for p in (source,enterprise)):
        raise DissertationError('Use a separate new output directory; original checkpoints are preserved.')
    if output.exists() and any(output.iterdir()):
        raise DissertationError('Partial-selection output must be empty/new.')
    with output_lock(source), output_lock(output):
        original = json.loads((source/'report.json').read_text())
        if original.get('schema_version') != VERSION:
            raise DissertationError('Input is not a native NVIDIA preparation run.')
        if original.get('errors') and original['errors'] != ['native_stage_failed:KeyboardInterrupt']:
            raise DissertationError('Only interruption can be finalized; other source failures require investigation.')
        if original.get('status') not in ('RUNNING','FAILED'):
            raise DissertationError('Input already completed; use its existing validated plan.')
        candidates = list((source/'_runs').glob('*/fingerprints.sqlite3'))
        if len(candidates)!=1:
            raise DissertationError('Expected one unambiguous checkpoint run; multiple runs need explicit investigation.')
        db_source=candidates[0]
        if db_source.is_symlink():
            raise DissertationError('Symlink checkpoint refused.')
        db=sqlite3.connect(db_source.as_uri()+'?mode=ro',uri=True)
        try:
            if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':
                raise DissertationError('SQLite checkpoint integrity check failed.')
            aliases=[r[0] for r in db.execute('SELECT DISTINCT source FROM native_decisions ORDER BY source')]
            accepted=db.execute("SELECT count(*) FROM native_decisions WHERE json_extract(data,'$.accepted')=1").fetchone()[0]
            if not accepted:
                raise DissertationError('No committed accepted records to finalize.')
            work=output/'_runs'/db_source.parent.name;work.mkdir(parents=True)
            copy=sqlite3.connect(work/'fingerprints.sqlite3')
            try:db.backup(copy)
            finally:copy.close()
        finally:db.close()
        for directory in ('_metadata','_tokenizer'):
            tree=source/directory
            if tree.exists():
                if tree.is_symlink() or any(p.is_symlink() for p in tree.rglob('*')):
                    raise DissertationError('Symlink source metadata/tokenizer assets refused.')
                shutil.copytree(tree,output/directory)
        # Preserve original error/status evidence; never rewrite the source report.
        dump_json(output/'interrupted_report.json',original)
        engine=NativePipeline(enterprise,output,settings,original['max_samples'],tuple(aliases),seed,
                              None,None,None,None,tokenizer,provider,progress)
        engine.max_tokens=max_tokens
        try:
            # Cached metadata/tokenizer and checkpoint decisions only. No process()/stream() call.
            if engine.report['versions']!=original['versions']:
                raise DissertationError('Runtime changed since ingestion; restore original locked environment.')
            for alias in aliases:
                if not (output/'_metadata'/(digest([alias,None])+'.json')).is_file():
                    raise DissertationError('Original cached source pin missing; no network rediscovery allowed.')
            engine.discover({})
            if engine.report['run_id']!=db_source.parent.name:
                raise DissertationError('Original settings/code/source fingerprint mismatch. No selection was approved.')
            engine.report['warnings'].append('user_finalized_committed_partial_selection')
            engine.report['partial_selection']=True
            engine.report['original_requested_max_samples']=original['max_samples']
            engine.report['source_report_status']=original['status']
            engine.finalize()
            report=engine.report
            if not report['data_prepared'] or report['accepted_examples']!=accepted:
                raise DissertationError('Partial selection counts or preparation validation failed.')
            for stat in report['source_statistics']:
                if stat['stop_reason']=='failed_or_interrupted':stat['stop_reason']='user_finalized_partial_selection'
            plan=json.loads((output/'source_plan.json').read_text())
            plan['partial_selection']=True
            plan['original_requested_max_samples']=original['max_samples']
            plan['interrupted_source_report_hash']=file_hash(source/'report.json')
            dump_json(output/'source_plan.json',plan)
            dump_json(output/'report.json',report)
            return report
        except (Exception,KeyboardInterrupt):
            if engine.gate:engine.gate.close();engine.gate=None
            dump_json(output/'source_plan.json',{'schema_version':VERSION,'status':'FAILED','data_prepared':False})
            dump_json(output/'report.json',{'schema_version':VERSION,'status':'FAILED','data_prepared':False,
                                         'errors':['partial_finalization_failed'],'original_inputs_preserved':True})
            raise
