"""Read-only validation of a native preparation plan; never generate or train."""
from collections import Counter
from itertools import islice
import json
from pathlib import Path
import uuid

from policy_dissertation.config import Pipeline3Settings,Pipeline4Settings
from policy_dissertation.data.hub import protect_exports
from policy_dissertation.data.training_inputs import TrainingInputGate,index_enterprise
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.dataset_preparation.tokenization import load_tokenizer
from policy_dissertation.pipelines.external_datasets.pipeline import now,file_hash
from policy_dissertation.pipelines.native_datasets.replay import iter_prepared_native
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock


def validate_native_plan(plan_directory,enterprise,output,settings,*,limit_records=None,overwrite=False,
                         tokenizer=None,provider=None,progress=lambda text:None):
    plan_directory,enterprise,output=(Path(p).resolve() for p in (plan_directory,enterprise,output))
    if limit_records is not None and (type(limit_records) is not int or limit_records<1):
        raise DissertationError('Validation record limit must be a positive integer.')
    for source in (plan_directory,enterprise):
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise DissertationError('Validation output must not overlap source artifacts.')
    with output_lock(output):
        protect_exports(output,overwrite)
        report={'schema_version':'native-validation-v1','status':'RUNNING','started_at':now(),
                'records_checked':0,'full_plan_validated':False,'errors':[],'training_started':False,
                'llm_calls':0,'external_tests_locally_executed':0,'limit_records':limit_records}
        dump_json(output/'report.json',report)
        gate=None;iterator=None
        try:
            plan=json.loads((plan_directory/'source_plan.json').read_text())
            if not plan.get('data_prepared'):raise DissertationError('Native plan is not prepared; run external installation first.')
            report.update(source_plan_run_id=plan['run_id'],expected_records=plan['accepted_examples'],
                          plan_hash=file_hash(plan_directory/'source_plan.json'))
            configured=settings.model_copy(deep=True)
            configured.pipeline3=Pipeline3Settings.model_validate(plan['normalization'])
            configured.pipeline4=Pipeline4Settings.model_validate(plan['quality'])
            token=configured.hf_token.get_secret_value() if configured.hf_token else None
            tokenizer=tokenizer or load_tokenizer(output,configured.pipeline4,token,progress)
            gate=TrainingInputGate(output/'_runs'/uuid.uuid4().hex/'fingerprints.sqlite3',configured.pipeline4.near_threshold)
            progress('Validating all enterprise splits and registering held-out examples')
            index_enterprise(enterprise,configured,tokenizer,plan['max_tokens'],gate)
            gate.seal_holdouts()
            progress('Replaying approved NVIDIA rows and checking source hashes, exact tokens and leakage')
            iterator=iter_prepared_native(plan_directory,configured,gate,tokenizer=tokenizer,provider=provider)
            partial=limit_records is not None and limit_records<plan['accepted_examples']
            selected=islice(iterator,limit_records) if partial else iterator
            sources=Counter();statuses=Counter()
            for record in selected:
                report['records_checked']+=1;sources[record['source']]+=1;statuses[record['verification']['status']]+=1
                if report['records_checked']==1 or report['records_checked']%100==0:
                    progress(f"  Validated {report['records_checked']}/{min(limit_records,plan['accepted_examples']) if partial else plan['accepted_examples']}")
            report.update(full_plan_validated=not partial and report['records_checked']==plan['accepted_examples'],
                          sources=dict(sources),verification_counts=dict(statuses),tokenizer=tokenizer.info,
                          scope='sample only' if partial else 'entire approved native selection; not full upstream corpora',
                          status='COMPLETED_WITH_WARNINGS' if partial else 'COMPLETED')
            if not report['records_checked']:raise DissertationError('No approved records were validated.')
            if file_hash(plan_directory/'source_plan.json')!=report['plan_hash']:
                raise DissertationError('Native source plan changed during validation.')
        except (Exception,KeyboardInterrupt) as exc:
            report.update(status='FAILED',full_plan_validated=False)
            report['errors'].append(str(exc) if isinstance(exc,DissertationError) else 'validation_failed:'+type(exc).__name__)
        finally:
            for resource in (iterator,gate):
                if resource is not None:
                    try:resource.close()
                    except Exception as exc:
                        report.update(status='FAILED',full_plan_validated=False)
                        report['errors'].append('validation_cleanup_failed:'+type(exc).__name__)
            report['finished_at']=now()
            dump_json(output/'report.json',report)
        return report
