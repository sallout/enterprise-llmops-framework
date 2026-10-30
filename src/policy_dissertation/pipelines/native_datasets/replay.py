"""Replay a validated native selection without a materialized training dataset."""
import json
from importlib.metadata import version
from pathlib import Path
import sqlite3

from policy_dissertation.config import Pipeline3Settings, Pipeline4Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.data.training_inputs import validate_native_record
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from policy_dissertation.pipelines.external_datasets.sources import HubProvider
from policy_dissertation.pipelines.external_datasets.sampling import sampled
from .pipeline import VERSION, decision_digest


def iter_prepared_native(plan_path,settings,gate,*,tokenizer,provider=None):
    """Yield only committed accepted rows, checking pinned replay and a fresh global gate.

    The caller must first index the exact selected enterprise splits/all other chosen
    holdouts in `gate` and seal it. This function does not start model training.
    """
    root=Path(plan_path).resolve()
    if root.is_file():root=root.parent
    plan=json.loads((root/'source_plan.json').read_text())
    if plan.get('schema_version')!=VERSION or not plan.get('data_prepared') or plan.get('status')=='FAILED':
        raise DissertationError('Native source plan is not validated/prepared.')
    if gate.threshold!=plan['quality']['near_threshold']:
        raise DissertationError('Replay leakage threshold must match the validated source plan.')
    if not gate.sealed:raise DissertationError('Register all selected holdouts and seal the fresh gate before replay.')
    if not any(s.get('split_hashes')==plan['enterprise_split_hashes'] for s in gate.sources if s.get('type')=='enterprise'):
        raise DissertationError('Replay gate must index the same enterprise split hashes as the source plan.')
    for key in ('model','revision','chat_template_hash'):
        if tokenizer.info.get(key)!=plan['tokenizer'].get(key):raise DissertationError('Native replay tokenizer/template changed.')
    if any(version(name)!=value for name,value in plan['versions'].items()):
        raise DissertationError('Native dataset runtime versions changed; prepare a new source plan.')
    # Reject stale adapter code: a new normalization implementation needs a new audit.
    package=Path(__file__).parents[2]
    if any(not (package/name).is_file() or file_hash(package/name)!=h for name,h in plan['adapter_code_hashes'].items()):
        raise DissertationError('Native adapter implementation changed; prepare a new source plan.')
    db_path=(root/plan['decision_database']).resolve()
    if not db_path.is_relative_to(root) or file_hash(db_path)!=plan['decision_database_hash']:
        raise DissertationError('Native decision database checksum/path mismatch.')
    db=sqlite3.connect(db_path.as_uri()+'?mode=ro',uri=True)
    try:
        if decision_digest(db)!=plan['decisions_digest']:raise DissertationError('Native selection digest mismatch.')
        configured=settings.model_copy(deep=True)
        configured.pipeline3=Pipeline3Settings.model_validate(plan['normalization'])
        configured.pipeline4=Pipeline4Settings.model_validate(plan['quality'])
        token=configured.hf_token.get_secret_value() if configured.hf_token else None
        provider=provider or HubProvider(configured.pipeline3,token)
        emitted=0
        for source in plan['sources']:
            # A quota-zero source has no approved rows and requires no network reads.
            last=db.execute("SELECT max(position) FROM native_decisions WHERE source=? AND json_extract(data,'$.accepted')=1",(source['alias'],)).fetchone()[0]
            if last is None:continue
            stream=sampled(provider.stream(source),plan['seed'],configured.pipeline3,lambda *_:None)
            try:
                for position,text in db.execute('SELECT position,data FROM native_decisions WHERE source=? AND position<=? ORDER BY position',(source['alias'],last)):
                    expected=json.loads(text)
                    try:item=next(stream)
                    except StopIteration:raise DissertationError('Native stream ended before planned accepted rows.') from None
                    if item['ordinal']!=expected['ordinal'] or item['raw_hash']!=expected['raw_hash']:
                        raise DissertationError('Pinned native row content/order changed; replay refused.')
                    if not expected['accepted']:continue
                    record=validate_native_record(item['raw'],source,item['ordinal'],configured,tokenizer,plan['max_tokens'])
                    if (record['id']!=expected['id'] or record['metadata']['preparation']['token_count']!=expected['token_count'] or
                        record['metadata']['preparation'].get('rendered_token_hash')!=expected['rendered_token_hash']):
                        raise DissertationError('Native normalized content/tokenization changed; replay refused.')
                    gate.accept_training(record)
                    emitted+=1
                    yield record
            finally:stream.close()
        if emitted!=plan['accepted_examples']:raise DissertationError('Native replay accepted count mismatch.')
    finally:db.close()
