"""Direct native NVIDIA iterators and a shared disk-backed cross-source leakage gate.

No trainer, model weights, merged NVIDIA export, or optimizer is constructed here.
All selected holdouts must be indexed before training iterators may be consumed.
"""
from collections import Counter
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from policy_dissertation.data.adapters import adapt, InvalidExample
from policy_dissertation.data.schemas import UnifiedRecord
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.dataset_preparation.deduplication import features, candidates, unpack
from policy_dissertation.pipelines.dataset_preparation.quality import scan_sensitive, validate_tokens
from policy_dissertation.pipelines.external_datasets.normalization import normalize, RejectedRecord
from policy_dissertation.pipelines.external_datasets.sources import HubProvider, inspected_source, retry
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash


class TrainingInputGate:
    """Index fingerprints, never full conversations; fail closed on cross-split similarity.

    This is a per-run audit, not a resumable training cursor. Reusing a nonempty index
    is refused. Register all selected enterprise/external evaluation sets, then seal.
    Future external holdouts cannot be silently introduced after training starts.
    """
    def __init__(self, path, threshold=.9, *, resume=False, autocommit=True):
        if not .5 <= threshold <= 1:raise ValueError('Invalid near-duplicate threshold')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists() and not resume:raise DissertationError('Use a fresh training-input audit database.')
        self.db=sqlite3.connect(self.path)
        self.db.row_factory=sqlite3.Row
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS records(rid INTEGER PRIMARY KEY,exact TEXT,formatted TEXT,shingles BLOB,split TEXT,component TEXT);
          CREATE INDEX IF NOT EXISTS exacts ON records(exact);
          CREATE INDEX IF NOT EXISTS components ON records(component);
          CREATE TABLE IF NOT EXISTS prefixes(token BLOB,rid INTEGER,n INTEGER,PRIMARY KEY(token,rid));
          CREATE TEMP TABLE candidates(rid INTEGER PRIMARY KEY);
        ''')
        self.autocommit=autocommit
        if self.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':raise DissertationError('Invalid fingerprint database.')
        self.threshold=threshold;self.sealed=False;self.counts=Counter();self.sources=[]

    def seal_holdouts(self):self.sealed=True

    def _check(self,record,split):
        exact,formatted,values=features(record);current=set(values)
        component=record['metadata']['component_group']
        for other in self.db.execute('SELECT * FROM records WHERE exact=? OR component=?',(exact,component)):
            if other['split']!=split:raise InvalidExample('cross_source_split_leakage')
            if other['exact']==exact:raise InvalidExample('duplicate_conversation')
        prefix=candidates(SimpleNamespace(db=self.db),values,self.threshold)
        for other in self.db.execute('SELECT r.* FROM records r JOIN candidates c ON r.rid=c.rid'):
            previous=unpack(other['shingles']);similarity=len(previous & current)/len(previous | current)
            if similarity+1e-12>=self.threshold:
                if other['split']!=split:raise InvalidExample('cross_source_near_split_leakage')
                if other['formatted']==formatted:raise InvalidExample('duplicate_format_conversation')
        rid=self.db.execute('INSERT INTO records(exact,formatted,shingles,split,component) VALUES (?,?,?,?,?)',
            (exact,formatted,b''.join(values),split,component)).lastrowid
        self.db.executemany('INSERT INTO prefixes VALUES (?,?,?)',((v,rid,len(values)) for v in prefix))
        if self.autocommit:self.db.commit()
        self.counts['indexed_'+split]+=1
        return rid

    def register(self,record,split):
        if self.sealed:raise DissertationError('Register all holdouts before sealing the training-input gate.')
        if split not in ('train','validation','test'):raise ValueError('Unknown split')
        return self._check(record,split)

    def accept_training(self,record):
        if not self.sealed:raise DissertationError('Index all selected holdouts and seal the gate before reading training records.')
        return self._check(record,'train')

    def close(self):self.db.close()


def index_enterprise(folder,settings,tokenizer,maximum,gate):
    """Validate prepared enterprise exports and preload ALL splits before external data."""
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    report=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else {
        'training_ready':manifest.get('training_ready'),'preparation':{'tokenizer':manifest.get('tokenizer')},
        'output_hashes':manifest.get('output_hashes',{}),'splits':manifest.get('splits',{}),
        'source_run_complete':manifest.get('source_run_complete',manifest.get('source_run',{}).get('complete',True))}
    if not report.get('training_ready') or not manifest.get('training_ready'):
        raise DissertationError('Enterprise dataset is not prepared/training-ready.')
    info=report['preparation']['tokenizer']
    for field in ('model','revision','chat_template_hash'):
        if info.get(field)!=tokenizer.info.get(field):raise DissertationError('Enterprise tokenizer/template does not match training loader.')
    secrets=[s.get_secret_value() for s in (settings.hf_token,settings.model.api_key) if s]
    # All three splits must be checked before callers are allowed to seal and stream.
    for split in ('validation','test','train'):
        path=folder/(split+'.jsonl')
        actual=file_hash(path)
        if actual!=report['output_hashes'].get(path.name) or actual!=manifest['output_hashes'].get(path.name):
            raise DissertationError('Enterprise split checksum mismatch.')
        count=0
        with path.open() as stream:
            for line in stream:
                scan_sensitive(line,settings.pipeline4,secrets)
                raw=json.loads(line);UnifiedRecord.model_validate(raw)
                record=adapt(raw,raw['metadata']['pipeline'],{'stage':'training_input_validation'})
                validate_tokens(record,tokenizer,maximum,settings.pipeline4)
                if raw['metadata'].get('training_split') != split:raise DissertationError('Enterprise split label mismatch.')
                gate.register(record,split)
                count+=1
        if count != report['splits'].get(split):raise DissertationError('Enterprise split count mismatch.')
    gate.sources.append({'type':'enterprise','manifest_hash':file_hash(folder/'manifest.json'),
                         'splits':report['splits'],'split_hashes':{name:file_hash(folder/(name+'.jsonl')) for name in ('train','validation','test')},'source_run_complete':report.get('source_run_complete',True)})


def iter_nvidia(alias,settings,tokenizer,maximum,gate,*,config=None,revision=None,max_records=None,
                provider=None,progress=lambda text:None):
    """Yield validated canonical training inputs directly from a pinned native stream.

    Native data is not rewritten to JSONL. Closed marked reasoning removal follows
    explicit Pipeline 3 settings; original tests remain externally reported evidence.
    Limits count decoded rows, not accepted rows. Do not interpret a bounded preview
    as a complete corpus leakage audit. A global gate is mandatory across sources.
    """
    if alias not in ('nemotron','opencode'):raise DissertationError('Unsupported NVIDIA source.')
    if max_records is not None and max_records<1:raise ValueError('Record cap must be positive')
    if not gate.sealed:raise DissertationError('Holdouts must be indexed before streaming NVIDIA training data.')
    token=settings.hf_token.get_secret_value() if settings.hf_token else None
    provider=provider or HubProvider(settings.pipeline3,token)
    info=retry(lambda:provider.inspect(alias,revision),settings.pipeline3,progress)
    source=inspected_source(info,config or ('chat' if alias=='nemotron' else 'train'),'train')
    gate.sources.append({k:source[k] for k in ('dataset','config','split','revision','license','license_url')})
    secrets=[s.get_secret_value() for s in (settings.hf_token,settings.model.api_key) if s]
    stream=None
    try:
        stream=provider.stream(source)
        from itertools import islice
        selected=islice(stream,max_records) if max_records is not None else stream
        for ordinal,raw in enumerate(selected):
            gate.counts['external_decoded']+=1
            try:
                record=validate_native_record(raw,source,ordinal,settings,tokenizer,maximum)
                gate.accept_training(record)
            except (InvalidExample,RejectedRecord) as exc:
                gate.counts['rejected:'+str(exc)]+=1
                continue
            except (ValueError,KeyError,TypeError):
                gate.counts['rejected:invalid_native_schema']+=1
                continue
            for change in record['metadata']['provenance'].get('transformations',[]):
                gate.counts['normalization:'+change['action']]+=1
            gate.counts['external_accepted']+=1
            yield record
    finally:
        if stream is not None and hasattr(stream,'close'):stream.close()


def validate_native_record(raw,source,ordinal,settings,tokenizer,maximum):
    """Shared non-generative adapter/filter boundary; no gate mutation or persistence."""
    text=json.dumps(raw,ensure_ascii=False,allow_nan=False)
    if len(text.encode())>settings.pipeline3.max_record_bytes:raise InvalidExample('oversized_raw_record')
    secrets=[s.get_secret_value() for s in (settings.hf_token,settings.model.api_key) if s]
    scan_sensitive(text,settings.pipeline4,secrets)
    normalized=normalize(raw,source,ordinal,settings.pipeline3)
    record=adapt(normalized,'pipeline3',{'stage':'native_training_adapter','ordinal':ordinal})
    validate_tokens(record,tokenizer,maximum,settings.pipeline4)
    if record['metadata'].get('official_holdout'):
        raise DissertationError('Declared external holdout encountered in training source; pre-register all holdouts before training.')
    return record


def pinned_enterprise_settings(folder,settings):
    """Use the enterprise export's immutable tokenizer revision on a new machine."""
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    report=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else None
    if not manifest.get('training_ready') or (report is not None and not report.get('training_ready')):
        raise DissertationError('Enterprise dataset is not prepared/training-ready.')
    info=report.get('preparation',{}).get('tokenizer') if report else manifest.get('tokenizer')
    if not isinstance(info,dict) or info.get('model')!=settings.pipeline4.tokenizer_model or not info.get('revision'):
        raise DissertationError('Enterprise tokenizer provenance is missing or incompatible.')
    configured=settings.model_copy(deep=True)
    configured.pipeline4.tokenizer_revision=info['revision']
    return configured
