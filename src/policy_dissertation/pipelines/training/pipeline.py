"""Resumable LangGraph: validated train inputs → MLP pruning → recovery → NF4 export."""
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path
from time import monotonic
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from policy_dissertation.errors import DissertationError
from policy_dissertation.data.training_inputs import pinned_enterprise_settings
from policy_dissertation.pipelines.dataset_preparation.tokenization import load_tokenizer
from policy_dissertation.pipelines.external_datasets.pipeline import file_hash
from policy_dissertation.pipelines.policy_transformation.data_io import dump_json,digest,atomic_text
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from .backend import CudaBackend
from .inputs import snapshot_inputs,prepare_inputs

VERSION='qwen3-compression-v1'


class GraphState(TypedDict):
    output_directory: str
    completed_stage: str


def hashes(folder):
    return {str(p.relative_to(folder)):file_hash(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def verify_stage(folder,evidence):
    for name,expected in evidence['files'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file() or file_hash(path)!=expected:
            raise DissertationError('Committed compression stage checksum mismatch; do not reuse this checkpoint.')


def run_compression(enterprise,native,output,settings,config,*,resume=False,check_only=False,
                    progress=print,backend=None,tokenizer=None,provider=None,research_report=Path('report.md')):
    enterprise=Path(enterprise).resolve();native=Path(native).resolve();output=Path(output).resolve()
    for source in (enterprise,native):
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise DissertationError('Compression output must be separate from all input directories.')
    backend=backend or CudaBackend(settings,progress)
    started=monotonic()
    with output_lock(output):
        checkpoint=output/'checkpoint.json'
        if not resume and any((output/n).exists() for n in ('report.json','checkpoint.json','model_4bit')):
            raise DissertationError('Compression output already exists. Use --resume with identical inputs/settings or a new directory.')
        sources=snapshot_inputs(enterprise,native)
        configured=pinned_enterprise_settings(enterprise,settings)
        revision=configured.pipeline4.tokenizer_revision
        if config.model_revision is not None and config.model_revision!=revision:
            raise DissertationError('Model revision must match the immutable enterprise tokenizer revision.')
        config=config.model_copy(update={'model_revision':revision})
        if len(revision) not in (40,64) or any(c not in '0123456789abcdef' for c in revision):
            raise DissertationError('An immutable Qwen model revision is required.')
        code={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
        fingerprint=digest({'version':VERSION,'sources':sources,'configuration':config.model_dump(),'implementation':code})
        state=json.loads(checkpoint.read_text()) if checkpoint.exists() and not check_only else {'fingerprint':fingerprint,'stages':{}}
        if state.get('fingerprint')!=fingerprint:
            raise DissertationError('Compression inputs/settings/code changed. Use a new output directory; existing models are preserved.')
        report={'schema_version':VERSION,'status':'RUNNING','configuration':config.model_dump(),
                'input_hashes':sources,'fingerprint':fingerprint,'stages':{},'errors':[], 'warnings':[],
                'training_started':False,'final_model_ready':False,'semantic_quality_verified':False,
                'evaluation_pending':True,'check_only':check_only,'started_at':datetime.now(timezone.utc).isoformat()}
        dump_json(output/'report.json',report)
        try:
            progress('[1/5] Checking Linux/CUDA runtime and capacity before downloading model weights')
            runtime=backend.preflight(output,config);report['runtime']=runtime
            if state.get('versions') and state['versions']!=runtime['versions']:
                raise DissertationError('Compression runtime versions changed; resume refused.')
            state['versions']=runtime['versions']
            if check_only:
                report.update(status='PREFLIGHT_PASSED',final_model_ready=False)
                return report
            dump_json(checkpoint,state)
            tokenizer=tokenizer or load_tokenizer(output,configured.pipeline4,
                settings.hf_token.get_secret_value() if settings.hf_token else None,progress)
            if tokenizer.info.get('revision')!=revision or tokenizer.info.get('model')!=config.model_id:
                raise DissertationError('Model and tokenizer provenance mismatch.')
            stage_root=output/'_runs'/fingerprint
            def stage(name,operation):
                folder=output/'model_4bit' if name=='export' and (output/'model_4bit').exists() else stage_root/name
                if name in state['stages']:
                    verify_stage(folder,state['stages'][name]);progress('Reusing committed '+name+' stage')
                    evidence=state['stages'][name]['evidence']
                else:
                    if folder.exists() and name!='recovery':
                        if folder.is_symlink():raise DissertationError('Symlinked compression stage refused.')
                        shutil.rmtree(folder)
                    folder.mkdir(parents=True,exist_ok=True)
                    evidence=operation(folder)
                    if name=='export' and evidence.get('reload_verified') is not True:
                        raise DissertationError('Final NF4 checkpoint reload is unverified.')
                    state['stages'][name]={'files':hashes(folder),'evidence':evidence}
                    dump_json(checkpoint,state)
                report['stages'][name]=evidence;dump_json(output/'report.json',report)
                return folder,evidence
            def inputs_node(_):
                progress('[2/5] Revalidating enterprise holdouts and exact approved NVIDIA selection; caching supervised train tokens')
                _,info=stage('inputs',lambda folder:prepare_inputs(enterprise,native,folder,configured,config,tokenizer,
                                                               provider=provider,progress=progress))
                report['warnings']=info['warnings']
                return {'completed_stage':'inputs'}
            def prune_node(_):
                progress('[3/5] Measuring calibration activations and pruning gated-MLP channels')
                stage('pruned',lambda folder:backend.prune(folder,stage_root/'inputs',tokenizer,config))
                return {'completed_stage':'pruned'}
            def recover_node(_):
                progress('[4/5] Recovering and specializing pruned model with BF16 LoRA; merging adapter')
                report['training_started']=True;dump_json(output/'report.json',report)
                stage('recovery',lambda folder:backend.recover(folder,stage_root/'pruned',stage_root/'inputs',tokenizer,config,resume=resume))
                return {'completed_stage':'recovery'}
            def export_node(_):
                progress('[5/5] Exporting NF4 4-bit and testing the saved checkpoint reload')
                _,evidence=stage('export',lambda folder:backend.quantize(folder,stage_root/'recovery'/'merged',stage_root/'inputs',tokenizer,config))
                if evidence.get('reload_verified') is not True:raise DissertationError('Final NF4 checkpoint reload is unverified.')
                return {'completed_stage':'export'}
            graph=StateGraph(GraphState)
            for name,node in (('inputs',inputs_node),('prune',prune_node),('recover',recover_node),('export',export_node)):graph.add_node(name,node)
            graph.add_edge(START,'inputs');graph.add_edge('inputs','prune');graph.add_edge('prune','recover');graph.add_edge('recover','export');graph.add_edge('export',END)
            graph.compile().invoke({'output_directory':str(output),'completed_stage':''})
            if sources!=snapshot_inputs(enterprise,native):raise DissertationError('Source artifacts changed during compression; export not approved.')
            final=output/'model_4bit'
            if not final.exists(): (stage_root/'export').rename(final)
            if not final.is_dir() or final.is_symlink():
                raise DissertationError('Final model directory is missing or symlinked.')
            report.update(status='COMPLETED_WITH_WARNINGS' if report['warnings'] else 'COMPLETED',
                          final_model_ready=True,model_directory=str(final),
                          original_parameters=report['stages']['pruned']['original_parameters'],
                          pruned_parameters=report['stages']['pruned']['pruned_parameters'],
                          export_bytes=sum(p.stat().st_size for p in final.rglob('*') if p.is_file()))
            return report
        except BaseException as exc:
            report.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',final_model_ready=False)
            report['errors'].append(str(exc) if isinstance(exc,DissertationError) else 'compression_stage_failed:'+type(exc).__name__)
            raise DissertationError('Compression did not complete; inspect report.json and resume matching checkpoints on Linux/CUDA.') from None
        finally:
            report['finished_at']=datetime.now(timezone.utc).isoformat();report['duration_seconds']=round(monotonic()-started,2)
            dump_json(output/'report.json',report)
            backend.release()
            if research_report is not None:
                try:
                    path=Path(research_report)
                    text=path.read_text(encoding='utf-8') if path.exists() else ''
                    block=f"\n\n## Compression execution — {report['finished_at']}\n\nStatus: {report['status']}; training_started={report['training_started']}; final_model_ready={report['final_model_ready']}. Evaluation pending; no semantic correctness claim. Evidence: `{output/'report.json'}`.\n"
                    atomic_text(path,text+block)
                except OSError:progress('Could not append research report; execution report remains available.')
