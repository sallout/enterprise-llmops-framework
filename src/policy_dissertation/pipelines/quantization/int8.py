"""Post-training LLM.int8 export from committed merged BF16; no training."""
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict, Field
from policy_dissertation.errors import DissertationError
from policy_dissertation.pipelines.policy_transformation.pipeline import output_lock
from policy_dissertation.pipelines.policy_transformation.data_io import digest, dump_json
from policy_dissertation.pipelines.training.backend import CudaBackend
from policy_dissertation.pipelines.training.config import CompressionConfig
from policy_dissertation.pipelines.training.pipeline import hashes, verify_stage
from policy_dissertation.pipelines.model_evaluation.backend import json_metadata


class Int8Config(BaseModel):
    model_config = ConfigDict(extra='forbid')
    threshold: float = Field(default=6.0, gt=0, le=100)
    minimum_free_vram_gib: float = Field(default=24, ge=1)
    minimum_free_disk_gib: float = Field(default=32, ge=1)


class Int8Backend(CudaBackend):
    def export(self, directory, source, input_directory, config):
        from transformers import AutoTokenizer, BitsAndBytesConfig
        import bitsandbytes as bnb
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=config.threshold,
                              llm_int8_enable_fp32_cpu_offload=False)
        row = json.loads((input_directory/'calibration.json').read_text())[0]
        model = self.load(source, local_files_only=True, quantization_config=q)
        try:
            count = sum(isinstance(layer, bnb.nn.Linear8bitLt) for layer in model.modules())
            if not count or not getattr(model, 'is_loaded_in_8bit', False):
                raise DissertationError('LLM.int8 quantization was not applied.')
            self.smoke(model, row)
            model.save_pretrained(directory, safe_serialization=True, max_shard_size='4GB')
            tokenizer.save_pretrained(directory)
        finally:
            del model
            self.release()
        saved = json.loads((directory/'config.json').read_text()).get('quantization_config', {})
        if not saved.get('load_in_8bit', saved.get('_load_in_8bit')):
            raise DissertationError('Saved checkpoint is missing its 8-bit configuration.')
        model = self.load(directory, local_files_only=True)
        try:
            if not getattr(model, 'is_loaded_in_8bit', False):
                raise DissertationError('Saved checkpoint did not reload in 8-bit.')
            smoke = self.smoke(model, row)
            return {'format': 'Transformers + bitsandbytes LLM.int8', 'nominal_weight_bits': 8,
                'quantized_linear_modules': count, 'reload_verified': True, 'smoke': smoke,
                'model_memory_bytes': model.get_memory_footprint(),
                'quantization_config': json_metadata(model.config.quantization_config),
                'note': 'Mixed precision/outlier handling; not every tensor or operation is int8. Quality not assessed.'}
        finally:
            del model
            self.release()


def export_8bit(compression_run, output, settings, config=None, *, resume=False, backend=None, progress=print):
    config = config or Int8Config()
    root = Path(compression_run).resolve(); output = Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise DissertationError('8-bit export must be outside the compression experiment; NF4/BF16 inputs stay preserved.')
    producer = json.loads((root/'report.json').read_text())
    if producer.get('status') not in ('COMPLETED', 'COMPLETED_WITH_WARNINGS') or not producer.get('final_model_ready') or producer.get('errors'):
        raise DissertationError('Provide a completed compression experiment with its committed BF16 recovery stage.')
    checkpoint = json.loads((root/'checkpoint.json').read_text())
    fingerprint = producer.get('fingerprint')
    import re
    if not isinstance(fingerprint, str) or not re.fullmatch('[a-f0-9]{64}', fingerprint) or checkpoint.get('fingerprint') != fingerprint:
        raise DissertationError('Compression report/checkpoint fingerprint mismatch.')
    stage = root/'_runs'/fingerprint
    for name in ('inputs', 'recovery'):
        if name not in checkpoint.get('stages', {}): raise DissertationError('Required committed compression stage is missing.')
        verify_stage(stage/name, checkpoint['stages'][name])
    source = stage/'recovery'/'merged'
    if not (source/'config.json').is_file() or not any(source.glob('*.safetensors')):
        raise DissertationError('Merged BF16 checkpoint is missing; do not use the NF4 model as the source.')
    source_config = json.loads((source/'config.json').read_text())
    if source_config.get('model_type') != 'qwen3' or source_config.get('quantization_config'):
        raise DissertationError('8-bit export source must be unquantized merged dense Qwen3.')
    identity = digest({'compression_fingerprint': fingerprint, 'source_stage': checkpoint['stages']['recovery'],
        'inputs_stage': checkpoint['stages']['inputs'], 'configuration': config.model_dump(),
        'implementation': {p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
    backend = backend or Int8Backend(settings, progress)
    final = output/'model_8bit'; pending = output/'_pending_8bit'
    with output_lock(output):
        report_path = output/'report.json'
        if report_path.exists():
            previous = json.loads(report_path.read_text())
            if not resume or previous.get('fingerprint') != identity:
                raise DissertationError('8-bit output exists or inputs/settings/code changed. Use --resume for identical export or a new directory.')
            if previous.get('final_model_ready'):
                verify_stage(final, {'files': previous['output_hashes']})
                progress('Reusing verified saved 8-bit export; no model loaded.')
                return previous
        elif final.exists() or pending.exists():
            raise DissertationError('Unrecognized existing export files; use a new output directory.')
        report = {'schema_version': 'qwen3-int8-export-v1', 'status': 'RUNNING', 'fingerprint': identity,
            'source_compression_fingerprint': fingerprint, 'source_directory': str(source),
            'configuration': config.model_dump(), 'training_started': False, 'final_model_ready': False,
            'semantic_quality_verified': False, 'errors': [], 'started_at': datetime.now(timezone.utc).isoformat()}
        dump_json(report_path, report)
        try:
            progress('[1/3] Checking Linux CUDA and export capacity; no training')
            capacity = CompressionConfig(minimum_free_vram_gib=config.minimum_free_vram_gib,
                minimum_free_disk_gib=config.minimum_free_disk_gib)
            report['runtime'] = backend.preflight(output, capacity)
            if pending.is_symlink() or final.is_symlink(): raise DissertationError('Symlinked export output refused.')
            if final.exists(): raise DissertationError('Uncommitted final export exists; preserve it and use another directory.')
            if pending.exists(): shutil.rmtree(pending)
            pending.mkdir()
            progress('[2/3] Exporting merged BF16 weights to LLM.int8 and testing saved reload')
            evidence = json_metadata(backend.export(pending, source, stage/'inputs', config))
            if evidence.get('reload_verified') is not True: raise DissertationError('8-bit reload is not verified.')
            for name in ('inputs', 'recovery'): verify_stage(stage/name, checkpoint['stages'][name])
            progress('[3/3] Recording verified 8-bit export; original NF4 remains unchanged')
            exported_hashes = hashes(pending)
            pending.rename(final)
            report.update(status='COMPLETED', final_model_ready=True, model_directory=str(final),
                evidence=evidence, output_hashes=exported_hashes,
                export_bytes=sum(p.stat().st_size for p in final.rglob('*') if p.is_file()))
            return report
        except BaseException as exc:
            report.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED', final_model_ready=False)
            report['errors'].append(str(exc) if isinstance(exc, DissertationError) else 'int8_export_failed:'+type(exc).__name__)
            raise DissertationError('8-bit export did not complete; inspect report.json. BF16/NF4 source models are preserved.') from None
        finally:
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            dump_json(report_path, report)
            backend.release()
