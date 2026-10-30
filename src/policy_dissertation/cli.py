"""Command-line entry point for the research scaffold."""
import logging
from pathlib import Path
from collections.abc import Callable

import typer
from pydantic import ValidationError

from policy_dissertation.config import Settings
from policy_dissertation.errors import DissertationError
from policy_dissertation.logging import configure_logging

app = typer.Typer(help="Corporate policy dissertation research framework.",
                  no_args_is_help=False, invoke_without_command=True, pretty_exceptions_enable=False)


def run_safely(action: Callable[[], None]) -> None:
    """Map failures to stable exit codes without leaking configuration values."""
    try:
        action()
    except ValidationError:
        typer.echo("Error: invalid configuration. Check environment settings.", err=True)
        raise typer.Exit(code=2) from None
    except DissertationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except Exception:
        logging.getLogger("policy_dissertation").error("Unexpected application failure.")
        raise typer.Exit(code=1) from None


@app.callback()
def initialize(ctx: typer.Context) -> None:
    """Research commands. Run without arguments for the interactive workspace."""
    if ctx.invoked_subcommand is None:
        from policy_dissertation.interface.dashboard import run_dashboard
        run_safely(run_dashboard)


@app.command('menu')
def interactive_menu() -> None:
    """Open the interactive data-preparation workspace."""
    from policy_dissertation.interface.dashboard import run_dashboard
    run_safely(run_dashboard)


@app.command()
def status() -> None:
    """Validate configuration and report the implementation stage."""
    def execute() -> None:
        settings = Settings()
        configure_logging(settings.log_level)
        typer.echo("Project initialized. Document ingestion is available; Enterprise data is available via enterprise_data (legacy pip1_data/pip2_code retained), Pipeline 3 via pip3_datasets, and enterprise preparation is automatic; pip4_prepare is retained as a legacy command. Experimental Linux/CUDA pruning, LoRA recovery and NF4 export are available via compress_model; actual GPU execution and final evaluation remain separate.")
    run_safely(execute)


policies = typer.Typer(help="Corporate policy ingestion and structured preparation.", no_args_is_help=True)
app.add_typer(policies, name="policies")


@policies.command("parse")
def parse_policies(
    input_directory: Path | None = typer.Option(None, "--input", help="Source directory (default: POLICY_DIR or dumy_policies)."),
    output_directory: Path | None = typer.Option(None, "--output", help="Artifact directory (default: PARSED_OUTPUT_DIR or artifacts/pipeline1/parsed)."),
) -> None:
    """Parse PDF, Markdown, and TXT sources into traceable JSON artifacts."""
    def execute() -> None:
        from policy_dissertation.pipelines.policy_transformation.ingestion import ingest_documents

        settings = Settings()
        configure_logging(settings.log_level)
        summary = ingest_documents(
            input_directory if input_directory is not None else settings.policy_dir,
            output_directory if output_directory is not None else settings.parsed_output_dir,
        )
        typer.echo(f"Documents discovered: {summary.discovered}\n"
                   f"Successfully parsed: {summary.parsed}\n"
                   f"Failed: {summary.failed}\nUnsupported: {summary.unsupported}\n"
                   f"Output artifacts: {summary.output_directory}")
        for warning in summary.discovery_warnings:
            typer.echo(f"Warning: {warning}", err=True)
        if summary.failed or summary.discovery_warnings:
            raise DissertationError("Ingestion completed with failures; inspect summary.json and document artifacts.")
    run_safely(execute)


@policies.command("chunk")
def chunk_policies(
    input_directory: Path | None = typer.Option(None, "--input", help="Parsed artifact directory (default: PARSED_OUTPUT_DIR)."),
    output_directory: Path | None = typer.Option(None, "--output", help="Chunk artifact directory (default: CHUNK_OUTPUT_DIR)."),
    max_tokens: int = typer.Option(2048, min=1, help="UTF-8 byte-token upper bound per chunk; indivisible oversized clauses are flagged."),
) -> None:
    """Assess, conservatively normalize, and chunk parsed policy artifacts."""
    def execute() -> None:
        from policy_dissertation.pipelines.policy_transformation.chunk_batch import chunk_documents

        settings = Settings()
        configure_logging(settings.log_level)
        summary = chunk_documents(input_directory or settings.parsed_output_dir,
                                  output_directory or settings.chunk_output_dir, max_tokens)
        typer.echo(f"Documents processed: {summary.documents_processed}/{summary.documents_seen}\n"
                   f"Skipped: {summary.documents_skipped}; failed: {summary.documents_failed}\n"
                   f"Chunks: {summary.chunk_count}; oversized: {summary.oversized_chunks}\n"
                   f"Quality issues: {summary.quality_issue_count}; corrected: {summary.corrected_issue_count}; "
                   f"unresolved: {summary.unresolved_issue_count}\n"
                   f"Output artifacts: {summary.output_directory}")
        if summary.documents_failed:
            raise DissertationError("Chunking completed with document failures; inspect summary.json.")
    run_safely(execute)


@app.command('enterprise_data')
def enterprise_data(
    documents: Path | None = typer.Option(None, exists=True, file_okay=False, help='Corporate document folder; optional.'),
    code: Path | None = typer.Option(None, exists=True, file_okay=False, help='Source repository folder; optional.'),
    documents_dataset: Path | None = typer.Option(None, exists=True, help='Reuse an existing accepted document JSONL or output directory.'),
    code_dataset: Path | None = typer.Option(None, exists=True, help='Reuse an existing accepted coding JSONL or output directory without generation.'),
    output: Path = typer.Option(Path('artifacts/enterprise'), help='New combined dataset output directory.'),
    max_tokens: int = typer.Option(2048, min=1, help='Source document chunk / code-unit limit, not an LLM response limit.'),
    prepare: bool = typer.Option(True, '--prepare/--skip-preparation', help='Automatically validate exact Qwen tokens, deduplicate and export leakage-safe splits.'),
    training_max_tokens: int = typer.Option(4096, min=1, help='Exact final Qwen conversation limit including template overhead.'),
    seed: int = typer.Option(42, help='Deterministic enterprise split seed.'),
    limit_documents: int | None = typer.Option(None, min=1, help='Optional document test limit.'),
    limit_units: int | None = typer.Option(None, min=1, help='Optional repository-unit test limit.'),
    task_types: str = typer.Option('implementation,explanation,debugging,unit_tests,refactoring,architecture,secure_coding'),
    allow_incomplete_import: bool = typer.Option(False, help='Explicitly reuse validated accepted subsets from failed/incomplete producer runs.'),
    overwrite: bool = typer.Option(False, help='Explicitly resume/replace enterprise exports; original inputs remain protected.'),
    push_to_hub: str | None = typer.Option(None, help='Optionally publish one private OWNER/NAME enterprise dataset.'),
    allow_incomplete_run: bool = typer.Option(False, help='Explicitly publish an accepted subset despite incomplete source runs.'),
) -> None:
    """Generate one enterprise dataset from documents, code, or both; reuse prior exports optionally."""
    def execute():
        from policy_dissertation.pipelines.enterprise.pipeline import run_enterprise
        from policy_dissertation.data.hub import publish
        settings = Settings()
        configure_logging(settings.log_level)
        typer.echo('Enterprise — Policy and Code Training Data')
        result = run_enterprise(output, settings, documents=documents, code=code,
            documents_dataset=documents_dataset, code_dataset=code_dataset,
            allow_incomplete_import=allow_incomplete_import, max_tokens=max_tokens,
            limit_documents=limit_documents, limit_units=limit_units,
            task_types=tuple(t.strip() for t in task_types.split(',') if t.strip()),
            overwrite=overwrite, progress=lambda text: typer.echo(text, err=True))
        if prepare and result['status'] != 'FAILED':
            from policy_dissertation.pipelines.enterprise.preparation import prepare_enterprise
            result=prepare_enterprise(output,output,settings,maximum=training_max_tokens,seed=seed,overwrite=overwrite,
                in_place=True,research_report=Path('report.md'),progress=lambda text:typer.echo(text,err=True))
        typer.echo(f"\nAccepted enterprise examples: {result['accepted_examples']}\n"
                   f"Duplicates removed: {result['duplicates_removed']}\n"
                   f"Invalid imported/exported records: {result['invalid_records']}\n"
                   f"Source runs complete: {result['source_run_complete']}\n"
                   f"Training data: {result['output_directory']}/training.jsonl\nStatus: {result['status']}")
        if prepare:
            typer.echo(f"Prepared splits: {result.get('splits', {})}; enterprise training ready: {result.get('training_ready', False)}")
            typer.echo(f"Preparation rejected: {result.get('preparation',{}).get('rejected_examples',0)}; preparation duplicates: {result.get('preparation',{}).get('duplicates_removed',0)}")
        if result['errors']:typer.echo('Errors: '+'; '.join(result['errors']), err=True)
        if result['warnings']:typer.echo('Warnings: '+'; '.join(result['warnings']), err=True)
        if result['status']=='FAILED' and not (push_to_hub and allow_incomplete_run and result['accepted_examples']):
            raise DissertationError('Enterprise workflow failed; inspect report.json. Existing input datasets are preserved.')
        if push_to_hub:
            publication=publish(output,push_to_hub,settings,allow_incomplete_run=allow_incomplete_run)
            typer.echo(f"Published privately: {publication['repo_id']} @ {publication['revision']}")
        if result['status']=='FAILED':
            raise DissertationError('Accepted subset published; enterprise processing remains failed/incomplete.')
    run_safely(execute)


@app.command('enterprise_prepare')
def enterprise_prepare(
    input_directory: Path = typer.Option(..., '--input', exists=True, file_okay=False),
    output: Path = typer.Option(Path('artifacts/enterprise_ready')),
    max_tokens: int = typer.Option(4096, min=1),
    seed: int = typer.Option(42),
    overwrite: bool = typer.Option(False),
) -> None:
    """Prepare existing enterprise records without policy/code regeneration."""
    def execute():
        from policy_dissertation.pipelines.enterprise.preparation import prepare_enterprise
        result=prepare_enterprise(input_directory,output,Settings(),maximum=max_tokens,seed=seed,overwrite=overwrite,research_report=Path('report.md'),
            progress=lambda text:typer.echo(text,err=True))
        typer.echo(f"Status: {result['status']}; enterprise training ready: {result['training_ready']}; splits: {result.get('splits',{})}")
        if result['status']=='FAILED':raise DissertationError('Enterprise preparation failed; inspect report.json.')
    run_safely(execute)


@app.command("pip1_data")
def pip1_data(
    input_directory: Path = typer.Option(Path("dumy_policies"), "--input"),
    output_directory: Path = typer.Option(Path("artifacts/pipeline1"), "--output"),
    max_tokens: int = typer.Option(2048, min=1, help="Maximum source chunk size in UTF-8 byte-token units."),
    push_to_hub: str | None = typer.Option(None, help="Optionally publish accepted records to a private OWNER/NAME dataset."),
    overwrite: bool = typer.Option(False, help="Explicitly allow replacing exports, including checkpoint resume."),
    limit_documents: int | None = typer.Option(None, min=1, help="Process at most N supported source documents for testing; omit to process all."),
) -> None:
    """Compatibility command: use enterprise_data --documents for new enterprise datasets."""
    def execute() -> None:
        from policy_dissertation.pipelines.policy_transformation.pipeline import run_pipeline

        settings = Settings()
        configure_logging(settings.log_level)
        typer.echo("Pipeline 1 — Corporate Policy Transformation")
        from policy_dissertation.data.hub import protect_exports, publish
        protect_exports(output_directory, overwrite)
        result = run_pipeline(input_directory, output_directory, settings, max_tokens,
                              progress=lambda text: typer.echo(text, err=True), limit_documents=limit_documents, overwrite=overwrite)
        typer.echo(f"\nDocuments processed: {result['documents_processed']}\n"
                   f"Documents excluded by limit: {result['documents_excluded_by_limit']}\n"
                   f"Chunks generated: {result['chunks_generated']}\n"
                   f"Training examples generated: {result['examples_generated']}\n"
                   f"Accepted examples: {result['accepted_examples']}\n"
                   f"Rejected examples: {result['rejected_examples']}\n\n"
                   f"Training data: {result['output_directory']}/training.jsonl\n"
                   f"Report: {result['output_directory']}/report.json\n\n"
                   f"Status: {result['status']}")
        if result['status'] == 'FAILED':
            raise DissertationError('Pipeline 1 failed mandatory processing or validation; inspect report.json and rejected.jsonl.')
        if push_to_hub:
            publication = publish(output_directory, push_to_hub, settings)
            typer.echo(f"Published privately: {publication['repo_id']} @ {publication['revision']}")
    run_safely(execute)


@app.command("pip2_code")
def pip2_code(
    input_directory: Path = typer.Option(Path("code_repos"), "--input"),
    output_directory: Path = typer.Option(Path("artifacts/pipeline2"), "--output"),
    max_tokens: int = typer.Option(2048, min=1, help="Maximum source-unit UTF-8 byte-token count; no source truncation."),
    push_to_hub: str | None = typer.Option(None, help="Optionally publish accepted records to a private OWNER/NAME dataset."),
    overwrite: bool = typer.Option(False, help="Explicitly allow replacing exports, including checkpoint resume."),
    limit_units: int | None = typer.Option(None, min=1, help="Optional total source-unit cap for a test run, balanced across repositories."),
    task_types: str = typer.Option("implementation,explanation,debugging,unit_tests,refactoring,architecture,secure_coding",
                                   help="Comma-separated task types, rotated across source units."),
) -> None:
    """Compatibility command: use enterprise_data --code for new enterprise datasets."""
    def execute() -> None:
        from policy_dissertation.pipelines.repository_data.pipeline import run_pipeline2

        settings = Settings()
        configure_logging(settings.log_level)
        typer.echo("Pipeline 2 — Repository-to-Training-Data")
        from policy_dissertation.data.hub import protect_exports, publish
        protect_exports(output_directory, overwrite)
        result = run_pipeline2(input_directory, output_directory, settings, max_tokens,
                               overwrite=overwrite, limit_units=limit_units, task_types=tuple(t.strip() for t in task_types.split(',') if t.strip()),
                               progress=lambda text: typer.echo(text, err=True))
        typer.echo(f"\nRepositories indexed: {result['repositories_discovered']}\n"
                   f"Source units selected: {result['units_selected']}\n"
                   f"Failed units: {result['failed_units']}; pending units: {result['pending_units']}\n"
                   f"Training examples generated: {result['examples_generated']}\n"
                   f"Accepted examples: {result['accepted_examples']}\n"
                   f"Runtime-verified accepted: {result['accepted_verified']}\n"
                   f"Unverified knowledge examples: {result['accepted_unverified_knowledge']}\n"
                   f"Rejected examples: {result['rejected_examples']}\n"
                   f"Quarantined examples: {result['quarantined_examples']}\n\n"
                   f"Training data: {result['output_directory']}/training.jsonl\n"
                   f"Report: {result['output_directory']}/report.json\n"
                   f"Status: {result['status']}")
        if result['status'] == 'FAILED':
            reasons = ', '.join(f'{reason}: {count}' for reason, count in result['rejection_reasons'].items())
            if reasons:
                typer.echo('Failure/rejection reasons: ' + reasons)
            raise DissertationError('Pipeline 2 failed; inspect report.json and rejected.jsonl.')
        if push_to_hub:
            publication = publish(output_directory, push_to_hub, settings)
            typer.echo(f"Published privately: {publication['repo_id']} @ {publication['revision']}")
    run_safely(execute)


@app.command("pip3_datasets")
def pip3_datasets(
    output_directory: Path = typer.Option(Path("artifacts/pipeline3"), "--output"),
    overwrite: bool = typer.Option(False, help="Explicitly allow replacing exports, including checkpoint resume."),
    max_samples: int = typer.Option(10000, min=1, help="Total accepted-example cap across selected datasets."),
    dataset: list[str] | None = typer.Option(None, "--dataset", help="Repeat nemotron/opencode or their full Hub IDs."),
    seed: int = typer.Option(42, help="Seed for bounded prefix-buffer sampling."),
    allocation: list[str] | None = typer.Option(None, "--allocation", help="Repeat ALIAS=accepted_count; unspecified datasets share the remainder."),
    config: list[str] | None = typer.Option(None, "--config", help="Repeat ALIAS=config; nemotron=all selects all inspected configurations."),
    split: list[str] | None = typer.Option(None, "--split", help="Repeat ALIAS=split; defaults to inspected train split."),
    revision: list[str] | None = typer.Option(None, "--revision", help="Repeat ALIAS=Hub_revision; resolved SHA is pinned in checkpoints."),
    filter_config: Path | None = typer.Option(None, "--filter-config", exists=True, dir_okay=False, help="JSON overrides for Pipeline3Settings."),
    refresh_metadata: bool = typer.Option(False, help="Resolve current Hub revisions instead of reusing pinned metadata."),
) -> None:
    """Stream and prepare NVIDIA datasets without LLM calls or model downloads."""
    def execute() -> None:
        import json
        from policy_dissertation.config import Pipeline3Settings
        from policy_dissertation.pipelines.external_datasets.pipeline import run_pipeline3
        from policy_dissertation.pipelines.external_datasets.sources import DATASETS

        def alias(value):
            return next((a for a, name in DATASETS.items() if value in (a, name)), value)

        def assignments(values, multiple=False, numeric=False):
            result = {}
            for item in values or []:
                if '=' not in item:
                    raise DissertationError('Options require ALIAS=value.')
                key, value = item.split('=', 1)
                key = alias(key)
                if not value or (key in result and not multiple):
                    raise DissertationError('Empty or repeated option assignment.')
                if numeric:
                    try:
                        value = int(value)
                    except ValueError:
                        raise DissertationError('Allocations must be integer counts.') from None
                if multiple:
                    result.setdefault(key, []).append(value)
                else:
                    result[key] = value
            return result
        settings = Settings()
        configure_logging(settings.log_level)
        if filter_config:
            try:
                overrides = json.loads(filter_config.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                raise DissertationError('Filter configuration must be readable JSON.') from None
            if not isinstance(overrides, dict):
                raise DissertationError('Filter configuration must be a JSON object.')
            settings.pipeline3 = Pipeline3Settings.model_validate(settings.pipeline3.model_dump() | overrides)
        typer.echo('Pipeline 3 — External Dataset Ingestion & Preparation')
        from policy_dissertation.data.hub import protect_exports
        protect_exports(output_directory, overwrite)
        result = run_pipeline3(output_directory, settings, max_samples, overwrite=overwrite,
            selected=tuple(alias(a) for a in dataset) if dataset else ('nemotron', 'opencode'), seed=seed,
            allocation=assignments(allocation, numeric=True), configs=assignments(config, multiple=True),
            splits=assignments(split), revisions=assignments(revision), refresh=refresh_metadata,
            progress=lambda text: typer.echo(text, err=True))
        typer.echo(f"\nRecords processed: {result['records_processed']}\n"
                   f"Accepted examples: {result['accepted_examples']}/{result['max_samples']}\n"
                   f"Rejected examples: {result['rejected_examples']}\n"
                   f"Decoded rows read this run (includes prefetch/replay): {result['records_read_this_run']}\n"
                   f"Resumed decisions: {result['resumed_decisions']}\n"
                   f"Training data: {result['output_directory']}/training.jsonl\n"
                   f"Report: {result['output_directory']}/report.json\n"
                   f"Status: {result['status']}")
        if result['status'] == 'FAILED':
            typer.echo('Failure categories: ' + '; '.join(result['errors']), err=True)
            raise DissertationError('Pipeline 3 failed; inspect report.json.')
    run_safely(execute)


@app.command("pip4_prepare", hidden=True)
def pip4_prepare(
    input_directory: Path = typer.Option(Path('artifacts'), '--input'),
    output_directory: Path | None = typer.Option(None, '--output'),
    max_tokens: int | None = typer.Option(None, min=1, help='Exact maximum token count including Qwen chat-template/tool overhead.'),
    seed: int | None = typer.Option(None),
    overwrite: bool = typer.Option(False, help='Explicitly allow replacing existing exports or resuming.'),
    max_examples: int | None = typer.Option(None, min=1, help='Optional cap on the without-replacement mixed cohort.'),
    mix_ratio: list[str] | None = typer.Option(None, '--mix-ratio', help='Repeat pipeline1=0.4, pipeline2=0.3, pipeline3=0.3 assignments.'),
    split_ratio: list[str] | None = typer.Option(None, '--split-ratio', help='Repeat train=0.9, validation=0.05, test=0.05 assignments.'),
    config: Path | None = typer.Option(None, '--config', exists=True, dir_okay=False, help='Dataset-source YAML or legacy JSON Pipeline4Settings overrides.'),
    tokenizer_revision: str | None = typer.Option(None, help='Qwen3-14B tokenizer revision; resolved commit and asset hashes are recorded.'),
    local_files_only: bool = typer.Option(False, help='Require previously downloaded, checksummed tokenizer assets.'),
    allow_partial_sources: bool = typer.Option(False, help='Diagnostic only: permit missing/failed upstream sources; never claim full training readiness.'),
) -> None:
    """Unify, validate, balance, split, and prepare Qwen3 datasets without training."""
    def execute():
        import json
        from policy_dissertation.config import Pipeline4Settings
        from policy_dissertation.pipelines.dataset_preparation.pipeline import run_pipeline4
        settings=Settings()
        configure_logging(settings.log_level)
        from policy_dissertation.data.hub import protect_exports
        sources = None
        chosen_output = output_directory or Path('artifacts/pipeline4')
        chosen_tokens = max_tokens if max_tokens is not None else 4096
        chosen_seed = seed if seed is not None else 42
        values=settings.pipeline4.model_dump()
        if config:
            if config.suffix.lower() in ('.yaml', '.yml'):
                from policy_dissertation.data.sources import read_config
                parsed = read_config(config)
                sources = parsed.sources
                chosen_output = output_directory or Path(parsed.output.directory)
                chosen_tokens = max_tokens if max_tokens is not None else parsed.output.max_tokens
                chosen_seed = seed if seed is not None else parsed.split.seed
                values['mix_ratios'] = {s.name: s.weight for s in sources}
                values['split_ratios'] = parsed.split.model_dump(exclude={'seed'})
            else:
                try: overrides=json.loads(config.read_text(encoding='utf-8'))
                except (OSError,ValueError):raise DissertationError('Preparation configuration must be readable JSON.') from None
                if not isinstance(overrides,dict):raise DissertationError('Preparation configuration must be an object.')
                values.update(overrides)
        def assigned(items, key):
            result=dict(values[key])
            seen=set()
            for item in items or []:
                if '=' not in item:raise DissertationError('Ratios require NAME=value.')
                name,value=item.split('=',1)
                if name not in result or name in seen:raise DissertationError('Unknown or repeated ratio name.')
                try:result[name]=float(value)
                except ValueError:raise DissertationError('Ratio values must be finite numbers.') from None
                seen.add(name)
            return result
        values['mix_ratios']=assigned(mix_ratio,'mix_ratios')
        values['split_ratios']=assigned(split_ratio,'split_ratios')
        if tokenizer_revision:values['tokenizer_revision']=tokenizer_revision
        if local_files_only:values['local_files_only']=True
        settings.pipeline4=Pipeline4Settings.model_validate(values)
        typer.echo('Pipeline 4 — Unified Dataset Validation & Training Preparation')
        protect_exports(chosen_output, overwrite)
        result=run_pipeline4(input_directory,chosen_output,settings,chosen_tokens,seed=chosen_seed,max_examples=max_examples,
            overwrite=overwrite,sources=sources,allow_partial_sources=allow_partial_sources,progress=lambda text:typer.echo(text,err=True))
        typer.echo(f"\nInput records: {result['input_records']}\n"
                   f"Validated examples: {result['validated_examples']}\n"
                   f"Rejected invalid examples: {result['rejected_examples']}\n"
                   f"Duplicates removed: {result['duplicates_removed']}\n"
                   f"Excluded by mixing: {result['not_selected_examples']}\n"
                   f"Exported examples: {result['accepted_examples']}\n"
                   f"Splits: {result['splits']}\n"
                   f"Training ready: {result['training_ready']}\n"
                   f"Output: {result['output_directory']}\nStatus: {result['status']}")
        if result['input_issues']:typer.echo('Input issues: '+'; '.join(result['input_issues']),err=True)
        if result['status']=='FAILED':
            typer.echo('Failure categories: '+'; '.join(result['errors']),err=True)
            raise DissertationError('Pipeline 4 preparation failed; inspect report.json. No training was started.')
    run_safely(execute)


@app.command('nvidia_data')
def nvidia_data(
    enterprise: Path = typer.Option(Path('artifacts/enterprise_ready'), exists=True, file_okay=False),
    output: Path = typer.Option(Path('artifacts/nvidia')),
    max_samples: int = typer.Option(10000, min=1, help='Total accepted native examples across selected datasets.'),
    max_tokens: int = typer.Option(4096, min=1, help='Exact Qwen conversation bound including original template.'),
    dataset: list[str] | None = typer.Option(None, '--dataset', help='Repeat nemotron/opencode; defaults to both.'),
    allocation: list[str] | None = typer.Option(None, '--allocation', help='Optional accepted allocation ALIAS=count.'),
    revision: list[str] | None = typer.Option(None, '--revision', help='Optional ALIAS=Hub_revision; first resolution is pinned/cached.'),
    nemotron_config: str = typer.Option('chat', help='Inspected Nemotron configuration.'),
    opencode_config: str = typer.Option('train', help='Inspected OpenCode configuration.'),
    seed: int = typer.Option(42),
    shuffle_buffer: int | None = typer.Option(None, min=1, max=128, help='Seeded bounded prefix buffer; not globally uniform sampling.'),
    max_records_per_dataset: int | None = typer.Option(None, min=1, help='Optional unique-candidate cap for a small local test.'),
) -> None:
    """Prepare a resumable native NVIDIA source plan; no full dataset rewrite or training."""
    def execute():
        from policy_dissertation.pipelines.native_datasets.pipeline import run_native_datasets
        settings=Settings()
        configure_logging(settings.log_level)
        if shuffle_buffer is not None:settings.pipeline3.shuffle_buffer=shuffle_buffer
        selected=tuple(dataset or ('nemotron','opencode'))
        def mappings(items,numeric=False):
            result={}
            for item in items or []:
                if '=' not in item:raise DissertationError('Native options require ALIAS=value.')
                alias,value=item.split('=',1)
                if alias not in selected or alias in result or not value:raise DissertationError('Unknown/repeated native alias or empty value.')
                try:result[alias]=int(value) if numeric else value
                except ValueError:raise DissertationError('Native allocations must be integers.') from None
            return result
        configs={a: nemotron_config if a=='nemotron' else opencode_config for a in selected}
        typer.echo('External Data — NVIDIA Native Preparation')
        result=run_native_datasets(enterprise,output,settings,max_samples=max_samples,max_tokens=max_tokens,
            selected=selected,allocation=mappings(allocation,True),revisions=mappings(revision),configs=configs,
            seed=seed,max_records_per_dataset=max_records_per_dataset,progress=lambda t:typer.echo(t,err=True),
            research_report=Path('report.md'))
        typer.echo(f"\nRecords processed: {result['records_processed']}\nAccepted: {result['accepted_examples']}/{max_samples}\n"
                   f"Rejected: {result['rejected_examples']}\nRows read by sampler this run (includes buffer/replay): {result['decoded_rows_read_this_run']}\n"
                   f"Resumed decisions: {result['resumed_decisions']}\nSource plan: {output.resolve()}/source_plan.json\n"
                   f"Status: {result['status']}; data prepared: {result['data_prepared']}")
        if result['errors']:typer.echo('Errors: '+'; '.join(result['errors']),err=True)
        if result['status']=='FAILED':raise DissertationError('Native NVIDIA preparation failed; inspect report.json. No training started.')
    run_safely(execute)


@app.command('nvidia_validate')
def nvidia_validate(
    plan: Path = typer.Option(..., exists=True, file_okay=False, help='Existing native preparation directory.'),
    enterprise: Path = typer.Option(Path('artifacts/enterprise_ready'), exists=True, file_okay=False),
    output: Path = typer.Option(Path('artifacts/nvidia_validation')),
    limit_records: int | None = typer.Option(None, min=1, help='Optional approved-record sample limit; not a full validation if smaller than the plan.'),
    overwrite: bool = typer.Option(False),
) -> None:
    """Validate a prepared native selection read-only; never generate, install or train."""
    def execute():
        from policy_dissertation.interface.validation import validate_native_plan
        result=validate_native_plan(plan,enterprise,output,Settings(),limit_records=limit_records,overwrite=overwrite,
            progress=lambda text:typer.echo(text,err=True))
        typer.echo(f"Status: {result['status']}; records checked: {result['records_checked']}; full approved selection validated: {result['full_plan_validated']}")
        if result['status']=='FAILED':raise DissertationError('Native validation failed; inspect the validation report. Source data is unchanged.')
    run_safely(execute)


@app.command('check_training_sources')
def check_training_sources(
    enterprise: Path = typer.Option(..., exists=True, file_okay=False, help='Prepared enterprise output with train/validation/test files.'),
    output: Path = typer.Option(Path('artifacts/training_input_check')),
    dataset: list[str] | None = typer.Option(None, '--dataset', help='Repeat nemotron/opencode; defaults to both.'),
    max_external_records: int = typer.Option(10, min=1, help='Bounded decoded rows per external dataset; this is a preflight, not a full audit.'),
    max_tokens: int = typer.Option(4096, min=1),
) -> None:
    """Check direct native NVIDIA inputs against enterprise splits; do not train or export NVIDIA rows."""
    def execute():
        from policy_dissertation.data.training_inputs import TrainingInputGate, index_enterprise, iter_nvidia, pinned_enterprise_settings
        from policy_dissertation.pipelines.dataset_preparation.tokenization import load_tokenizer
        from policy_dissertation.pipelines.policy_transformation.data_io import dump_json
        from policy_dissertation.data.hub import protect_exports
        settings=Settings()
        settings=pinned_enterprise_settings(enterprise,settings)
        selected=dataset or ['nemotron','opencode']
        if any(d not in ('nemotron','opencode') for d in selected) or len(set(selected))!=len(selected):
            raise DissertationError('Select unique nemotron/opencode datasets.')
        protect_exports(output)
        output.mkdir(parents=True,exist_ok=True)
        tokenizer=load_tokenizer(output,settings.pipeline4,settings.hf_token.get_secret_value() if settings.hf_token else None,lambda t:typer.echo(t,err=True))
        gate=TrainingInputGate(output/'fingerprints.sqlite3',settings.pipeline4.near_threshold)
        result={'scope':'bounded preflight; not full-corpus readiness or training','status':'RUNNING','max_records_per_source':max_external_records,'llm_calls':0,'model_weights_downloaded':False}
        try:
            index_enterprise(enterprise,settings,tokenizer,max_tokens,gate)
            gate.seal_holdouts()
            for alias in selected:
                typer.echo('Checking native '+alias+' stream')
                for _ in iter_nvidia(alias,settings,tokenizer,max_tokens,gate,max_records=max_external_records,
                                      progress=lambda t:typer.echo(t,err=True)):pass
            result.update(status='COMPLETED_WITH_WARNINGS' if any(k.startswith('rejected:') for k in gate.counts) else 'COMPLETED')
        except Exception as exc:
            result.update(status='FAILED',error=str(exc) if isinstance(exc,DissertationError) else 'input_access_or_validation_failed:'+type(exc).__name__)
            raise DissertationError('Training-input preflight failed; inspect its report.json.') from None
        finally:
            result.update(counts=dict(gate.counts),sources=gate.sources,tokenizer=tokenizer.info,training_started=False)
            dump_json(output/'report.json',result)
            gate.close()
        typer.echo(f"Status: {result['status']}; counts: {result['counts']}. No training started.")
    run_safely(execute)


@app.command('compress_model')
def compress_model(
    enterprise: Path = typer.Option(..., exists=True, file_okay=False, help='Prepared enterprise splits.'),
    native_plan: Path = typer.Option(..., exists=True, file_okay=False, help='Prepared NVIDIA native selection directory.'),
    output: Path = typer.Option(Path('artifacts/compression')),
    config_file: Path | None = typer.Option(None, '--config', exists=True, dir_okay=False, help='Optional strict CompressionConfig JSON.'),
    prune_ratio: float | None = typer.Option(None, min=.001, max=.2, help='Fraction of MLP channels, NOT total model parameters.'),
    max_tokens: int | None = typer.Option(None, min=32, max=32768),
    max_steps: int | None = typer.Option(None, min=1),
    calibration_samples: int | None = typer.Option(None, min=1, max=2048),
    allow_incomplete_sources: bool = typer.Option(False, help='Explicitly use the accepted subset of incomplete enterprise producers.'),
    resume: bool = typer.Option(False, help='Resume only identical input/config/code/runtime checkpoints.'),
    check_only: bool = typer.Option(False, help='Linux/CUDA/dependency/capacity checks only; no model weights or training.'),
) -> None:
    """Linux/CUDA only: structured MLP pruning → BF16 LoRA recovery → saved NF4 4-bit export."""
    def execute():
        from policy_dissertation.pipelines.training.config import CompressionConfig
        from policy_dissertation.pipelines.training.pipeline import run_compression
        settings=Settings()
        config=CompressionConfig.model_validate_json(config_file.read_text()) if config_file else settings.training
        updates={k:v for k,v in {'prune_ratio':prune_ratio,'max_tokens':max_tokens,'max_steps':max_steps,
                                 'calibration_samples':calibration_samples}.items() if v is not None}
        if allow_incomplete_sources:updates['allow_incomplete_sources']=True
        config=CompressionConfig.model_validate(config.model_dump() | updates)
        typer.echo('Model Compression — Qwen3-14B / Linux NVIDIA CUDA')
        result=run_compression(enterprise,native_plan,output,settings,config,resume=resume,check_only=check_only,
                              progress=lambda text:typer.echo(text,err=True))
        typer.echo(f"Status: {result['status']}; final model ready: {result['final_model_ready']}")
        if result.get('model_directory'):typer.echo(f"4-bit model: {result['model_directory']}")
        typer.echo(f"Report: {output.resolve()/'report.json'}")
        typer.echo('Final quality/security evaluation remains a separate next stage.')
    run_safely(execute)


@app.command('download_dataset')
def download_dataset_command(
    repo_id: str = typer.Option(..., '--repo-id', help='Published enterprise dataset OWNER/NAME.'),
    output: Path = typer.Option(Path('artifacts/enterprise_ready'), '--output'),
    revision: str = typer.Option('main', help='Branch, tag or immutable commit; resolved and recorded before downloading.'),
    overwrite: bool = typer.Option(False, help='Explicitly replace existing exports after validating the download.'),
) -> None:
    """Restore published enterprise data without extraction, generation or LLM calls."""
    def execute():
        from policy_dissertation.interface.restoration import download_dataset
        result = download_dataset(repo_id, output, Settings(), revision=revision, overwrite=overwrite)
        typer.echo(f"Downloaded: {repo_id} @ {result['restoration']['revision']}")
        typer.echo(f"Accepted examples: {result['accepted_examples']}; training ready: {result['training_ready']}")
        typer.echo(f"Output: {output.resolve()}")
        typer.echo(f"Status: {result['status']}")
    run_safely(execute)


@app.command('publish_dataset')
def publish_dataset(
    input_directory: Path = typer.Option(..., '--input', exists=True, file_okay=False),
    repo_id: str = typer.Option(..., '--repo-id', help='Private dataset repository OWNER/NAME.'),
    allow_incomplete_run: bool = typer.Option(False, help='Publish validated accepted records despite an incomplete/failed producer run; preserve that status.'),
) -> None:
    """Publish an existing accepted dataset without generation or LLM calls."""
    def execute():
        from policy_dissertation.data.hub import publish
        result = publish(input_directory, repo_id, Settings(), allow_incomplete_run=allow_incomplete_run)
        typer.echo(f"Published privately: {result['repo_id']} @ {result['revision']}")
        if result['accepted_subset_only']:
            typer.echo('Accepted subset published; the source run remains failed/incomplete.')
    run_safely(execute)


def main() -> None:
    app()
