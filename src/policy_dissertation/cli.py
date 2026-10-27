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
                  no_args_is_help=True, pretty_exceptions_enable=False)


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
def initialize() -> None:
    """Research commands."""


@app.command()
def status() -> None:
    """Validate configuration and report the implementation stage."""
    def execute() -> None:
        settings = Settings()
        configure_logging(settings.log_level)
        typer.echo("Project initialized. Document ingestion is available; Pipeline 1 is available via pip1_data, Pipeline 2 via pip2_code, Pipeline 3 via pip3_datasets, and Pipeline 4 via pip4_prepare. Model fine-tuning is not implemented.")
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


@app.command("pip1_data")
def pip1_data(
    input_directory: Path = typer.Option(Path("dumy_policies"), "--input"),
    output_directory: Path = typer.Option(Path("artifacts/pipeline1"), "--output"),
    max_tokens: int = typer.Option(2048, min=1, help="Maximum source chunk size in UTF-8 byte-token units."),
    limit_documents: int | None = typer.Option(None, min=1, help="Process at most N supported source documents for testing; omit to process all."),
) -> None:
    """Run Pipeline 1 from source policies to validated conversational records."""
    def execute() -> None:
        from policy_dissertation.pipelines.policy_transformation.pipeline import run_pipeline

        settings = Settings()
        configure_logging(settings.log_level)
        typer.echo("Pipeline 1 — Corporate Policy Transformation")
        result = run_pipeline(input_directory, output_directory, settings, max_tokens,
                              progress=lambda text: typer.echo(text, err=True), limit_documents=limit_documents)
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
    run_safely(execute)


@app.command("pip2_code")
def pip2_code(
    input_directory: Path = typer.Option(Path("code_repos"), "--input"),
    output_directory: Path = typer.Option(Path("artifacts/pipeline2"), "--output"),
    max_tokens: int = typer.Option(2048, min=1, help="Maximum source-unit UTF-8 byte-token count; no source truncation."),
    limit_units: int | None = typer.Option(None, min=1, help="Optional total source-unit cap for a test run, balanced across repositories."),
    task_types: str = typer.Option("implementation,explanation,debugging,unit_tests,refactoring,architecture,secure_coding",
                                   help="Comma-separated task types, rotated across source units."),
) -> None:
    """Generate repository-grounded data; execute code only in an available restricted container."""
    def execute() -> None:
        from policy_dissertation.pipelines.repository_data.pipeline import run_pipeline2

        settings = Settings()
        configure_logging(settings.log_level)
        typer.echo("Pipeline 2 — Repository-to-Training-Data")
        result = run_pipeline2(input_directory, output_directory, settings, max_tokens,
                               limit_units=limit_units, task_types=tuple(t.strip() for t in task_types.split(',') if t.strip()),
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
    run_safely(execute)


@app.command("pip3_datasets")
def pip3_datasets(
    output_directory: Path = typer.Option(Path("artifacts/pipeline3"), "--output"),
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
        result = run_pipeline3(output_directory, settings, max_samples,
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


@app.command("pip4_prepare")
def pip4_prepare(
    input_directory: Path = typer.Option(Path('artifacts'), '--input'),
    output_directory: Path = typer.Option(Path('artifacts/pipeline4'), '--output'),
    max_tokens: int = typer.Option(4096, min=1, help='Exact maximum token count including Qwen chat-template/tool overhead.'),
    seed: int = typer.Option(42),
    max_examples: int | None = typer.Option(None, min=1, help='Optional cap on the without-replacement mixed cohort.'),
    mix_ratio: list[str] | None = typer.Option(None, '--mix-ratio', help='Repeat pipeline1=0.4, pipeline2=0.3, pipeline3=0.3 assignments.'),
    split_ratio: list[str] | None = typer.Option(None, '--split-ratio', help='Repeat train=0.9, validation=0.05, test=0.05 assignments.'),
    config: Path | None = typer.Option(None, '--config', exists=True, dir_okay=False, help='Strict JSON Pipeline4Settings overrides.'),
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
        values=settings.pipeline4.model_dump()
        if config:
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
        result=run_pipeline4(input_directory,output_directory,settings,max_tokens,seed=seed,max_examples=max_examples,
            allow_partial_sources=allow_partial_sources,progress=lambda text:typer.echo(text,err=True))
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


def main() -> None:
    app()
