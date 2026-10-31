# Corporate Policy Dissertation Framework

## Start the interactive workspace

```bash
uv sync --locked
uv run llmops
```

Choose enterprise extraction/validation, NVIDIA preparation/validation or environment
setup with arrow keys/numbers. Forms collect paths, token budgets, local test limits,
source revisions, output/resume and optional private publication; no pipeline commands
need to be typed manually. Existing coding data can be reused without generation.
Linux/CUDA model compression and fine-tuning is available as an explicit action;
Model security assessment is available (limited post-training scope); the separate future customization entry remains unavailable. Prompt injection is outside scope. Secrets are hidden and `.env` editing requires explicit confirmation with
a private backup. See [the interactive CLI guide](docs/interactive-cli.md).

The console displays **Bridging the Cusem**, **By Mohammed Sallout**, and
[LinkedIn](https://www.linkedin.com/in/msallout/). Branding is configurable in Env setup.
`uv run llmops menu` is an alias; all existing direct commands/`--help` remain available.


A Python LLMOps research framework for the dissertation. The preferred enterprise
workflow combines corporate-policy documents and source-code repositories into one
validated conversational dataset, accepting either input independently and reusing
existing accepted exports when requested. The external-data workflow prepares NVIDIA
datasets; final preparation validates, deduplicates, mixes and splits enterprise and
external records using the actual Qwen tokenizer. The original document and repository
commands remain compatible. Experimental Linux/CUDA model compression and fine-tuning
is implemented; real GPU execution must be verified on the training server.

## Model compression — Linux / NVIDIA CUDA

The interactive **Model compression & fine-tuning** action and `compress_model`
command run training-only input validation, limited structural Qwen3 MLP channel
pruning, BF16 LoRA recovery/merge and NF4 4-bit checkpoint export/reload. This is
experimental: final quality/security evaluation remains separate. macOS/MPS/CPU
training is intentionally unsupported. The old t3.small server cannot run it.

On a suitable Linux GPU server:

```bash
uv sync --locked --extra training
uv run llmops
```

Start with the environment-check action, then the opt-in tiny CUDA integration and
a bounded 14B smoke experiment described in [the server runbook](docs/compression.md).
BF16 pruning/recovery requires substantially more VRAM than 4-bit inference; the
capacity guard is not a measured peak requirement. No weights were downloaded and
no GPU training was verified during local implementation. See the runbook for the
output contract, resumption and full configuration.

## External Data — NVIDIA

The external stage is now available as `nvidia_data`: it inspects and pins both native
NVIDIA sources, streams/adapts them, validates exact Qwen tokens and enterprise holdout
isolation, and writes a resumable source plan without rewriting the entire dataset.

```bash
uv run llmops nvidia_data \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/nvidia_local \
  --max-samples 20 --max-records-per-dataset 30 --shuffle-buffer 2
```

Repeat the same command to resume; increase `--max-samples` with a separate output on
the server for larger preparation. `HF_TOKEN` is optional for public access; no LLM
credentials, GPU or Docker are required. The original enterprise/coding datasets are
read-only inputs. See [native NVIDIA preparation and server reuse](docs/nvidia-data.md)
for source allocation, revisions, recovery, the approved-selection replay API and limits.
This produces data-loading artifacts, not a training run.

## Current workflow: no separate Pipeline 4 step

`enterprise_data` now automatically validates exact Qwen token lengths, removes exact
and near-format duplicates, groups related sources, audits leakage, and exports
`train.jsonl`, `validation.jsonl` and `test.jsonl`. `training.jsonl` is the retained
all-split audit union: train on **train.jsonl**, not that union. Source-generation
`--max-tokens` and final `--training-max-tokens` are separate (defaults 2048 / 4096).

```bash
uv run llmops enterprise_data \
  --documents ./dumy_policies --code ./code_repos \
  --output ./artifacts/enterprise --training-max-tokens 4096
```

For already generated enterprise data, do not regenerate documents or code:

```bash
uv run llmops enterprise_prepare \
  --input ./artifacts/enterprise_combined \
  --output ./artifacts/enterprise_ready --max-tokens 4096
```

NVIDIA datasets can stay in their original Hub format. The native streaming adapters
in `data/training_inputs.py` convert/filter records in memory and retain externally
reported tests as external evidence. No NVIDIA JSONL export or merge is required.
A bounded, CPU-only compatibility check is available:

```bash
uv run llmops check_training_sources \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/training_input_check --max-external-records 10
```

This checks both native datasets against all enterprise splits and writes only an
audit/fingerprint index. It is not a full-corpus audit or training run. The future
trainer must index all selected holdouts before accepting training rows through the
shared cross-source gate. Mixing weights and the model trainer remain future work.
`pip3_datasets` remains an optional cached-export workflow; `pip4_prepare` remains a
hidden compatibility command, but is no longer required by the new workflow. No model
weights, GPU or mandatory Docker are needed for these data checks. Small generation
tests can use `--skip-preparation`; their output is not final training-ready data.

## Enterprise data: one dataset from policies and code

Use `enterprise_data` with `--documents`, `--code`, or both. The tested document and
repository components remain internal branches; the old commands stay available for
compatibility. The new output is one `training.jsonl` with domain and source provenance.

```bash
uv run llmops enterprise_data \
  --documents ./dumy_policies \
  --code ./code_repos \
  --output ./artifacts/enterprise \
  --max-tokens 2048
```

To avoid rerunning existing coding work, replace `--code` with
`--code-dataset ./artifacts/pipeline2`. If its producer report is incomplete, add
`--allow-incomplete-import` to explicitly reuse only the accepted subset. Existing
coding training files remain untouched. Use `--limit-documents 1` / `--limit-units 1`
with a separate output for generation tests. Add `--push-to-hub OWNER/enterprise-sft`
for optional one-dataset private publishing; incomplete source subsets also require
`--allow-incomplete-run` to publish. See [the enterprise guide](docs/enterprise.md).

Enterprise splitting is automatic. NVIDIA sources can be loaded directly through the
shared adapters instead of requiring a combined export.

## Portable training datasets (Pipelines 1–4)

All new accepted exports use one NVIDIA-inspired `unified-qwen-sft-v1` schema:
`id`, `messages`, `domain`, `source`, `verification`, and provenance `metadata`.
Existing datasets remain readable through legacy adapters. Optional tool definitions
and multi-turn/tool messages are preserved. See [the shared contract and workflow](docs/data-portability.md).

- **Mac:** run `enterprise_data` for policy/code inputs; optionally add `--push-to-hub OWNER/DATASET`
  to publish accepted data privately using `HF_TOKEN`.
- **CPU server:** prepare existing enterprise records with `enterprise_prepare` as needed,
  then check direct native NVIDIA inputs with `check_training_sources`.
- **GPU server:** use validated Pipeline 4 outputs in a future trainer; Pipeline 5
  is not part of this implementation.

Edit the example YAML with your Hub namespace. Relative paths resolve against its
folder. Hub revisions and hashes are pinned. Publishing is disabled by default;
failed uploads retain local exports. Existing output files are protected: use a new
output directory or explicitly add `--overwrite` to resume/replace exported files.
The original Pipeline 4 `--input ./artifacts` interface remains available.

## Development setup

Install [uv](https://docs.astral.sh/uv/), then run these commands from the repository root:

```sh
uv sync --locked
uv run policy-dissertation --help
uv run policy-dissertation status
uv run pytest
```

The project requires Python 3.12 or newer; `.python-version` selects Python 3.12 for development. uv creates `.venv/` and installs the local package in editable mode. Commit `uv.lock` to preserve the resolved environment. To build distributable packages, run `uv build`.

## Configuration

Pydantic Settings reads `.env` in the current working directory and then applies environment-variable overrides. Run the CLI from the repository root to use the existing `.env` and default policy directory. The existing resources are preserved:

- `dumy_policies/`: synthetic corporate policy inputs reserved for Pipeline 1.
- `.env`: local LLM settings and credentials. Never commit or print this file.

Supported settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL__BASE_URL` | unset | Future LLM endpoint |
| `MODEL__API_KEY` | unset | Future LLM credential, stored as a masked secret |
| `MODEL__MODEL_NAME` | unset | Future model identifier |
| `POLICY_DIR` | `dumy_policies` | Policy input directory |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, or CRITICAL |

Endpoint settings remain optional during initialization. `status` validates configuration types and reports the current stage; it does not check endpoint connectivity, authenticate, read policy documents, or validate future pipeline requirements. Do not log settings or explicitly extract credentials outside future authorized client setup.

## Architecture

```text
src/policy_dissertation/
├── __init__.py
├── __main__.py
├── cli.py
├── config.py
├── errors.py
├── logging.py
└── pipelines/
    ├── __init__.py
    ├── policy_transformation/  # schemas.py, parsers.py, ingestion.py
    ├── repository_data/       # Pipeline 2 source-grounded examples
    ├── external_datasets/     # Pipeline 3 streaming adapters and checkpoints
    ├── dataset_preparation/   # Pipeline 4 unified validation and Qwen preparation
    ├── training/__init__.py
    ├── quantization/__init__.py
    ├── security_auditing/__init__.py
    ├── benchmarking/__init__.py
    └── deployment/__init__.py
tests/test_scaffold.py
pyproject.toml
uv.lock
report.md
```

The `src/` layout ensures imports resolve through the installed package. The policy transformation package contains document schemas, parsers, batch ingestion, quality assessment, normalization, and section-aware chunking. Repository-data, external-dataset, and dataset-preparation packages implement Pipelines 2, 3, and 4; training, quantization, auditing, benchmarking, and deployment remain empty namespaces. Parser dependencies load only when the parse command runs.

Application logs go to stderr. Expected application failures use `DissertationError` with safe messages and exit code 1; configuration failures return 2 without exposing validation inputs. Unexpected failures return 1 with a generic error. Help exits successfully without loading configuration. The module entry point is also available through `uv run python -m policy_dissertation --help`.

## Research and development

Keep implementation decisions, evidence, experiment plans, and limitations in [report.md](report.md). Add functional modules and meaningful tests as each research milestone is approved. Generated datasets, model artifacts, logs, caches, credentials, and build outputs are excluded from version control; the existing `dumy_policies/` source documents remain eligible for version control.

Next implementation scope awaits instruction. Model fine-tuning, quantization, security auditing, benchmarking, and deployment remain unimplemented. The initialization milestone downloaded no datasets or standalone models; later Pipeline 3 verification ingested only bounded external samples.


## Pipeline 1: document ingestion and parsing

```sh
uv run llmops policies parse --help
uv run llmops policies parse --input dumy_policies/ --output artifacts/pipeline1/parsed
# The original executable supports the same command:
uv run policy-dissertation policies parse --input dumy_policies/
```

Without explicit flags, `POLICY_DIR` defaults to `dumy_policies` and `PARSED_OUTPUT_DIR` defaults to `artifacts/pipeline1/parsed`. Input and output directory trees must not overlap. Discovery recursively enumerates files in sorted order, reports unsupported extensions, and skips symbolic links. Sources are never rewritten. UTF-8 TXT and Markdown retain original text, line endings, headings, lists, and tables. A UTF-8 BOM is decoded without becoming content. Other encodings fail explicitly rather than silently losing characters.

PDF extraction uses PyMuPDF 1.26.x and PyMuPDF4LLM 0.0.27 in a local, model-free mode. Per-page Markdown preserves inferred structure where possible; raw page text provides an additional audit reference. Physical page numbers are 1-based. OCR, image extraction, semantic chunking, LLM calls, and training-example generation are disabled or absent. Scans, password-protected files, corruption, invalid encodings, and empty documents receive failure artifacts. If any PDF page is blank or fails extraction, the document is conservatively marked failed, and available page content is retained. Legitimate blank pages may therefore require manual review.

Outputs:

```text
artifacts/pipeline1/parsed/
├── documents/<document_id>.json
└── summary.json
```

Each document artifact has `schema_version`, `document_id`, `source_path`, `source_type`, `content_hash`, `extracted_content`, `extraction_metadata`, and `processing_status`. Metadata records parser versions, content format, page count, page content, raw page text, warnings, and errors. TXT and Markdown have no fabricated page numbers. `content_hash` is SHA-256 of source bytes; it is null when the source was not read (unsupported or unreadable). `document_id` is SHA-256 of the normalized absolute source file URI: edits preserve identity but change the content hash; relocating a source changes its identity.

`summary.json` holds counts, discovery warnings, and artifact references, never document text. The ingestion API returns this same lightweight summary, suitable for future orchestration without putting document contents into graph state. No LangGraph dependency or graph is introduced.

Exit code 0 means processing completed without failed documents or discovery errors; unsupported files are reported separately and do not cause a nonzero exit. Exit code 1 reports partial document failures, discovery errors, invalid directory selection, or output-write failures. Configuration validation errors return 2. Failure of an output write stops execution because a trustworthy artifact cannot be stored. Successful writes use atomic replacement. Reruns overwrite artifacts for the same source identities and refresh the manifest; artifacts for removed sources remain on disk, so consumers must use the current manifest.

Extraction is not a fidelity guarantee. Actual policy inspection found merged table-header words and bullet glyphs decoded as `G`; these are retained rather than guessed. Heading levels, reading order, whitespace, and complex tables may be inaccurate. Consult original PDFs and retained raw page text for review. Research verification and actual processing counts are recorded in [report.md](report.md).


## Pipeline 1: quality, normalization, and structure-aware chunking

```sh
uv run llmops policies chunk --help
uv run llmops policies chunk --input artifacts/pipeline1/parsed --output artifacts/pipeline1/chunks --max-tokens 2048
uv run pytest
```

Defaults use the existing `PARSED_OUTPUT_DIR`; `CHUNK_OUTPUT_DIR` defaults to `artifacts/pipeline1/chunks`. The original `policy-dissertation` executable supports the same command. This stage consumes `summary.json` and its referenced parsed artifacts; it does not parse source PDFs again. Nonparsed documents are skipped, invalid artifacts are reported without stopping other documents, and input/output directories cannot overlap. Document failures cause exit 1 after the batch report is written. Quality warnings and oversized chunks remain reviewable output and do not by themselves cause a nonzero exit.

New modules under `src/policy_dissertation/pipelines/policy_transformation/`:

| Module | Role |
| --- | --- |
| `chunk_schemas.py` | Typed chunks, source spans, normalization/quality reports, batch manifest |
| `structure.py` | Conservative Markdown and numbered-section recognition |
| `normalization.py` | Quality diagnostics, safe changes, immutable originals, source-offset mappings |
| `chunking.py` | Section hierarchy, clause grouping, byte-token accounting, stable IDs, coverage checks |
| `chunk_batch.py` | Existing ingestion-manifest consumption, JSONL output, separate reports and statistics |

```text
artifacts/pipeline1/chunks/
├── documents/<hash-of-document-id>.jsonl
├── normalization/<hash-of-document-id>.json
├── quality/<hash-of-document-id>.json
└── summary.json
```

Each JSONL line is a `PolicyChunk`. Fields include chunk/document/source identifiers, source hash/path, section ID/title, parent contexts, heading provenance, content-page numbers, content, token count/method, configured limit, oversized flag, line IDs, original character spans, normalization status, and structured quality warnings. All character offsets are half-open Python string indices into the original page's extracted Markdown; for nonpaged text they refer to `document.extracted_content`. Heading context has its own source spans, so a context heading can reference an earlier page than the chunk body. No chunk content enters batch state; the batch summary stores output references and counts.

Normalization reports retain original and normalized text separately, plus every original line, normalized line, source span, join separator, exclusion, and action. Original parsed JSON files are unchanged. The quality report records warning code, location, message, deterministic local warning ID, and `corrected`/`unresolved`/`informational` resolution. Findings are heuristic and may include false positives. Counts are findings, not accuracy scores.

Automatic changes are deliberately narrow: newline/blank-line whitespace, a single trailing space without changing Markdown hard breaks, explicit Unicode list markers converted to Markdown, repeated allowlisted nonnormative page-edge boilerplate, and unfinished prose joined to a lowercase continuation on the next page. Repetition requires at least three pages and 75% document coverage. Repeated unknown text or mandatory requirements are retained. Ambiguous `G` bullets, malformed/merged table text, and uncertain boundaries are flagged without speculative replacement. Code fences/indented code and table spacing receive conservative handling. The source-to-normalized invariant allows only whitespace, explicit marker normalization, and recorded allowlisted exclusions; lexical policy content is preserved.

Chunking follows Markdown headings and explicit numbered ancestry. A list clause includes subsequent paragraphs until another clause or section; marked conditional continuations stay with the previous clause. Uninterrupted prose runs are retained conservatively. Adjacent units pack within the same section and never overlap. Parent context is metadata rather than repeated body text. Numeric ancestry is retained and warned about if PDF heading depth is inconsistent. This is a deterministic structural approximation of semantic units, not model-verified semantics. Conditions in separately numbered sections or indirect/global references are not automatically resolved.

`--max-tokens` uses **UTF-8 byte count**, named `utf8_bytes_upper_bound_v1`. The default 2048 bounds ordinary byte-BPE content tokens conservatively; it does not mean 2048 tokens for a selected model. Metadata, prompt wrappers, and future chat formatting are outside the budget. No model tokenizer is selected or downloaded. Clauses/prose runs that cannot safely fit remain complete, receive `oversized_clause`, and set `oversized=true`. No truncation or silent fallback splitting occurs. Chunk identifiers hash the algorithm version, source identity/hash, exact content, ordered line IDs, and section context. Changing a limit preserves IDs where grouping/content stays unchanged.

Reruns replace outputs deterministically; consumers must follow the current summary rather than glob potentially stale files. Review unresolved warnings before dataset generation. Current verified corpus counts, privacy-policy inspection, and remaining risks are recorded in [report.md](report.md).


## Complete Pipeline 1: one command

```sh
uv sync --locked
uv run llmops pip1_data --input ./dumy_policies --output ./artifacts/pipeline1 --max-tokens 2048
```

The LangGraph workflow runs parsing → normalization/chunking → LLM generation → validation → export automatically. It reuses the existing components and loads the existing `MODEL__BASE_URL`, `MODEL__API_KEY`, and `MODEL__MODEL_NAME` through Pydantic Settings. Run from the repository root so `.env` and `report.md` resolve as intended. The endpoint must support OpenAI-compatible chat completions. This command sends the synthetic policy text to that configured endpoint.

User-facing outputs:

- `training.jsonl`: accepted conversational examples (`system`, `user`, `assistant`) with source/evidence metadata.
- `rejected.jsonl`: rejected candidate examples with reason codes, plus separately typed operational chunk failures.
- `report.json`: actual counts, status, quality findings, API-call/resume statistics, output hashes, and errors.

Internal parsed/chunked artifacts and checksummed generation/validation checkpoints live in `_runs/<run_id>/`. Formatting warnings alone do not block an example. Generation asks for one question/answer per chunk (at most three returned candidates are supported). Answers must reproduce a complete supplied source passage with only whitespace normalization. A separate model-assisted review checks relevance, grounding, conditions, and scope; full policy-wide Scope/Exceptions sections are retained as system-message context when recognized. Accepted records passed these automated checks, not human or legal certification.

`--max-tokens` remains the existing **source chunk UTF-8 byte-token limit**; it is not the response limit. Generation settings can be overridden through environment variables without editing `.env`:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PIPELINE1__RESPONSE_MAX_TOKENS` | 4096 | Provider generation output limit per generation/review call |
| `PIPELINE1__MAX_ATTEMPTS` | 3 | Bounded attempts per JSON call, including transport and malformed-output retries |
| `PIPELINE1__REQUEST_TIMEOUT` | 60 | Per-request timeout in seconds |

Oversized clauses are retained by chunking and rejected from generation rather than silently truncated or sent above the chunk limit. Unsupported files are reported; supported-document parsing failures and incomplete LLM/validation processing cause `FAILED` and exit 1. All-rejected runs also fail. Some rejected candidates with accepted records produce `COMPLETED_WITH_WARNINGS` and exit 0. Successful runs with retained source-quality caveats also use that warning status. Invalid settings produce exit 2.

Run the same command to resume. Successful generation and terminal validation decisions are reused per chunk; failed/incomplete calls can be retried. Source/configuration/prompt/dependency changes select another run fingerprint. Exports are rebuilt deterministically from current checkpoints rather than appended. Exact normalized question/answer duplicates are rejected globally. An output-directory lock prevents concurrent writers. The report is published last with hashes for the two JSONL files; consumers must check its status/hashes. Individual file writes are atomic; a hard process termination during publication is not a whole-directory transaction.

`report.md` receives a run-specific actual-results section automatically. Repeat invocations update that run's section rather than append duplicates. No credentials, raw provider error bodies, or hidden reasoning are exported. Provider errors are classified without displaying the credential-bearing connection configuration. Optional model-call tracing is disabled.

See [the Pipeline 1 contract](docs/pipeline1.md), [the canonical JSON schema](docs/schemas/pipeline1-training.schema.json), and [the research report](report.md). No Qwen model download, tokenizer conversion, training split, or model fine-tuning occurs in this workflow.

### Limit documents for a test run

```sh
uv run llmops pip1_data \
  --input ./dumy_policies \
  --output ./artifacts/pipeline1_test \
  --max-tokens 2048 \
  --limit-documents 1
```

Omit `--limit-documents` to process all supported documents. The limit counts supported source files attempted, including empty/corrupted files; unsupported files and symbolic links do not consume it. Selection follows deterministic sorted traversal: files within each directory first, then sorted subdirectories. Every chunk of each selected document is processed, so one document can still require multiple model calls. Discovery and fingerprinting still inspect the input tree.

Reports record the limit, supported documents discovered, selected/excluded counts, and exact source paths. Changing the limit selects a different checkpoint identity; repeated runs with the same limit can resume. A test subset is never reported as the whole corpus. Use a separate test output directory, as each invocation replaces the destination's public training/rejection exports with the current run's results.

The [research report](report.md) explains the transition from noisy extraction to normalized, traceable clauses and then automatically validated conversational records. “Clean” means passed the documented gates; ambiguous glyphs/table spacing and semantic uncertainty can remain visible in metadata.

## Pipeline 2: repository-to-training data

```sh
uv run llmops pip2_code \
  --input ./code_repos \
  --output ./artifacts/pipeline2 \
  --max-tokens 2048
```

For a small test, add `--limit-units 4` and use a separate output directory. `--task-types` accepts a comma-separated list; defaults cover implementation, explanation, debugging, unit tests, refactoring, architecture, and secure coding, rotated across actual source units.

Python AST and Kotlin/TypeScript Tree-sitter indexing provide source, symbol, dependency, and test context to the configured LLM. LangGraph orchestrates generation, validation, optional restricted-container testing, bounded correction from real failures, and resumable JSONL export. `training.jsonl` is the final accepted dataset; `rejected.jsonl` includes failed/quarantined candidates and reasons; `report.json` contains actual statistics and verification status.

Untrusted code is never run on the host. Python execution needs a running Docker daemon and a trusted local `PIPELINE2__PYTHON_IMAGE`; the pipeline does not install/start/pull these automatically. Kotlin/TypeScript runtime checks are currently unavailable. Unexecuted coding examples are quarantined; grounded explanation/architecture records can be retained with explicit runtime-unverified labels. Syntax validity does not imply runtime correctness. See [the full contract](docs/pipeline2.md) and [research results](report.md).

### Validation before server transfer

See [server validation](docs/server-validation.md) for local checks and server acceptance gates. Pipeline 2 uses an 8192-token model response budget independently of the source chunk limit. Each generated unit is validated and checkpointed before generating the next unit. The CLI distinguishes failed from pending units and prints failure categories. Kotlin/TypeScript execution backends remain unimplemented; Python runtime verification requires Docker and a prepared local image. Local mocked tests do not establish real execution correctness.

## Pipeline 3: external dataset preparation

```sh
uv run llmops pip3_datasets \
  --output ./artifacts/pipeline3 \
  --max-samples 10000
```

The total accepted cap is shared equally by default between `nvidia/Nemotron-Cascade-2-SFT-Data` (`chat` config) and `nvidia/OpenCodeInstruct` (`train` config). Select sources with repeated `--dataset nemotron` / `--dataset opencode`, set accepted quotas with `--allocation nemotron=6000 --allocation opencode=4000`, and configure `--seed`, `--config`, `--split`, `--revision`, or `--filter-config`. No complete dataset, Qwen weights, or tokenizer is downloaded by default.

LangGraph orchestrates pinned metadata inspection, bounded prefix-buffer streaming, dataset-specific normalization, structural filtering, disk-backed exact deduplication, and resumable export. Multi-turn/tool messages and original test metadata are preserved. Closed leading reasoning blocks are removed by an explicit recorded rule; strict rejection is configurable. External answers/tests remain unverified locally. Final outputs are `training.jsonl`, `rejected.jsonl`, and `report.json`; internals support audit/recovery. No LLM endpoint, Docker, GPU, Pipeline 4 mixing, or fine-tuning is required.

See [Pipeline 3 documentation](docs/pipeline3.md) for licenses, schema, filtering, sampling limitations, and actual measurement definitions.

## Pipeline 4: unified validation and Qwen preparation

```sh
uv run llmops pip4_prepare \
  --input ./artifacts \
  --output ./artifacts/pipeline4 \
  --max-tokens 4096
```

Loads the three upstream `training.jsonl` files and verifies their reports/hashes, adapts source-specific schemas, filters invalid examples, deduplicates/groups related examples, balances sources without replacement, and exports `train.jsonl`, `validation.jsonl`, `test.jsonl`, `rejected.jsonl`, `manifest.json`, and `report.json`. Actual pinned Qwen3-14B tokenizer assets and chat templates measure complete inputs without truncation; no model weights, GPU, Docker, LLM calls, or training are needed.

Defaults are 40/30/30 source weights and 90/5/5 split targets, configurable with repeated `--mix-ratio NAME=value` and `--split-ratio NAME=value`. Group constraints may change exact split sizes. Scarce sources limit the balanced cohort; records are never repeated to fill quotas.

The current checkout lacks Pipeline 1 exports and Pipeline 2's report is failed, so the normal command correctly fails and reports `training_ready=false`. Explicit `--allow-partial-sources` supports diagnostic integration only and never claims full readiness. See [Pipeline 4 documentation](docs/pipeline4.md) for the actual partial-validation command, source requirements, verification distinctions, tokenizer contract, and limitations.

To reuse a published enterprise dataset on another machine without regenerating
policy/code examples, use the interactive **Enterprise data validation → Download
an existing enterprise dataset from Hugging Face** option, or:

```bash
uv run llmops download_dataset --repo-id sallout/enterprise-sft --output ./artifacts/enterprise_ready
```

Configure a read-capable `HF_TOKEN` for private repositories. The download verifies
checksums, schemas, counts and prepared split membership, preserving source warnings.

### Model evaluation

The interactive menu now includes **Model evaluation**: freeze a configurable
HumanEval/MBPP base-test selection, compare original/final Qwen3 checkpoints (plus
optional intermediate stages), and review held-out enterprise answers. Use
`--max-tasks` per coding benchmark and `--enterprise-limit` for enterprise questions;
zero selects all available tasks. Results, model identities and resumable task
checkpoints are exported separately from training data. No training is started.

```bash
uv run llmops evaluate_model \
  --compression-run /workspace/artifacts/compression_opencode_10k \
  --enterprise ./artifacts/enterprise_ready \
  --output /workspace/artifacts/evaluation_pilot \
  --max-tasks 5 --enterprise-limit 5
```

Inference requires Linux/NVIDIA CUDA and the existing `training` extra. Executable
code scoring requires a Docker daemon and a local Python execution image. On
RunPod without Docker, select generation-only mode (`--executor disabled`), then
score the saved export using `llmops score_evaluation` on a Docker host, without
rerunning the GPU. Generation-only output has no pass@1 score. Enterprise lexical
scores require human semantic review. These are base tests, not EvalPlus + suites.
See [the evaluation guide](docs/model_evaluation.md) for frozen selections, complete
commands, limits, contamination checks, scoring and real opt-in integration tests.

### Private model experiment backup

The menu includes **Model backup to Hugging Face**. The equivalent command backs
up a completed compression experiment, not only the final quantized weights:

```bash
uv run llmops backup_model \
  --input /workspace/artifacts/compression_opencode_10k \
  --repo-id YOUR_USERNAME/qwen3-enterprise-opencode-10k-backup \
  --output /workspace/artifacts/model_backup_receipt
```

`--repo-id` is optional: the default uses the authenticated account and compression
fingerprint. `HF_TOKEN` needs write access. The destination must be a **private model
repository**; an existing public repository is refused. Keep the receipt separately
from the source experiment. Retrying the same command uses Hugging Face's large
upload cache; no training is started and no remote files are deleted.

The backup checks committed stage hashes, uploads final NF4, BF16 stages, adapters,
optimizer checkpoints and input/report evidence, then verifies every selected
remote file using LFS SHA-256 or Git blob SHA-1, plus its byte length. Local content
is checked again after upload. Ordinary caches, environment/key files and locks
are excluded; cache/lock files already referenced by a committed stage are retained
so that stage hash verification remains possible after restoration. Configured
credentials found in text artifacts stop the upload. This is not a guarantee of
finding every unknown historical secret; the backup remains private.

Successful `report.json` in the receipt directory contains `backup_verified=true`,
the repository, **verified payload `revision`**, manifest commit, file count and byte
count. `file_manifest.json` records all content hashes. Remote `backup_manifest.json`
is committed after payload verification; `manifest_revision` identifies that later
metadata commit. Restore using the verified payload revision, which includes all
selected experiment files and may precede `backup_manifest.json` itself:

```python
from huggingface_hub import snapshot_download
from policy_dissertation.config import Settings
settings = Settings()
snapshot_download(
    repo_id="YOUR_USERNAME/qwen3-enterprise-opencode-10k-backup",
    revision="VERIFIED_PAYLOAD_REVISION_FROM_RECEIPT",
    local_dir="/workspace/artifacts/compression_opencode_10k",
    token=settings.hf_token.get_secret_value(),
)
```

The restored final model is `model_4bit/`; intermediate models retain their existing
`_runs/` layout. This is a full experiment archive, not a root-level directly
loadable model repository. It can be substantially larger than the 4-bit export.
Optimizer resume still requires matching source inputs, paths/configuration and
runtime. Model inference/evaluation can reuse the saved weights without training.
Keep both the receipt and verified repository revision before releasing server
storage. Backup verification establishes byte preservation, not model quality.

### Export a saved trained model to 8-bit

`llmops export_8bit --compression-run PATH --output NEW_PATH` converts the committed
merged BF16 checkpoint to bitsandbytes LLM.int8 on Linux/NVIDIA CUDA, without
retraining or replacing the existing NF4 model. It saves `model_8bit/`, verifies
reload and records a separate report. The interactive compression menu includes
this option. See [the compression guide](docs/compression.md#standalone-8-bit-export-after-training)
for the server command, retry behaviour, evaluation and opt-in CUDA test.


### Post-training security assessment

Use **Model security assessment** in the interactive menu or `llmops security_check`.
This read-only command checks model files, uses the official garak replay detector
in an isolated environment, and runs a limited known-trigger screen. It never
changes training data or weights. Microsoft unknown-trigger scanning is explicitly
NOT_VALIDATED for Qwen3-14B and is not executed. No safety certification is issued.
See [methods, installation, commands, limitations and remediation options](docs/model_security.md).

```bash
uv sync --locked --project tools/security
uv run --locked --extra training llmops security_check \
  --model final=/workspace/path/to/model \
  --enterprise ./artifacts/enterprise_ready \
  --output /workspace/artifacts/security --max-cases 20
```

### Skill-aligned coding tests

The Model evaluation menu can profile a prepared OpenCode selection and freeze
independent official Python tests with full accepted-source overlap screening and
Docker reference verification. See [the method and command](docs/skill-aligned-evaluation.md).
The actual 77,891-row experiment's frozen development/final suites and attribution
are in [evaluation_suites/opencode_77k](evaluation_suites/opencode_77k/README.md).
