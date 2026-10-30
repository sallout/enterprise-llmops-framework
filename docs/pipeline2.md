# Pipeline 2 — Repository-to-Training-Data

> **Current export contract:** see [shared schema and portability](data-portability.md).
> New exports use `unified-qwen-sft-v1`; previous native schema descriptions below
> also document retained internal/legacy records. Existing exports are not migrated.
> Rerunning into an existing exported output now requires explicit `--overwrite`.


## Run

```sh
uv sync --locked
uv run llmops pip2_code --input ./code_repos --output ./artifacts/pipeline2 --max-tokens 2048
```

A limited smoke test:

```sh
uv run llmops pip2_code \
  --input ./code_repos \
  --output ./artifacts/pipeline2_test \
  --max-tokens 2048 \
  --limit-units 4 \
  --task-types explanation,architecture,implementation,unit_tests
```

The optional total unit limit uses round-robin selection across repositories; it is not a document or file limit. Without it, every eligible production source unit is considered. Task types rotate across the selected units: `implementation`, `explanation`, `debugging`, `unit_tests`, `refactoring`, `architecture`, `secure_coding`. Counts show actual generated/accepted types, not assumed coverage. Large runs can require many paid API calls.

The existing `.env` and `MODEL__*` settings are reused unchanged. `PIPELINE2__*` controls response size (8192), request timeout (180 seconds), attempts per model call (3), maximum implementation corrections (2), maximum file bytes (250000), execution mode (`auto` or `disabled`), Python image (`python:3.12-slim`), sandbox timeout (15 seconds), and maximum captured feedback (12000 bytes). The default response limit and source chunk limit are independent.

## Data flow

A compiled LangGraph runs static indexing → generation with immediate per-unit validation/isolated verification and correction → validation checkpoint barrier → export. A completed validation checkpoint is saved before the next source unit is generated, preserving validated progress if a later request is interrupted. Graph state contains stage names and artifact paths. Shared Pipeline 1 components provide LangChain JSON calls, credential handling, Pydantic conversation roles, atomic I/O, checksummed caches, exact deduplication, and output locking. Pipeline 1's record contract remains compatible; a shared base conversation has pipeline-specific schema versions/metadata.

Python uses the standard-library AST. Kotlin and TypeScript/TSX use bundled Tree-sitter grammars. Parsing never imports source modules, runs project build hooks, or executes repository test suites. Modules, functions, classes, interfaces, methods, and arrow-function declarations are discovered where supported. Whole structural source units exceeding the conservative UTF-8 byte budget are excluded and counted rather than truncated. Test symbols supply contextual evidence and are not separate production anchors.

Repository roots are detected from immediate child directories or a source root with common manifests/source files. Static dependency context comes from pyproject/package/Gradle/version manifests. Relevant tests are selected by source-symbol or filename matches; nearby symbols are included. This is bounded heuristic context, not a complete call graph or dependency resolver. Source code is passed to the LLM along with this context so tasks require actual source analysis.

Production anchors are ordered by available test context, function-like structure, size, then source location. The index preserves repository path/identity, safe-source snapshot, file hash, symbol, inclusive line range, imports, dependencies, tests, and license references. Local `.git` HEAD/refs are read without running Git hooks. Archives without Git metadata have `commit=null`, an explicit warning, and a content snapshot; a commit is never invented.

Generated/build/dependency/binary/symlink/hidden files and credential-like content are excluded. Supported languages are Python, Kotlin, and TypeScript/TSX; Java/JavaScript and arbitrary resources are not parsed as source in this version. Some generated-code locations and migration paths are excluded conservatively. Exclusion counts and paths/reasons are reported without secret contents. Secret detection is heuristic, not a comprehensive secret scanner.

## Training schema

`training.jsonl` is the final accepted Pipeline 2 dataset. Each line follows the existing canonical conversation layout: deterministic `example_id`, ordered system/user/assistant `messages`, and source/validation `metadata`. The schema version is `repository-data-v1`. The system message includes actual source and bounded context so the instruction is independently understandable. The assistant gives the explanation and, for accepted coding tasks, the exact tested final implementation, fixed tests, observed verification feedback, and any captured correction trajectory.

Code metadata includes task type, language, repository identity, commit or snapshot hash, file/hash/symbol/line references, exact source quotes, dependencies, tests, license references, syntax status, explicit verification status, execution command/image/result, correction trajectory, model-assisted grounding review, and duplicate decision. Only actual observed test feedback is used; model-supplied claims that tests passed are rejected. No hidden reasoning traces are requested or stored.

The messages can later be formatted with the Qwen3-14B chat template; metadata stays outside model loss. No tokenizer/model download, trainer, dataset split, or fine-tuning is performed here.

## Execution and correction policy

**Repository and generated code are never executed directly on the host.** Static AST/Tree-sitter parsing is not execution. Python runtime checks require a reachable Docker daemon and an already-local, trusted configured image. The runner does not start Docker Desktop, pull images, install repository dependencies, or run repository scripts automatically.

When available, a content-addressed Python image runs with no network, read-only root and generated-files mount, nonroot UID, no capabilities, no-new-privileges, CPU/memory/PID/file limits, limited temporary space/logging, and an execution timeout. Only a temporary directory containing `solution.py`, `test_solution.py`, and a trusted unittest runner is mounted; no checkout, API key, host home, or Docker socket is exposed. The container is removed after execution, including timeout/failure paths. Docker is an isolation boundary, not a proof against all kernel/container-runtime exploits; only trusted maintained images/runtimes should be configured.

The executable Python subset is deliberately narrow: standard-library modules and behavioral unittest tests importing the solution. Dynamic execution, filesystem/process/network imports, introspection, and skipped/expected-failure tests are restricted. This static preflight is an additional limit, not a replacement for container isolation. Kotlin/TypeScript execution is currently unconfigured and is explicitly unverified, even if Tree-sitter reports valid syntax.

After actual container test failure, the LLM receives the original task, fixed tests, implementation, and real captured feedback. It may correct only the implementation, within a configured bound. Tests are deliberately fixed to prevent weakening an oracle to manufacture success. Syntax and secret checks precede rerunning. A successful corrected example retains the initial implementation, observed failure, concise change summary, final code/tests, and real verification results. This provides an observable Problem → Implementation → Test → Error → Correction → Verification trajectory without hidden reasoning. It does not implement unconstrained automatic test rewriting.

## Validation and dispositions

- `verified`: an executable candidate passed static/source/review gates and the restricted runner reported positive test count, zero failures/errors/skips/expected failures/unexpected successes, and exit 0. This establishes only that suite's observed result, not universal correctness.
- `unverified`: execution unavailable/not supported, or runtime not applicable to a knowledge task. Coding candidates are quarantined in `rejected.jsonl`; grounded explanation/architecture candidates may enter training with explicit runtime-unverified and not-applicable metadata.
- `failed`: required schema/source/syntax/review checks or actual tests failed. Reasons and execution feedback distinguish validation failures from tests that actually ran; a syntax failure is never called a runtime test failure.

Pydantic validates candidates. Unit/language/task identity and exact source quotes establish references. Syntax gates use appropriate parsers. A separate call reviews grounding, self-containment, technical soundness, and test meaningfulness. Exact normalized question/answer duplicates are rejected globally. The same configured model generates and reviews, so this does not provide independent human evaluation. Passing generated tests can miss bugs; tests may have wrong expectations or weak coverage despite review. Claims of confirmed repository bugs/vulnerabilities require evidence; hypothetical mutation exercises must be identified as such.

`execution_requests_this_run` counts requests to the executor, including unavailable outcomes. `execution_attempts_this_run` counts Docker invocations with confirmed positive test collection; `tests_run_this_run` totals those reported tests. Mocked sandbox results in pytest are not actual Docker evidence. Cached trajectories retain earlier results while current-invocation execution counts stay zero.

## Outputs and recovery

```text
<output>/
├── training.jsonl
├── rejected.jsonl
├── report.json
├── .pipeline1.lock        # shared output-lock helper
├── _index/               # latest safe source inventory
└── _runs/<run-id>/
    ├── index/
    ├── generation/
    ├── validation/
    └── last_execution_report.json
```

`rejected.jsonl` includes explicit rejected/quarantined examples and operational unit failures; these are not training inputs. `report.json` contains actual counts/types/languages, source exclusions, runtime availability, verification statuses, API calls, resume statistics, and export hashes. The shared lock filename is inherited infrastructure, not another pipeline invocation.

Run identity incorporates safe source inventory, selected units, model/endpoint fingerprint, configuration, prompt and Pipeline 2 source-code hashes, dependencies, and runtime/image capability. Credentials are excluded. Completed generation/terminal validation decisions are reused with payload checksums; validation keys also include the actual generated candidate, avoiding reuse against a changed response. Code/runtime/configuration changes select fresh checkpoints. Failed provider calls can retry on a later run. Terminal failed test suites remain auditable; changing correction/configuration/code identity permits a new run.

Exports replace rather than append. Report hashes/status must be checked after interruption; atomic per-file publication is not a directory-wide transaction. The root research report gets a run-specific update. API exception bodies and credentials are suppressed. Optional model-call tracing is disabled.

`COMPLETED_WITH_WARNINGS` is expected when only nonexecuting knowledge tasks are accepted or some code is quarantined. Mandatory provider/discovery failures or no accepted examples produce `FAILED` and exit 1. Unknown task types/invalid options fail explicitly. No full-repository build success, real runtime verification, or model improvement is inferred from syntax checks or mocked tests.

## Validation before server transfer

Run `uv sync --locked`, `uv run pytest -q`, `uv build`, and `uv run llmops pip2_code --help` for local installation, automated regression, package, and CLI checks. Mocked endpoint/execution tests verify workflow behavior, not live model or runtime correctness.

Use a separate output directory for the real endpoint test:

```sh
uv run llmops pip2_code \
  --input ./code_repos \
  --output ./artifacts/pipeline2_preflight \
  --max-tokens 2048 \
  --limit-units 2
```

Inspect `failed_units`, `pending_units`, `rejection_reasons`, runtime capability, and accepted verification metadata. Failed model calls store safe error categories in rejection diagnostics. A provider `finish_reason=length` is reported as `response_truncated`; increasing `PIPELINE2__RESPONSE_MAX_TOKENS` is separate from the source `--max-tokens` budget and must respect the endpoint's capacity. Never interpret pending work as a completed failed validation.

Server acceptance still requires a running Docker daemon with the configured Python image already available locally, an actual positive isolated Python test run, and a limited live generation run with documented results. Images are never pulled automatically. Kotlin/TypeScript runtime verification is not implemented; executable examples for those languages remain quarantined. Full multilanguage runtime readiness cannot be claimed until those executors are implemented and tested. Do not bypass quarantine or run generated code directly on the host.
