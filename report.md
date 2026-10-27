# Dissertation Project Report

Updated: 26 October 2026.

## 1. Project purpose and current deliverable

This project implements **Pipeline 1 — Corporate Policy Transformation**, **Pipeline 2 — Repository-to-Training-Data**, **Pipeline 3 — External Dataset Ingestion & Preparation**, and **Pipeline 4 — Unified Dataset Validation & Training Preparation**. Policy/source-code pipelines use structured analysis and LLM generation; Pipeline 3 streams existing NVIDIA data into separate traceable conversational artifacts without LLM calls.

**Pipelines 1–3 each export accepted records in `training.jsonl`. Pipeline 4 exports separate `train.jsonl`, `validation.jsonl`, and `test.jsonl` files.** Rejected records, reports, manifests, tokenizer assets, and checkpoints support audit/recovery and are not additional training datasets; held-out test records must remain outside training and hyperparameter selection.

Pipelines 1, 2, and 3 have implemented workflows with the verification limitations recorded below. Pipeline 2 runtime verification requires an available restricted container, and Kotlin/TypeScript execution backends remain unimplemented; unexecuted code is never labeled runtime-verified. Pipeline 4 preparation is implemented and uses actual Qwen tokenization, source balancing, and isolated dataset splits, but the current complete three-source run is blocked by missing Pipeline 1 exports and a failed Pipeline 2 report. Model fine-tuning remains unimplemented. Prepared records carry a training-readiness flag; downstream trainers must apply the pinned template and appropriate assistant-only loss masking. No training performance or corpus-wide semantic correctness is claimed.

This report summarizes the current system. The complete earlier report, including implementation milestones, failed attempts, test results, and detailed research observations, is preserved in [the development history](docs/history/report-development-history.md). Earlier artifact paths recorded there may no longer exist in the working directory.

## 2. Running the pipeline

Process all supported documents:

```sh
uv run llmops pip1_data \
  --input ./dumy_policies \
  --output ./artifacts/pipeline1 \
  --max-tokens 2048
```

Process one document for testing:

```sh
uv run llmops pip1_data \
  --input ./dumy_policies \
  --output ./artifacts/pipeline1_test \
  --max-tokens 2048 \
  --limit-documents 1
```

The optional document limit selects supported source files in deterministic sorted directory traversal before parsing. Failed supported files consume a slot; unsupported files do not. Every chunk of a selected document is processed, so one document can require multiple LLM calls. Discovery/fingerprinting still inspect the input tree. Reports distinguish selected documents from intentional exclusions. Omit the option to process all supported documents.

`--max-tokens` limits source chunks using the documented conservative UTF-8 byte-token count. It is **not** the model response limit. Separate environment settings control response size, attempts, and timeout: `PIPELINE1__RESPONSE_MAX_TOKENS` (4096), `PIPELINE1__MAX_ATTEMPTS` (3), and `PIPELINE1__REQUEST_TIMEOUT` (60 seconds).

The existing `.env` supplies the endpoint, model, and API credential without being modified. Original documents in `dumy_policies/` remain unchanged. Run from the project root. Use a separate test output directory because public exports in a destination are replaced by the current run's results.

## 3. What each generated file and folder means

The layout inside the selected output directory is:

```text
<output>/
├── training.jsonl
├── rejected.jsonl
├── report.json
├── .pipeline1.lock
└── _runs/<run-id>/
    ├── parsed/
    ├── chunks/
    ├── generation/
    └── validation/
```

| File or folder | Purpose | Used as training input? |
| --- | --- | --- |
| **`training.jsonl`** | Final accepted conversational records, with source and validation metadata | **Yes: use the `messages` field; keep metadata outside model loss** |
| `rejected.jsonl` | Rejected candidates and their reasons; separately typed operational chunk failures | No |
| `report.json` | Actual counts, status, quality warnings, errors, resume statistics, and export hashes | No |
| `.pipeline1.lock` | Prevents simultaneous writers to the same output directory | No |
| `_runs/<run-id>/` | Internal artifacts for one source/configuration fingerprint | No |
| `parsed/` | Extracted document text, raw page text, source hashes, and page references | No; audit/source evidence |
| `chunks/` | Policy chunks, original/normalized text mappings, quality reports, and manifest | No; generation input and audit evidence |
| `generation/` | Saved generated candidates before final acceptance | No; resume avoids repeating completed calls |
| `validation/` | Saved acceptance/rejection decisions and associated evidence | No; resume and audit |

These folders explain **how the final dataset was produced**. They should not be merged into the training dataset. Keeping them enables inspection and resume after interruption; removing them loses that recovery/evidence. This documentation cleanup does not delete or alter any generated artifacts.

The run ID changes when relevant sources, selection/configuration, prompts, or dependency versions change. A repeated identical run can reuse completed work. Exports are rebuilt rather than appended, avoiding duplicate committed output. Writes are atomic per file, not a whole-directory transaction; consumers should check the final report status and JSONL hashes.

## 4. From dirty extraction to clean, traceable records

“Dirty data” refers to extraction noise and unvalidated generated examples, not an assumption that the original policy requirements are incorrect. “Clean” means a record passed the stated automated acceptance gates; it does not mean every extraction defect or semantic uncertainty has disappeared.

| Stage | Problem addressed | Implemented treatment |
| --- | --- | --- |
| Discover and parse | Unsupported, empty, corrupt, or unreadable files; PDF layout | Discover PDF/Markdown/TXT recursively; record failures; retain original text, page references, deterministic IDs, and byte hashes |
| Assess quality | Suspicious `G` bullets, merged table words, repeated page text, irregular spacing, section/boundary uncertainty | Produce structured findings with locations; retain uncertain content |
| Normalize safely | Harmless whitespace, repeated headers/footers, interrupted page continuations | Record conservative whitespace changes, exclude repeated allowlisted boilerplate, and join reliable continuations; preserve original and normalized forms separately |
| Chunk by structure | Requirements separated from their conditions or exceptions | Preserve section hierarchy, parent context, clause continuations, and page spans; avoid overlapping body text; flag oversized indivisible clauses |
| Generate examples | Unsupported answers or misleading selective quotations | Ask the configured LLM for a specific instruction and a complete extractive source-passage answer, with recognized policy-wide scope/exception context |
| Validate | Invalid schema, wrong references, altered source wording, missing relevant conditions | Apply Pydantic, identity/passage checks, complete-passage equality, credential suppression, and a separate model-assisted grounding/relevance/condition/scope review |
| Deduplicate and export | Repeated conversations and duplicate resume output | Reject exact normalized question/answer duplicates and publish accepted records separately from rejected records |

Mandatory language—MUST, MUST NOT, SHOULD, MAY, and Strictly Prohibited—is not rewritten. Numeric thresholds, approvals, conditions, and exceptions must remain intact. Whitespace changes are distinguished from lexical changes. Source references and recorded normalization actions allow an answer to be traced back to the extracted document and original PDF.

Some visible bullet glyphs extract as `G`, and some table words lose spaces. Unknown replacements are not guessed. Formatting-only warnings do not automatically block a supported example; their metadata remains available for review. This is why accepted records and a `COMPLETED_WITH_WARNINGS` status can coexist.

## 5. Architecture and dataset contract

The implementation uses a `uv`-managed Python `src/` package, Typer CLI, Pydantic Settings, Pydantic schemas, and pytest. PyMuPDF/PyMuPDF4LLM provide PDF extraction; LangChain connects to the configured compatible chat endpoint; a compiled LangGraph orchestrates parse → chunk → generate → validate → export.

Graph state contains stage names and artifact paths rather than document contents. Per-chunk checksummed files provide durable recovery; this implementation does not use a LangGraph SQLite checkpointer. Parsing/chunking rerun deterministically while completed expensive LLM work can be reused.

The existing parsing/chunking components are reused. New generation, validation, checkpoint I/O, and orchestration modules live under `src/policy_dissertation/pipelines/policy_transformation/`. Detailed implementation and contracts are documented in [docs/pipeline1.md](docs/pipeline1.md).

Each accepted record uses schema version `pipeline1-extractive-v1`:

- `example_id`: a deterministic record identifier.
- `messages`: exactly three ordered `system`, `user`, and `assistant` role/content objects.
- `metadata`: document/source hash, chunk/section/parent context, page and character references, exact evidence quote, quality warnings, and validation decisions.

The system message retains recognized policy-wide scope and exceptions. The user message contains the generated instruction. The assistant message reproduces a complete selected passage with whitespace normalization. No hidden reasoning is exported.

The [canonical JSON Schema](docs/schemas/pipeline1-training.schema.json) describes the record. For later Qwen3-14B preparation, select its messages and apply the model's chat template/tokenizer; metadata is not training target text. Template application, thinking-mode selection, splits, tokenization, label masking, and training have not been performed here.

## 6. Verified execution results

### Latest available run: one-document AI-policy test

The current [execution report](artifacts/pipeline1_test/report.json) records a run completed on **25 October 2026 at 22:21 Dubai time**, using `--limit-documents 1` against `dumy_policies/`. It selected the AI/LLM Acceptable Use and Governance Policy (`POL-AI-005`).

| Metric | Recorded result |
| --- | ---: |
| Supported documents discovered | 10 |
| Documents selected / processed | 1 / 1 |
| Documents intentionally excluded by limit | 9 |
| Failed documents / failed chunks | 0 / 0 |
| Chunks generated | 14 |
| Training examples generated | 14 |
| Accepted examples | 14 |
| Rejected examples | 0 |
| Model-call attempts recorded | 28 |
| Quality findings: corrected / unresolved | 7 / 47 |
| Final status | **COMPLETED_WITH_WARNINGS** |

Current dataset: [artifacts/pipeline1_test/training.jsonl](artifacts/pipeline1_test/training.jsonl). The rejected export is empty for this run. Zero rejections does not prove semantic correctness or complete document coverage.

During this documentation cleanup, all 14 accepted records were checked against the canonical Pydantic schema. Example IDs were unique, assistant answers matched their stored evidence after whitespace normalization, and both export hashes matched `report.json`. No pipeline/LLM calls were made and no new test-suite execution is claimed for this cleanup. These checks establish structural/evidence consistency, not independent semantic evaluation.

### Earlier verification, retained as history

The latest previously executed automated suite recorded **89 passed in 2.72s**, with five upstream SWIG warnings and no failures/skips. It covered parsing, chunking, mocked end-to-end orchestration, validation/rejection, resume, output locking, and document limits. Mocked responses are not evidence of real endpoint success.

A separate earlier live privacy-policy integration generated 13 examples: 12 accepted and one rejected for omitting requirements relevant to its question. Its resume check recorded zero additional calls. Earlier full-corpus parsing/chunking processed ten documents and 40 pages into 134 chunks; that does not represent a full-corpus live training-data generation run. Detailed commands, intermediate failures and fixes, and historical statistics remain in [the archived development report](docs/history/report-development-history.md). Some historical artifact directories are no longer present; these historical results were not rerun during this cleanup.

## 7. Status, validation limits, and next stage

- `COMPLETED`: mandatory stages completed, accepted records exist, and no relevant quality/rejection/input warnings remain; exit 0.
- `COMPLETED_WITH_WARNINGS`: accepted records exist alongside retained quality warnings or candidate rejections; exit 0.
- `FAILED`: mandatory processing/verification failed or no records were accepted; exit 1. Configuration/argument errors use exit 2.

Accepted records passed automated schema, provenance, complete-passage, grounding-review, and exact-deduplication gates. The reviewer is a separate call to the same configured model, so correlated mistakes remain possible. Exact copying cannot by itself prove the question is fully answered; contextual omissions, prompt injection, source glyph errors, table compaction, and global exception dependencies remain risks. Exact deduplication does not detect every semantic near-duplicate.

Source chunk counts are UTF-8 byte-token upper bounds, not a selected model's exact token counts. Whole-clause oversize handling is conservative. There is no OCR or guarantee for complex PDF layout. Execution is sequential, endpoint context capacity matters, and cross-platform operation has not been established. Absolute source paths may need redaction before sharing.

Before fine-tuning, review a representative sample of accepted records and unresolved findings, define dataset acceptance criteria, and prepare model-specific formatting and evaluation splits. No legal-validity, semantic-accuracy, model-quality, or fine-tuning-performance claim is made.

## 8. Pipeline 2 — Repository-to-Training-Data

### Workflow and source understanding

```sh
uv run llmops pip2_code \
  --input ./code_repos \
  --output ./artifacts/pipeline2 \
  --max-tokens 2048
```

The single command runs a LangGraph workflow: **static indexing → source-grounded generation → validation / isolated testing / bounded correction → export**. It reuses Pipeline 1's LangChain JSON gateway, Pydantic conversation roles, atomic/checksummed files, exact deduplication, output lock, and existing `.env` configuration. Graph state contains stage/path references rather than source bodies.

Python uses standard-library AST; Kotlin and TypeScript/TSX use Tree-sitter. Added locked dependencies are `tree-sitter` 0.25.2, `tree-sitter-kotlin` 1.1.0, and `tree-sitter-typescript` 0.23.2. Source parsing does not import modules or run repository builds/tests. Static manifest/import context, nearby symbols, and relevant test excerpts accompany actual code in LLM prompts. Test relevance is heuristic, not complete call-graph resolution.

The repository index examined **188 supported source files**, discovered **748 symbols**, and identified **459 eligible production source units** within the configured budget. It counted 43 oversized symbols, 11 invalid symbols, 17 generated-file exclusions, and one credential-like/binary content exclusion. One file had syntax diagnostics; some Kotlin grammar constructs may produce parser limitations. These are indexing counts, not functional correctness or whole-repository coverage claims. Invalid/oversized units are not silently truncated into examples.

The Android repository contributed 61 indexed source files and 159 eligible production units; the FastAPI/Python/TypeScript repository contributed 127 files and 300 units. Both are supplied archives without usable local Git commit metadata: `commit=null` is retained explicitly alongside source snapshots, file hashes, symbols, and inclusive line references. License/notice references are preserved, without asserting that reuse rights have been evaluated.

### Data generation and validation

Task types rotate across actual source units: implementation, explanation, debugging, unit tests, refactoring, architecture/dependencies, and secure coding. Prompts ask for transferable, self-contained engineering tasks grounded in real code, not raw-file JSON conversion or invented confirmed vulnerabilities. A run reports actual types generated; a limited test need not cover all seven.

Each candidate must match the requested task/language/unit, quote exact source evidence, satisfy Pydantic constraints, and avoid unsupported execution claims or credentials. Generated code/tests receive language-aware syntax checks. A separate model call reviews grounding, self-containment, technical soundness, and test meaningfulness. Exact normalized question/answer duplicates are rejected globally. The generator and reviewer share the configured model, so these checks are not independent human semantic validation.

The canonical Pipeline 1 message layout is reused through a shared conversation base: fixed system/user/assistant role/content messages, deterministic example ID, and source/validation metadata. Pipeline 2 uses schema version `repository-data-v1` with repository/commit-or-snapshot/file/symbol/line references, task/language, source quotes, dependency/test/license references, syntax status, execution evidence, and correction trajectory. The system message supplies source context; metadata stays outside model loss during later Qwen3-14B formatting. No model/tokenizer download, chat-template conversion, split, or fine-tuning occurs here.

### Restricted execution and self-correction

**No repository or generated code is run directly on the host.** Python execution requires a reachable Docker daemon and a trusted, already-local image. The runner uses a content-addressed image, no network, a read-only root/files mount, nonroot UID, dropped capabilities, no-new-privileges, CPU/memory/PID/file/log limits, temporary-space limits, and a timeout. Only generated solution/tests plus a trusted runner are mounted; no checkout, credentials, home directory, or Docker socket is exposed. Containers are removed after completion/failure.

The current executable subset supports standalone standard-library Python and meaningful unittest tests importing the solution. Kotlin/TypeScript runtime execution is not configured. Syntax checks are never presented as runtime proof.

When actual container tests fail, bounded correction calls receive the real command/result/error feedback and the fixed tests. The implementation can be corrected and rerun; tests are not weakened or rewritten. An accepted repaired example contains the initial implementation, observed failure, short change summary, final code/tests, and actual verification feedback. Hidden reasoning traces are neither requested nor exported.

On this machine the Docker executable is present but the daemon is unavailable. Therefore **no actual repository/generated-code test execution or successful live self-correction has been demonstrated**. Correction/runner behavior is covered by mocked tests; this is not runtime verification evidence for the existing repositories.

### Output and acceptance rules

Pipeline 2 produces its own `training.jsonl`, `rejected.jsonl`, and `report.json`. Internal `_index/` and `_runs/<run-id>/` folders hold static inventory and resumable generation/validation checkpoints. The shared `.pipeline1.lock` filename is reused output-lock infrastructure. Readers must check report status/hashes; individual atomic writes are not a whole-directory transaction.

| Status on an example | Explicit interpretation |
| --- | --- |
| `verified` | Actual restricted-container tests collected at least one test and completed successfully with no failures/errors/skips/expected failures; all source/syntax/review gates passed |
| `unverified` | Execution unavailable/unsupported, or runtime not applicable to a knowledge task; never a runtime success claim |
| `failed` | Required validation or actual tests failed; reasons distinguish static failures from tests that ran |

Unexecuted coding examples are quarantined in `rejected.jsonl`. Grounded explanation/architecture records may enter training with explicit runtime-unverified/not-applicable metadata. Failed required processing makes a run `FAILED`; valid records can still be preserved for inspection. Knowledge-only success uses `COMPLETED_WITH_WARNINGS`. No accepted example implies universal correctness.

Run identity includes source/configuration/prompt/code/dependency/runtime/image information, not API secrets. Validation cache identity also includes the actual generated response. Successful checkpoints can resume without duplicated committed output. Changing runtime availability or validation code selects fresh work. Execution requests, confirmed Docker test invocations, and tests actually collected are counted separately; mock/unavailable requests are not actual test runs.

### Files, tests, and documentation

Created `repository_data/schemas.py`, `parsing.py`, `discovery.py`, `llm.py`, `sandbox.py`, `validation.py`, and `pipeline.py`; added CLI/configuration, a shared conversation base/gateway extension, tests, and documentation. Pipeline 1's existing tests remain passing. Detailed settings, isolation boundaries, artifact layout, and limitations are in [docs/pipeline2.md](docs/pipeline2.md); the [canonical Pipeline 2 JSON Schema](docs/schemas/pipeline2-training.schema.json) is saved separately.

Actually executed: `uv add` for the three Tree-sitter packages, CLI help, static indexing against both existing repositories, `uv sync --locked`, and pytest. The initial Pipeline 2 test run found one fixture failure caused by the global deduplicator correctly collapsing identical mock answers; the mock instructions were made source-symbol-specific and the test rerun passed. The full suite subsequently recorded **130 passed in 3.06s**. After adding an explicit real-container availability test, it recorded **130 passed, one skipped, five existing SWIG warnings in 3.79s**. The skip is `docker_daemon_unavailable`, not a successful runtime test.

New tests use small synthetic repositories and mocked LLM/sandbox feedback. They cover source/commit/line preservation, all supported language parsers, credential/generated/dependency exclusions, source size/syntax failures, all seven task types, canonical exports, source/review rejection, mocked execution/correction limits with fixed tests, unavailable runtimes, exact duplicates, resume/source changes, restricted Docker arguments/result parsing, CLI, and invalid limits/types. No untrusted code was executed on the host during tests.

### Actual live integration results

Executed a four-unit trial against the existing repositories with `--task-types explanation,architecture,implementation,unit_tests`. The initial 60-second coding request exhausted bounded timeout attempts; its failed report is preserved under `artifacts/pipeline2_integration/attempt1/`. The subsequent trial used the 180-second timeout and shorter requested implementations/tests. It generated **three examples from four selected units**: two knowledge examples accepted, one Python unit-test example quarantined after model review (`failed_tests_meaningful`), and one Kotlin implementation unit unable to produce schema-valid JSON after bounded attempts (`malformed_response`). The latter is an operational unit failure, not a fabricated generated example. The run correctly reported **FAILED** and exit 1 for incomplete mandatory coverage. Its report is `artifacts/pipeline2_integration/report.json`.

That broader trial recorded nine model calls, one unavailable executor request, **zero confirmed runtime test invocations**, and zero runtime-verified accepted records. The reviewer rejection is an observed model decision, not an independent determination that the repository or proposed tests are wrong. No generated-code runtime pass/fail result is available.

Then executed a successful, limited knowledge-only smoke test:

```sh
uv run llmops pip2_code \
  --input ./code_repos \
  --output ./artifacts/pipeline2_smoke \
  --max-tokens 2048 \
  --limit-units 2 \
  --task-types explanation,architecture
```

| Metric | Actual result |
| --- | ---: |
| Repositories indexed / selected | 2 / 2 |
| Source units selected | 2 |
| Generated examples | 2 |
| Accepted examples | 2 |
| Accepted task types | 1 explanation, 1 architecture |
| Source languages in the limited test | Kotlin, TypeScript |
| Rejected examples / failed units | 0 / 0 |
| Runtime-verified accepted | **0** |
| Runtime-unverified knowledge examples | **2** |
| Model calls in first completed invocation | 4 |
| Confirmed generated-code test invocations | **0** |
| Status / CLI exit | **COMPLETED_WITH_WARNINGS / 0** |

The Kotlin example discusses the actual `TaskRepository.refresh` interface method; the TypeScript example discusses the actual `cn` styling utility in `frontend/src/lib/utils.ts`. Source context and evidence accompany both records. This is limited real endpoint verification, not full-corpus generation or validation of executable solutions. Python was included in static indexing and the broader live coding trial; it is not a source language in the final two-unit knowledge subset.

Validated both accepted JSONL lines against the canonical CodeTrainingRecord schema, checked unique IDs, exact source quotes, actual file hashes and inclusive source line slices, stored grounding flags, explicit runtime-unverified/not-applicable metadata, and both published export hashes. All **378 protected resource files** (`.env`, policy documents, and repository files) matched pre-task SHA-256 baselines.

Repeated the same smoke command: two generation and two validation checkpoints were reused, **zero additional model calls or runtime executions** occurred, and the training/rejection export hashes were unchanged. The first completed four-call report is retained under the smoke run's `verification/initial_completed_report.json`; the public report describes the latest resumed invocation and correctly records zero current calls.

The source distribution and wheel also built successfully with `uv build`. Archive inspection confirmed that Pipeline 2 modules are packaged while `.env` and local `code_repos/` inputs are excluded. Added `/code_repos/` to `.gitignore`; repository files remain on disk and unchanged.

The final full pytest run after CLI/documentation work recorded **130 passed, one skipped, five existing SWIG warnings in 3.30s**. The real Docker test remained skipped because its daemon was unavailable. Mocked API/sandbox tests demonstrate orchestration and decision logic, not real runtime correctness or a successful live self-correction trajectory.

Current limited dataset: [training.jsonl](artifacts/pipeline2_smoke/training.jsonl). Current statistics: [report.json](artifacts/pipeline2_smoke/report.json). Broader trial failures: [trial report](artifacts/pipeline2_integration/report.json). A later full-corpus generation attempt was started and interrupted before validation (see the failure analysis below). Real repository builds and model fine-tuning have not been run.

### Research significance and limits

Pipeline 2 turns real source structure and related evidence into auditable software-engineering conversations. Observed test feedback can teach verification and correction when an isolated executor is available. The current environment demonstrates indexing, generation, validation, quarantine, and recovery; it does not establish actual code correctness, full-repository build success, fine-tuning benefit, or reliable automatic bug detection.

Remaining limits include Kotlin grammar diagnostics, syntax without typechecking, bounded heuristic dependency/test context, missing Git commits in archives, conservative size/secret/generated-file exclusions, exact rather than semantic deduplication, correlated model review, generated-test oracle/coverage weaknesses, provider timeouts/malformed JSON, no configured Kotlin/TypeScript executor, and no running Docker daemon. The container design also depends on maintained trusted images/runtime and is not a proof against all kernel exploits. Default coding-call timeout is now 180 seconds after a shorter live request timed out; retries/corrections remain bounded. Future Linux execution uses the same configuration/container boundary; GPU training is outside this pipeline.

## 9. Execution log

The following run block is maintained automatically by Pipeline 1. Earlier run logs are preserved in the archived history.

<!-- pipeline1-run:d0e71e851cf17bb6ccd17f1555d4c74680347675d316cef41a6d41317ab06251 -->
## Pipeline 1 automated execution — 2026-10-25T22:21:36.694598+04:00

- Run ID: `d0e71e851cf17bb6ccd17f1555d4c74680347675d316cef41a6d41317ab06251`; status: **COMPLETED_WITH_WARNINGS**.
- LangGraph stages: parse → normalize/chunk → generate → validate → export.
- Documents processed: 1; chunks: 14.
- Document limit: 1; excluded by limit: 9.
- Generated examples: 14; accepted: 14; rejected: 0.
- Failed chunks: 0; LLM calls this invocation: 28.
- Validation: Pydantic schema, source identity, complete extractive passage, model-assisted grounding/conditions/scope, global exact question/answer deduplication.
- Accepted means these automated gates passed; semantic/legal correctness and downstream fine-tuning performance remain unverified.
- Output report: `/Users/mas/Development/DissertationProject/artifacts/pipeline1_test/report.json`.
- Errors: [].

No Pipeline 2 or model fine-tuning was performed.
<!-- /pipeline1-run:d0e71e851cf17bb6ccd17f1555d4c74680347675d316cef41a6d41317ab06251 -->

<!-- pipeline2-run:49aaee9295e7566b7c2b4bf6f24ef8da485bfe585714432d2820b2da96d50777 -->
## Pipeline 2 execution — 2026-10-25T22:55:33.353552+04:00

- Status: **FAILED**; repositories indexed: 2; selected units: 4.
- Generated examples: 2; accepted: 2; rejected/quarantined: 0.
- Accepted runtime-verified: 0; accepted nonexecuting knowledge tasks: 2.
- Rejected/quarantined verification statuses: {}.
- Source languages: {"kotlin": 2, "typescript": 1, "python": 1}; generated task types: {"explanation": 1, "architecture": 1}.
- Sandbox: docker_daemon_unavailable; real test invocations this run: 0.
- LLM calls this run: 7; resumed generation/validation units: 0/0.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline2_integration/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: ["mandatory_unit_processing_incomplete"].

No Pipeline 3 or model fine-tuning was performed.
<!-- /pipeline2-run:49aaee9295e7566b7c2b4bf6f24ef8da485bfe585714432d2820b2da96d50777 -->

<!-- pipeline2-run:d12d115223964fa949d8b8b709931dd7800ac47aa7520254f00482aee2d370ca -->
## Pipeline 2 execution — 2026-10-25T23:03:04.175311+04:00

- Status: **FAILED**; repositories indexed: 2; selected units: 4.
- Generated examples: 3; accepted: 2; rejected/quarantined: 1.
- Accepted runtime-verified: 0; accepted nonexecuting knowledge tasks: 2.
- Rejected/quarantined verification statuses: {"unverified": 1}.
- Source languages: {"kotlin": 2, "typescript": 1, "python": 1}; generated task types: {"explanation": 1, "architecture": 1, "unit_tests": 1}.
- Sandbox: docker_daemon_unavailable; runtime test invocations confirmed this run: 0.
- LLM calls this run: 9; resumed generation/validation units: 0/0.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline2_integration/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: ["mandatory_unit_processing_incomplete"].

No Pipeline 3 or model fine-tuning was performed.
<!-- /pipeline2-run:d12d115223964fa949d8b8b709931dd7800ac47aa7520254f00482aee2d370ca -->

<!-- pipeline2-run:313a612b8092f6edb3fd6491a7311eeec2411e9e2b6d6c272af543bcfde6b48b -->
## Pipeline 2 execution — 2026-10-25T23:08:23.711859+04:00

- Status: **COMPLETED_WITH_WARNINGS**; repositories indexed: 2; selected units: 2.
- Generated examples: 2; accepted: 2; rejected/quarantined: 0.
- Accepted runtime-verified: 0; accepted nonexecuting knowledge tasks: 2.
- Rejected/quarantined verification statuses: {}.
- Source languages: {"kotlin": 1, "typescript": 1}; generated task types: {"explanation": 1, "architecture": 1}.
- Sandbox: docker_daemon_unavailable; runtime test invocations confirmed this run: 0.
- LLM calls this run: 0; resumed generation/validation units: 2/2.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline2_smoke/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: [].

No Pipeline 3 or model fine-tuning was performed.
<!-- /pipeline2-run:313a612b8092f6edb3fd6491a7311eeec2411e9e2b6d6c272af543bcfde6b48b -->

<!-- pipeline2-run:95712ad18fc291861b272c756b700ba020232be3e4de39aadf064cb0773ba4a3 -->
## Pipeline 2 execution — 2026-10-26T07:02:43.813121+04:00

- Status: **FAILED**; repositories indexed: 2; selected units: 459.
- Generated examples: 2; accepted: 0; rejected/quarantined: 2.
- Accepted runtime-verified: 0; accepted nonexecuting knowledge tasks: 0.
- Rejected/quarantined verification statuses: {"unverified": 2}.
- Source languages: {"kotlin": 159, "typescript": 219, "python": 81}; generated task types: {"explanation": 1, "unit_tests": 1}.
- Sandbox: docker_daemon_unavailable; runtime test invocations confirmed this run: 0.
- LLM calls this run: 11; resumed generation/validation units: 0/0.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline2/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: ["pipeline_stage_failed:KeyboardInterrupt", "mandatory_unit_processing_incomplete", "no_accepted_training_examples"].

No Pipeline 3 or model fine-tuning was performed.
<!-- /pipeline2-run:95712ad18fc291861b272c756b700ba020232be3e4de39aadf064cb0773ba4a3 -->

## Pipeline 2 failure analysis — 2026-10-26

Inspected the existing public reports, rejection records, and orchestration code; no new LLM requests or generated-code execution were performed for this diagnosis. The main `artifacts/pipeline2/report.json` records **FAILED**, 459 selected source units, 18 model calls, four generated candidates, and zero accepted examples. Its rejection records distinguish four `malformed_response` unit failures, 451 `generation_incomplete` units, and four generated examples marked `validation_incomplete`. Thus `failed_units: 459` combines actual generation errors with unfinished work; it does not mean 459 completed attempts each failed validation.

The saved error is `pipeline_stage_failed:KeyboardInterrupt`: execution was interrupted. The current graph generates all selected units before entering validation. Consequently, the four generated examples had no validation evidence at interruption and were quarantined. This sequencing delays usable results and makes large trial runs difficult to assess.

The gateway groups JSON parsing/schema failures into `malformed_response`, without saving a safe validation-error category or provider finish reason. These artifacts cannot establish whether the four malformed responses resulted from truncation, invalid JSON, or schema mismatches. That cause remains unresolved and must not be reported as confirmed. Earlier integration attempts also recorded endpoint timeout and a model-review rejection for weak tests.

The runtime capability report is `docker_daemon_unavailable`; zero isolated tests ran. Kotlin and TypeScript executors are also not configured. Enabling Docker alone therefore cannot establish executable validation for every indexed language.

Priority corrective work: (1) add credential-safe response diagnostics and validate endpoint behavior on a small sample; (2) process and checkpoint generation/validation per unit so useful validated progress survives interruption; (3) distinguish failed, pending, and interrupted units in statistics; (4) provide and verify the required isolated language runtimes before accepting executable examples; (5) rerun a bounded mixed-task integration test before a full run. Do not weaken acceptance checks to manufacture success.

The two accepted records in `artifacts/pipeline2_smoke` demonstrate only the limited explanation/architecture path. They are not evidence that the requested complete executable training-data workflow succeeds. No new fixes or test passes are claimed by this diagnostic entry.

<!-- pipeline2-run:98ec37299a60ccc308b65ff2131fc6be12c9eade92b41e61eda554d05ed93037 -->
## Pipeline 2 execution — 2026-10-26T07:07:10.768526+04:00

- Status: **FAILED**; repositories indexed: 2; selected units: 2.
- Generated examples: 1; accepted: 1; rejected/quarantined: 0.
- Accepted runtime-verified: 0; accepted nonexecuting knowledge tasks: 1.
- Rejected/quarantined verification statuses: {}.
- Source languages: {"kotlin": 1, "typescript": 1}; generated task types: {"explanation": 1}.
- Sandbox: docker_daemon_unavailable; runtime test invocations confirmed this run: 0.
- LLM calls this run: 5; resumed generation/validation units: 0/0.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline2/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: ["mandatory_unit_processing_incomplete"].

No Pipeline 3 or model fine-tuning was performed.
<!-- /pipeline2-run:98ec37299a60ccc308b65ff2131fc6be12c9eade92b41e61eda554d05ed93037 -->

<!-- pipeline2-run:4ad8f113fb38e8f45e3c39da7bf1008847175adc4cb39af174f7fb0bae59dcf2 -->
## Pipeline 2 execution — 2026-10-26T07:23:27.571195+04:00

- Status: **FAILED**; repositories indexed: 2; selected units: 2.
- Generated examples: 1; accepted: 1; rejected/quarantined: 0.
- Accepted runtime-verified: 0; accepted nonexecuting knowledge tasks: 1.
- Rejected/quarantined verification statuses: {}.
- Source languages: {"kotlin": 1, "typescript": 1}; generated task types: {"explanation": 1}.
- Sandbox: docker_daemon_unavailable; runtime test invocations confirmed this run: 0.
- LLM calls this run: 5; resumed generation/validation units: 0/0.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline2_preflight/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: ["mandatory_unit_processing_incomplete"].

No Pipeline 3 or model fine-tuning was performed.
<!-- /pipeline2-run:4ad8f113fb38e8f45e3c39da7bf1008847175adc4cb39af174f7fb0bae59dcf2 -->

<!-- pipeline2-run:5e9c01eb0ccbe6747b133697fcd772f00ee057edae96df9bc6964e665d3ecf18 -->
## Pipeline 2 execution — 2026-10-26T07:27:31.853076+04:00

- Status: **COMPLETED_WITH_WARNINGS**; repositories indexed: 2; selected units: 2.
- Generated examples: 2; accepted: 1; rejected/quarantined: 1.
- Accepted runtime-verified: 0; accepted nonexecuting knowledge tasks: 1.
- Rejected/quarantined verification statuses: {"unverified": 1}.
- Source languages: {"kotlin": 1, "typescript": 1}; generated task types: {"implementation": 1, "explanation": 1}.
- Sandbox: docker_daemon_unavailable; runtime test invocations confirmed this run: 0.
- LLM calls this run: 4; resumed generation/validation units: 0/0.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline2_preflight_expanded/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: [].

No Pipeline 3 or model fine-tuning was performed.
<!-- /pipeline2-run:5e9c01eb0ccbe6747b133697fcd772f00ee057edae96df9bc6964e665d3ecf18 -->

## Local validation before server transfer — 2026-10-26

The objective was to validate the implementation locally before server execution, not to begin fine-tuning. Existing failure reports were preserved; new live trials used separate output directories.

### Fixes and verification evidence

- Shared `policy_transformation/generation.py`: bounded model-call diagnostics now distinguish provider truncation (`finish_reason=length`) from schema/JSON errors, validate nested code candidates, provide safe category-only retry feedback, and show retry progress. Diagnostics exclude raw responses, input values, exception bodies, and credentials.
- `repository_data/pipeline.py`: each successfully generated candidate is validated and checkpointed before the next source unit is generated. Interruption preserves completed validation. Failed and pending work are counted separately; provider circuit-breaker skips are pending rather than completed failures. Shared gateway source changes now participate in cache identity.
- `cli.py`: prints failed/pending units and failure/rejection reasons. `config.py`: Pipeline 2 response limit increased from 4096 to 8192, independent of the unchanged source chunk limit. `.env` was not modified.
- `tests/test_pipeline2.py`: added regression coverage for truncated responses, safe schema diagnostics, nested-schema retries, interrupted-run recovery without duplicates, and pending circuit-breaker work. Documentation updated in README, `docs/pipeline2.md`, and new `docs/server-validation.md`.

Executed `uv sync --locked`, `uv run pytest -q`, `uv build`, `uv run llmops pip2_code --help`, and `uv run python -m compileall -q src`. Final full-suite result: **135 passed, one skipped, five existing SWIG deprecation warnings in 3.66 seconds**. The skipped real Docker test reports `docker_daemon_unavailable`; mocked runtime results are not actual execution evidence. The wheel includes eight Pipeline 2 modules and excludes credentials and local input repositories/policies. SHA-256 comparison confirmed **378 protected files unchanged**, including the existing credentials/policy/repository inputs.

During development, one temporary indentation error caused test collection to fail and was repaired. A new diagnostics test initially used the actual synthetic secret, correctly triggering the credential guard rather than the intended schema path; that fixture was corrected. Those failures are not represented as passing executions.

### Actual endpoint trials

Endpoint model-list connectivity succeeded, and a bounded minimal chat request returned successfully. The live two-unit trial in `artifacts/pipeline2_preflight` reproduced the failure with the original 4096-token response budget. All three implementation-generation attempts returned the provider finish reason `length`: **response truncation is confirmed for this trial**, rather than inferred from generic malformed JSON. That run accepted one explanation but failed one generation unit. This finding does not retrospectively prove the cause of every historical malformed response.

Repeated the live two-unit workflow with `PIPELINE2__RESPONSE_MAX_TOKENS=8192 PIPELINE2__MAX_ATTEMPTS=1` and output `artifacts/pipeline2_preflight_expanded`:

```sh
PIPELINE2__RESPONSE_MAX_TOKENS=8192 PIPELINE2__MAX_ATTEMPTS=1 uv run llmops pip2_code \
  --input ./code_repos \
  --output ./artifacts/pipeline2_preflight_expanded \
  --max-tokens 2048 \
  --limit-units 2
```

Verified results: **two selected units, two generated examples, one accepted explanation, one quarantined Kotlin implementation, zero failed units, zero pending units; COMPLETED_WITH_WARNINGS, exit 0**. The implementation and test snippets passed schema and syntax checks and the model review, but the example remained quarantined with `execution_unverified:language_runtime_not_configured`. No actual code tests ran. The accepted record passed conversational-schema, current source-file hash, exact source-quote, and recorded review-flag checks; public export hashes matched. These checks establish traceability and observed automated validation, not semantic certification.

The larger response budget resolved truncation for this observed coding request; it cannot guarantee all future responses fit or are correct. Full-corpus generation and full runtime readiness remain unverified. The final code changes pass the full automated suite; the live trial used one configured attempt and the immediate-validation workflow. Subsequent small changes to reporting/cache identity/default configuration were verified by the automated suite rather than another paid live rerun.

### Server gates and current limits

Local installation, packaging, CLI, regression, source preservation, endpoint connectivity, mixed-task generation, immediate validation, and quarantine behavior have evidence. This is not a claim that 100% of a mixed-language executable dataset is ready. Python requires a running Docker daemon and prepared local image; an actual isolated Python test plus limited real coding integration must succeed on the server. Kotlin/TypeScript execution backends are still unimplemented and need implementation and real tests before claiming complete mixed-language runtime support. Moving machines alone does not remove that code limitation. The server acceptance procedure is documented in `docs/server-validation.md`.

Machine-readable local evidence: `artifacts/pipeline2_preflight/local_validation.json`. Existing `.env`, source policies, and repositories remain unchanged. No Pipeline 3 or model fine-tuning was implemented.

<!-- pipeline3-run:f91965033a4795de399fd54e200bccde11b54a07cd2410b287315c4a99b0e74b -->
## Pipeline 3 execution — 2026-10-26T07:56:48.746836+04:00

- Status: **COMPLETED_WITH_WARNINGS**; accepted target/cap: 6.
- Unique records processed: 6; accepted: 6; rejected: 0.
- Decoded rows read this invocation (including prefetch/replay): 12; resumed decisions: 0.
- Rejection reasons: {}; accepted-record transformations: {"strip_leading_think": 5}.
- Source statistics: [{"alias": "nemotron", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data", "config": "chat", "split": "train", "revision": "9f36020daf067f1a8b39336bf619fe30af30bb02", "quota": 3, "scan_limit": 30, "processed": 3, "accepted": 3, "rejected": 0, "stop_reason": "target_met"}, {"alias": "opencode", "dataset": "nvidia/OpenCodeInstruct", "config": "train", "split": "train", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "quota": 3, "scan_limit": 30, "processed": 3, "accepted": 3, "rejected": 0, "stop_reason": "target_met"}].
- Seed: 42; bounded buffer: 4. Revisions and card license snapshots are in report.json and the internal manifest.
- No LLM calls, local code tests, model downloads, dataset mixing, or fine-tuning. External test metadata is not local verification. Network byte totals are not measured.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline3_integration/report.json`; errors: [].
<!-- /pipeline3-run:f91965033a4795de399fd54e200bccde11b54a07cd2410b287315c4a99b0e74b -->

<!-- pipeline3-run:556ae84eb6dfbbeea0ee753230f0e59b6282c6c4fa1cf4c3d7e4a1891b55093a -->
## Pipeline 3 execution — 2026-10-26T08:04:32.769142+04:00

- Status: **COMPLETED_WITH_WARNINGS**; accepted target/cap: 6.
- Unique records processed: 6; accepted: 6; rejected: 0.
- Decoded rows read this invocation (including prefetch/replay): 0; resumed decisions: 6.
- Rejection reasons: {}; accepted-record transformations: {"strip_leading_think": 5}.
- Source statistics: [{"alias": "nemotron", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data", "config": "chat", "split": "train", "revision": "9f36020daf067f1a8b39336bf619fe30af30bb02", "quota": 3, "scan_limit": 30, "processed": 3, "accepted": 3, "rejected": 0, "stop_reason": "target_met"}, {"alias": "opencode", "dataset": "nvidia/OpenCodeInstruct", "config": "train", "split": "train", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "quota": 3, "scan_limit": 30, "processed": 3, "accepted": 3, "rejected": 0, "stop_reason": "target_met"}].
- Seed: 42; bounded buffer: 4. Revisions and card license snapshots are in report.json and the internal manifest.
- No LLM calls, local code tests, model downloads, dataset mixing, or fine-tuning. External test metadata is not local verification. Network byte totals are not measured.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline3_integration/report.json`; errors: [].
<!-- /pipeline3-run:556ae84eb6dfbbeea0ee753230f0e59b6282c6c4fa1cf4c3d7e4a1891b55093a -->

## Pipeline 3 — External Dataset Ingestion & Preparation (26 October 2026)

### Objective and implemented flow

Implement a single, reproducible external-data workflow for Qwen3-14B preparation, using the existing `uv`/src/Typer/Pydantic architecture. LangGraph orchestrates metadata discovery → ingestion → dataset-specific normalization → quality filtering / exact deduplication → export → statistics/research reporting. The ingestion/normalization/filtering nodes loop per record and commit durable decisions before proceeding. State stores only stage names and artifact paths. No LLM, Docker, GPU, model weights, or tokenizer is required or downloaded. Pipeline 4 mixing, final training splits, and fine-tuning are outside scope.

```sh
uv run llmops pip3_datasets \
  --output ./artifacts/pipeline3 \
  --max-samples 10000
```

`--max-samples` is a total accepted-record upper bound across both datasets. Default allocation is equal; repeated `--allocation ALIAS=count` overrides quotas and unspecified datasets share the remainder. Selection, configurations, splits, revisions, seed, strict JSON filter configuration, and metadata refresh are exposed through CLI options. Default configurations are Nemotron `chat` and OpenCodeInstruct `train`; other inspected Nemotron configurations are selectable without claiming default full-domain coverage.

### Dataset inspection, schemas, and licenses

- [NVIDIA Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data): resolved revision `9f36020daf067f1a8b39336bf619fe30af30bb02`. The card declares eight configurations (math, science, chat, instruction_following, safety, conversational_agent, swe, terminal_agent), each with `train` data. Metadata and bounded first-record previews of the published JSONL files were inspected. Actual records have `domain`, `source`, `messages`, and `generator`; conversations vary in length, can have blank system messages, and can contain marked reasoning. The card declares `license: other`, named `nvidia-open-model-license`, linking the [NVIDIA agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Its use conditions, Trustworthy AI terms, redistribution notices, and separate-component terms need to be retained/reviewed for downstream use; this declaration is not reclassified as CC BY.
- [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct): resolved revision `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`. One `train` configuration/split; fields include `id`, `input`, `output`, `domain`, `generation_algorithm`, `llm_judgement`, `unit_tests`, `tests_execution_status`, and `average_test_score`. The published card describes five million records; that is a publisher total, not a downloaded/validated count. The card declares score as string while actual streamed rows supplied floats, which are preserved. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), including attribution, license-link, and change-indication obligations. Source title/NVIDIA attribution, revision, card snapshot, license URL, and transformation history are retained.

Public Hub metadata and source-card snapshots are inspected before streaming. Original tests and reported execution results remain metadata with `locally_executed=false` and `external_test_results_status=not_locally_verified`; no external pass is presented as a local pass. Declared license inspection is not a certification that every downstream use or redistribution is legally suitable. The pinned card and linked terms are documented separately because external agreements may change.

The existing canonical envelope (`schema_version`, `example_id`, `messages`, `metadata`) is shared through `ConversationArtifact`. Pipeline 1/2 retain their strict three-message specialization. Pipeline 3's `ExternalTrainingRecord` supports original optional/blank system messages, multi-turn user/assistant dialogue, and native tool calls/results with matching IDs and validated arguments. Original order and content are preserved except explicitly recorded reasoning removal. No synthetic system prompt, answer, tool result, or test outcome is inserted. Source metadata carries dataset/config/split/SHA, upstream ID or pinned-stream ordinal, raw-content hash, license/source URL, original auxiliary fields, indexed message extras, transformation hashes, and budget estimates. The full [Pydantic JSON Schema](docs/schemas/pipeline3-training.schema.json) is saved.

### Sampling, filtering, and reliability

Hugging Face Datasets packaged JSON/Parquet readers stream URLs pinned to inspected SHAs; remote repository scripts are not executed. JSON's implicit agent-trace transformation is explicitly disabled. A fixed seed and bounded prefix buffer provide reproducibility within the chosen configuration. This is not a uniform whole-corpus sample. Default buffer is 16; default candidate bound is ten times each source's accepted quota. Oversized records/conversations are rejected intact rather than truncated. The default 32768 budget counts UTF-8 bytes in serialized messages, a conservative estimate excluding future Qwen template/tool-definition overhead; exact final tokenizer limits must be rechecked later.

Default `strip_marked` removes only complete leading assistant `<think>…</think>` blocks and explicit reasoning fields, recording hashes and locations. Configurable `reject` excludes those records instead. Ambiguous, nested, unclosed, or empty-final-answer cases are rejected. Incompatible template tokens and configured additional reasoning markers are rejected; ordinary text/code is not broadly rewritten. Empty/malformed/nontext/unsupported-role records, invalid tool protocols, and exact duplicate conversations are filtered. Whitespace inside code is preserved. These are structural/filtering checks, not semantic answer verification.

SQLite stores decisions, a disk-backed unique conversation index, and transactionally updated progress counters. Network retries and metadata retries are bounded. A failed source does not prevent accessible sources from producing partial artifacts, but mandatory access failure returns `FAILED` and a nonzero CLI status. Completed runs reuse decisions without streaming; interrupted runs deterministically replay and verify committed raw hashes before continuing. This can reread source data. Source drift fails rather than silently mixing revisions. Exports are atomic per file and include final hashes; consumers must check the final report. Rejected records retain bounded provenance/reasons rather than full unwanted reasoning or large text.

### Source files and technology choices

Created `external_datasets/{__init__,schemas,sources,sampling,normalization,storage,pipeline}.py` and `tests/test_pipeline3.py`. Modified `cli.py`, `config.py`, and the shared `data_schemas.py`; updated dependencies/lock, README, and report. Added [docs/pipeline3.md](docs/pipeline3.md) and the schema file. Hugging Face Datasets **4.8.5** supplies streaming JSON/Parquet readers; huggingface-hub **1.33.0** supplies metadata, pinned URLs, authentication, and small card retrieval; PyArrow **26.0.0** is the installed Datasets dependency for Parquet/JSON batches. LangGraph **1.2.17** supplies the resumable processing loop; Pydantic validates configuration, records, and tool structures; standard-library SQLite provides bounded-memory transactional state/deduplication. No LangChain transformation is used because deterministic adapters suffice.

### Actual verification results

Executed `uv add 'datasets>=4,<5' 'huggingface-hub>=1,<2'`, `uv sync --locked`, CLI help, automated pytest checks, `uv build`, and source compilation checks. The initial dedicated test run recorded **33 passed**; later expanded runs passed as well. Final regression run: **176 passed, one skipped, five existing SWIG warnings in 6.74 seconds**; all **41 Pipeline 3 tests passed**. A final regression also rejects non-finite numeric metadata (such as NaN) and nonstandard tool-argument JSON so exports remain strict JSON. The skip is Pipeline 2's real Docker test (`docker_daemon_unavailable`), unrelated to Pipeline 3. Mocked streams test malformed/empty/oversized input, both adapters, multi-turn/tool preservation, reasoning/token configuration, source IDs, allocation caps, exact duplicates, deterministic sampling, exhaustion, access/midstream errors, checkpoint replay/drift detection, recovery, export failure, cleanup, and CLI behavior. Mocked tests are not counted as real ingestion evidence.

Ran limited real ingestion against both actual NVIDIA sources:

```sh
PIPELINE3__SHUFFLE_BUFFER=4 uv run llmops pip3_datasets \
  --output ./artifacts/pipeline3_integration \
  --max-samples 6
```

| Metric | Verified result for the final fresh invocation |
| --- | ---: |
| Nemotron chat accepted / examined | 3 / 3 |
| OpenCodeInstruct train accepted / examined | 3 / 3 |
| Total accepted / rejected | 6 / 0 |
| Decoded source rows read, including sampler prefetch | 12 |
| Decoded serialized source bytes read | 64493 |
| Recorded leading reasoning-block removals | 5 |
| LLM calls / locally executed code tests | 0 / 0 |
| Status / CLI exit | COMPLETED_WITH_WARNINGS / 0 |

`COMPLETED_WITH_WARNINGS` reflects the recorded normalization actions; no semantic correctness claim is attached. Earlier implementation-stage six-record trials used separate run fingerprints retained internally; this table describes the final fresh run, not a sum across all HTTP inspection/development requests. Actual network bytes and library read-ahead are not instrumented. The 12-row metric is decoded rows delivered to the sampler, including buffered rows, rather than a claim that only 12 rows crossed the network.

Repeated the identical final command: **six committed decisions reused, zero additional stream rows read, unchanged training/rejection hashes, and the same six accepted records**. The latest public report correctly describes the resumed invocation's zero reads. The first completed report for the final code is retained under `_runs/cdf9327d3f853d4f09e819d45974e54958928ad098c4eb8b219ec02894843a32/verification/initial_ingestion_report.json`.

Independently read eight additional source rows at the pinned revisions to retrieve all six selected originals, re-applied the deterministic adapter, and compared the complete exported records. All six matched, schemas validated, and export hashes matched. One Nemotron record (`row:3`) retained seven messages across three user/assistant turns and an original system message. Three OpenCodeInstruct records retained their original unit tests, reported statuses, and scores without executing them. The provenance/resume check is saved next to the initial report as `verification/source_and_resume_checks.json`. Hash checks confirmed **378 protected input files unchanged**, including `.env`, synthetic policies, and repository inputs. Wheel inspection confirmed all seven Pipeline 3 modules are packaged and credentials/input/artifact directories are excluded.

Final sample artifacts: [training.jsonl](artifacts/pipeline3_integration/training.jsonl), [rejected.jsonl](artifacts/pipeline3_integration/rejected.jsonl), [report.json](artifacts/pipeline3_integration/report.json).

### Limitations and research relevance

Only the six-record final sample and bounded schema previews were inspected; all-domain/full-corpus quality, global representativeness, factual/code correctness, benchmark leakage, license suitability, and fine-tuning gains remain unverified. Buffered prefix sampling is biased toward early records/files. Exact deduplication misses paraphrases and cross-pipeline duplicates, which await Pipeline 4. Reasoning removal relies on explicit markers and cannot recognize every unmarked trace. Datasets/Arrow/fsspec can read ahead and decode a large record before size checks; actual network transfer and absolute peak-memory bounds are not claimed. Replay after interruption can incur rereads. Final Qwen tokenization, tool-definition injection, loss masking, mixing, and train/evaluation splits remain future work.

This stage contributes a reproducible, source-attributed external-data stream alongside corporate policies and repository examples. Separation of upstream claims from observed local checks supports auditable dissertation experiments without inventing evidence. No Pipeline 4 or model fine-tuning was implemented.

<!-- pipeline3-run:cdf9327d3f853d4f09e819d45974e54958928ad098c4eb8b219ec02894843a32 -->
## Pipeline 3 execution — 2026-10-26T12:08:39.566624+04:00

- Status: **COMPLETED_WITH_WARNINGS**; accepted target/cap: 6.
- Unique records processed: 6; accepted: 6; rejected: 0.
- Decoded rows read this invocation (including prefetch/replay): 0; resumed decisions: 6.
- Rejection reasons: {}; accepted-record transformations: {"strip_leading_think": 5}.
- Source statistics: [{"alias": "nemotron", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data", "config": "chat", "split": "train", "revision": "9f36020daf067f1a8b39336bf619fe30af30bb02", "quota": 3, "scan_limit": 30, "processed": 3, "accepted": 3, "rejected": 0, "stop_reason": "target_met"}, {"alias": "opencode", "dataset": "nvidia/OpenCodeInstruct", "config": "train", "split": "train", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "quota": 3, "scan_limit": 30, "processed": 3, "accepted": 3, "rejected": 0, "stop_reason": "target_met"}].
- Seed: 42; bounded buffer: 4. Revisions and card license snapshots are in report.json and the internal manifest.
- No LLM calls, local code tests, model downloads, dataset mixing, or fine-tuning. External test metadata is not local verification. Network byte totals are not measured.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline3_integration/report.json`; errors: [].
<!-- /pipeline3-run:cdf9327d3f853d4f09e819d45974e54958928ad098c4eb8b219ec02894843a32 -->

<!-- pipeline3-run:fecc9c2bcd48f7f73df3476f31571acec329a1de713bace1456eca4afcc780c9 -->
## Pipeline 3 execution — 2026-10-26T12:57:13.537946+04:00

- Status: **COMPLETED_WITH_WARNINGS**; accepted target/cap: 1000.
- Unique records processed: 1040; accepted: 1000; rejected: 40.
- Decoded rows read this invocation (including prefetch/replay): 1070; resumed decisions: 0.
- Rejection reasons: {"oversized_conversation": 40}; accepted-record transformations: {"strip_leading_think": 780}.
- Source statistics: [{"alias": "nemotron", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data", "config": "chat", "split": "train", "revision": "9f36020daf067f1a8b39336bf619fe30af30bb02", "quota": 500, "scan_limit": 5000, "processed": 540, "accepted": 500, "rejected": 40, "stop_reason": "target_met"}, {"alias": "opencode", "dataset": "nvidia/OpenCodeInstruct", "config": "train", "split": "train", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "quota": 500, "scan_limit": 5000, "processed": 500, "accepted": 500, "rejected": 0, "stop_reason": "target_met"}].
- Seed: 42; bounded buffer: 16. Revisions and card license snapshots are in report.json and the internal manifest.
- No LLM calls, local code tests, model downloads, dataset mixing, or fine-tuning. External test metadata is not local verification. Network byte totals are not measured.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline3/report.json`; errors: [].
<!-- /pipeline3-run:fecc9c2bcd48f7f73df3476f31571acec329a1de713bace1456eca4afcc780c9 -->

<!-- pipeline4-run:c89df026be14702b7cf698a9036d3c03d258983fdde7f2aed98b5ab3adef3c2a -->
## Pipeline 4 execution — 2026-10-26T14:33:56.648348+04:00

- Status: **FAILED**; preparation training-ready: **False** (not semantic certification).
- Input records: 0; structurally/token valid: 0; rejected invalid: 0.
- Duplicates removed: 0; excluded by mixing: 0; exported: 0.
- Split counts: {"train": 0, "validation": 0, "test": 0}; distribution: {}.
- Tokenizer: {"validated": false}; selected token statistics: {}.
- Source issues: ["pipeline1:missing_training_jsonl", "pipeline2:upstream_run_failed_or_incomplete"]; errors: ["pipeline1:missing_training_jsonl", "pipeline2:upstream_run_failed_or_incomplete", "Mandatory upstream datasets are missing, failed, or inconsistent; inspect input_issues.", "no_exported_examples"].
- Resumed stages: []; no model training, LLM calls, or reexecution of source tests.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline4/report.json`. Held-out tests are excluded from training and hyperparameter-selection inputs.
<!-- /pipeline4-run:c89df026be14702b7cf698a9036d3c03d258983fdde7f2aed98b5ab3adef3c2a -->

<!-- pipeline4-run:04b10ce00eb860ca381e8f2c2092fc0b6e740f5359c677a9e967a073eeb48f1c -->
## Pipeline 4 execution — 2026-10-26T13:34:49.306031+04:00

- Status: **COMPLETED_WITH_WARNINGS**; preparation training-ready: **False** (not semantic certification).
- Input records: 1001; structurally/token valid: 923; rejected invalid: 78.
- Duplicates removed: 0; excluded by mixing: 823; exported: 100.
- Split counts: {"train": 90, "validation": 5, "test": 5}; distribution: {"pipeline": {"pipeline2": 1, "pipeline3": 99}, "domain": {"coding": 57, "general_dialogue": 43}, "source": {"nvidia/Nemotron-Cascade-2-SFT-Data": 43, "nvidia/OpenCodeInstruct": 56, "pipeline2": 1}, "status": {"externally_reported": 56, "not_applicable": 1, "unverified": 43}}.
- Tokenizer: {"model": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "files": {"tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101", "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", "config.json": "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a"}, "validated": true, "chat_template_hash": "41d5929bf73796beb66809ac700b2cf3ff81694f933e5c14d52b7fd6963c947d", "add_generation_prompt": false, "enable_thinking": false, "tokenization": "apply_chat_template(tokenize=True); no truncation"}; selected token statistics: {"min": 72, "max": 3929, "mean": 708.16, "total": 70816}.
- Source issues: ["pipeline1:missing_training_jsonl", "pipeline2:upstream_run_failed_or_incomplete"]; errors: [].
- Resumed stages: []; no model training, LLM calls, or reexecution of source tests.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline4_partial_validation/report.json`. Held-out tests are excluded from training and hyperparameter-selection inputs.
<!-- /pipeline4-run:04b10ce00eb860ca381e8f2c2092fc0b6e740f5359c677a9e967a073eeb48f1c -->

<!-- pipeline4-run:14c34aec92300ddabddb11f0fbf076b59b7074509f0e4796fa378700d21c8508 -->
## Pipeline 4 execution — 2026-10-26T13:52:38.988269+04:00

- Status: **COMPLETED_WITH_WARNINGS**; preparation training-ready: **False** (not semantic certification).
- Input records: 1001; structurally/token valid: 923; rejected invalid: 78.
- Duplicates removed: 0; excluded by mixing: 823; exported: 100.
- Split counts: {"train": 90, "validation": 5, "test": 5}; distribution: {"pipeline": {"pipeline2": 1, "pipeline3": 99}, "domain": {"coding": 55, "general_dialogue": 45}, "source": {"nvidia/Nemotron-Cascade-2-SFT-Data": 45, "nvidia/OpenCodeInstruct": 54, "pipeline2": 1}, "status": {"externally_reported": 54, "not_applicable": 1, "unverified": 45}}.
- Tokenizer: {"model": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "files": {"tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101", "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", "config.json": "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a"}, "validated": true, "max_context_tokens": 40960, "chat_template_hash": "41d5929bf73796beb66809ac700b2cf3ff81694f933e5c14d52b7fd6963c947d", "add_generation_prompt": false, "enable_thinking": false, "tokenization": "apply_chat_template(tokenize=True); no truncation"}; selected token statistics: {"min": 72, "max": 3929, "mean": 708.93, "total": 70893}.
- Source issues: ["pipeline1:missing_training_jsonl", "pipeline2:upstream_run_failed_or_incomplete"]; errors: [].
- Resumed stages: ["load", "deduplication"]; no model training, LLM calls, or reexecution of source tests.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline4_partial_validation/report.json`. Held-out tests are excluded from training and hyperparameter-selection inputs.
<!-- /pipeline4-run:14c34aec92300ddabddb11f0fbf076b59b7074509f0e4796fa378700d21c8508 -->

<!-- pipeline4-run:b6997189d36a7d0247296f944688392e1321b462e71361e726e38cc5e96f958c -->
## Pipeline 4 execution — 2026-10-26T13:58:01.468161+04:00

- Status: **COMPLETED_WITH_WARNINGS**; preparation training-ready: **False** (not semantic certification).
- Input records: 1001; structurally/token valid: 923; rejected invalid: 78.
- Duplicates removed: 0; excluded by mixing: 823; exported: 100.
- Split counts: {"train": 90, "validation": 5, "test": 5}; distribution: {"pipeline": {"pipeline2": 1, "pipeline3": 99}, "domain": {"coding": 55, "general_dialogue": 45}, "source": {"nvidia/Nemotron-Cascade-2-SFT-Data": 45, "nvidia/OpenCodeInstruct": 54, "pipeline2": 1}, "status": {"externally_reported": 54, "not_applicable": 1, "unverified": 45}}.
- Tokenizer: {"model": "Qwen/Qwen3-14B", "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18", "files": {"tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101", "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910", "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5", "config.json": "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a"}, "validated": true, "max_context_tokens": 40960, "chat_template_hash": "41d5929bf73796beb66809ac700b2cf3ff81694f933e5c14d52b7fd6963c947d", "add_generation_prompt": false, "enable_thinking": false, "tokenization": "apply_chat_template(tokenize=True); no truncation"}; selected token statistics: {"min": 72, "max": 3929, "mean": 708.93, "total": 70893}.
- Source issues: ["pipeline1:missing_training_jsonl", "pipeline2:upstream_run_failed_or_incomplete"]; errors: [].
- Resumed stages: ["load", "deduplication"]; no model training, LLM calls, or reexecution of source tests.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline4_partial_validation/report.json`. Held-out tests are excluded from training and hyperparameter-selection inputs.
<!-- /pipeline4-run:b6997189d36a7d0247296f944688392e1321b462e71361e726e38cc5e96f958c -->

## Pipeline 4 — Unified Dataset Validation & Training Preparation (26 October 2026)

### Architecture and scope

Implemented the independently executable `pip4_prepare` Typer command with Pydantic configuration and LangGraph discovery → source-specific loading/normalization/validation → deduplication/grouping → mixing → splitting → export/reporting. Source validation includes actual Qwen tokenization; no LLM API calls, repository-code execution, model weights, GPU, Docker, training, quantization, benchmarking, or security auditing are required. Ordinary credential/identifier checks are data-quality filters, not a security-auditing pipeline.

```sh
uv run llmops pip4_prepare \
  --input ./artifacts \
  --output ./artifacts/pipeline4 \
  --max-tokens 4096
```

Added nine modules under `src/policy_dissertation/pipelines/dataset_preparation/`: `__init__.py`, `schemas.py`, `adapters.py`, `quality.py`, `tokenization.py`, `storage.py`, `deduplication.py`, `selection.py`, and `pipeline.py`. Added `tests/test_pipeline4.py`, [docs/pipeline4.md](docs/pipeline4.md), and the [unified JSON Schema](docs/schemas/pipeline4-training.schema.json); extended CLI/configuration, README, dependencies, and lock file. Existing pipeline implementations were reused without rewriting their behavior.

### Unified schema and evidence

The NVIDIA-inspired schema has `id`, `messages`, `domain`, `source`, `verification`, and `metadata`, plus schema version and optional original tool definitions. Separate adapters validate Pipeline 1 extractive policy records, Pipeline 2 repository records, and Pipeline 3 multi-turn external records using their existing Pydantic models. Original messages/order, source references, tests, dataset revisions, licensing, and auxiliary metadata are retained. Policy evidence must match the extractive response, repository paths/hash/line references must be valid, and external references require valid source URL/revision/hash fields. Tool definitions and call/result protocols are checked.

Runtime labels distinguish `upstream_local_pass`, `externally_reported`, `not_applicable`, and `unverified`; `executed_in_pipeline4=false` always. Executable repository examples need consistent inherited Docker/image/positive-test/zero-failure evidence; mock or unverified execution is rejected. This checks consistency of upstream evidence, not authenticity through reexecution. External unit-test strings/statuses/scores are preserved without promoting them to local results. The selected Pipeline 2 input here is an explanation, not runtime-verified code. Original messages that already contained tests/context remain intact; no new metadata is injected as unsolicited conversation text.

### Validation, deduplication, mixing, and splitting

Input runs must be healthy with matching training-export hashes/counts. JSONL is streamed with bounded line reads; malformed, empty, non-finite, unsupported, incomplete, inconsistent-reference, marked-reasoning, and embedded-control-token records are rejected. Conservative configured-key/private-key/API-key/credential and sensitive-identifier patterns flag possible secrets/identifiers without storing their matched values. Entire conversations over the exact budget are rejected without truncation. These checks have false-positive/false-negative limitations and do not certify semantic or privacy correctness.

Exact deduplication uses actual rendered token-sequence hashes. SHA-256 word-five-gram sets, deterministic prefix filtering, length bounds, and full Jaccard verification detect near pairs at threshold 0.90. Formatting equivalents may be removed at 0.98; code indentation and meaningful word/operator/number/case differences are never normalized away for deletion. Other similar records remain connected into one split group. A brute-force regression verifies prefix-filter recall for the defined set metric. This is not exhaustive semantic-paraphrase detection. Representatives prioritize inherited verified coding, policies, repository knowledge, then external records. Complete duplicate provenance is in rejection audit references, with a bounded inline preview.

Default source weights are 40% policies, 30% repository, 30% external. The continuous availability bottleneck sets a conservative cohort cap; Hamilton rounding and optional `--max-examples` set integer quotas. No oversampling/duplicated examples manufacture missing supply. Domain quotas preserve diversity when capacity permits; verified coding is prioritized within its domain, then fixed-seed hashes sample equal-priority records. Actual selected/excluded counts are reported. A scarce source may make the balanced cohort very small.

Policy-document, repository-file/component, and external original-example groups are connected with exact/near links before selection, including unselected bridge records. SQLite stores data, prefix indexes, union-find state, compact group statistics, selection, and split assignments. Default 90/5/5 splitting uses seeded group-wise greedy allocation with source/domain/verification strata. Original explicit official test/validation/dev/evaluation labels and benchmark flags constrain placement; test takes precedence over validation conflicts. Groups remain intact and insufficient independent groups fail safely. Unknown benchmark overlap is not detected. Held-out tests are reserved for final evaluation and are never training or tuning inputs.

Transactional per-row loading/deduplication supports restart; source/tokenizer/code/dependency/configuration fingerprints prevent stale cache reuse. Exports are atomic per file, with report/manifest readiness and SHA checks. Source changes during processing, export failure, and research-report write failure clear readiness. Resolved source/output symlinks are guarded against overwrites.

### Actual Qwen3-14B preparation

Installed **Transformers 5.20.2**, **tokenizers 0.23.3**, and **Jinja2 3.1.6**. An attempted Transformers 4.x constraint conflicted with existing Hugging Face Hub 1.x; compatible 5.x resolved it. Initial template validation identified missing Jinja2, which was installed explicitly and the real tokenizer then passed. No unresolved tokenizer dependency error remains. The optional informational PyTorch-unavailable message is expected for this tokenizer-only installation.

Allowlisted small tokenizer/config files were downloaded from [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), pinned at `40c069824f4251a91eefaf281ebe4c544efd3e18`, with file hashes and original template hash recorded. No model weights or remote Python code were loaded. Exact `apply_chat_template` uses completed conversations, original tools, `add_generation_prompt=false`, `enable_thinking=false`, and `truncation=false`. The original template's own empty thinking wrapper is included in measured lengths. The snapshot's config/tokenizer guard is 40960 tokens; the requested 4096 is stricter. Tokenizer validation is preparation evidence, not model-runtime or long-context performance evidence. Downstream loss masking/training remains unimplemented.

### Executed tests and actual integration

Executed dependency installation, `uv sync --locked`, CLI help, full pytest, source compilation, `uv build`, default actual-input execution, explicit partial-source actual execution, repeat execution, and independent exported-record auditing. The final full suite recorded **214 passed, one skipped, five existing SWIG deprecation warnings in 7.99 seconds**. All **38 Pipeline 4 cases passed**, including one real cached Qwen-template/tool-tokenization test; the remaining preparation cases use small synthetic input records and fake length measurements where appropriate. The skip belongs to Pipeline 2's real Docker test (`docker_daemon_unavailable`). No mocked count is presented as an actual Qwen-token measurement.

Tests cover all adapters, multi-turn/tool messages, reference preservation, malformed/empty/oversized input, sensitive patterns, special tokens, unverified code rejection, exact/cross-source/format duplicates, retained differing mandatory rules, prefix recall, ratios, source/domain mixing, seeded splits, official-held-out propagation through duplicates, component/near leakage, source mutation, interrupted/resumed loading, export contracts, readiness rollback, and source overwrite/symlink guards. Initial fixture assertions failed because repeated identical context does not increase unique-shingle overlap; distinct context fixed that fixture. Later strict URL validation correctly exposed a renamed-dataset fixture with its old URL, which was corrected. Acceptance gates were not weakened to obtain passing tests.

**Strict actual-input run: FAILED, exit 1; exported train/validation/test counts 0/0/0; training_ready=false.** Actual discovery found no `artifacts/pipeline1/training.jsonl` (the Pipeline 1 directory is absent), and `artifacts/pipeline2/report.json` is `FAILED` despite one accepted explanation. Pipeline 3 has 1000 accepted rows and a healthy warning status. The strict workflow stopped before record validation rather than pretending all three required sources existed. Its [report](artifacts/pipeline4/report.json) and manifest record both blockers; empty exports are intentional failure artifacts, not training-ready data.

To exercise available sources without claiming complete readiness, executed:

```sh
uv run llmops pip4_prepare \
  --input ./artifacts \
  --output ./artifacts/pipeline4_partial_validation \
  --max-tokens 4096 \
  --max-examples 100 \
  --mix-ratio pipeline1=0 \
  --mix-ratio pipeline2=0.01 \
  --mix-ratio pipeline3=0.99 \
  --allow-partial-sources
```

| Final partial-source integration metric | Actual result |
| --- | ---: |
| Input records examined | 1001 |
| Validated within actual Qwen budget | 923 |
| Invalid / oversized rejected | 78 / 78 |
| Exact / formatting-near duplicates removed | 0 / 0 |
| Defined near pairs detected | 0 |
| Valid records excluded by explicit mixture | 823 |
| Exported | 100 |
| Train / validation / test | 90 / 5 / 5 |
| Component / exact / near leakage | 0 / 0 / 0 |
| Selected exact token length min / max | 72 / 3929 |
| Selected exact token length mean / sum | 708.93 / 70893 |
| Runtime-reexecuted tests / LLM calls | 0 / 0 |
| Status / exit | COMPLETED_WITH_WARNINGS / 0 |
| Full preparation readiness | **false (explicit diagnostic mode and missing/failed sources)** |

The 100 exports comprise **one repository explanation**, **54 OpenCodeInstruct records**, and **45 Nemotron chat records**; domains are **55 coding** and **45 general dialogue**. Verification distribution: **54 externally reported**, **one not applicable**, **45 unverified**, **zero inherited local runtime passes**. No corporate-policy examples were invented or substituted from synthetic tests. The rejection audit contains 901 references (78 invalid plus 823 selection exclusions); selection exclusions are not quality failures.

Repeated the identical command: completed load/tokenization and dedup checkpoints were reused, split files/hashes stayed identical, and no duplicate output was appended. Independently revalidated all 100 exports, reproduced their exact actual-token counts and rendered hashes, and compared their original messages and provenance with upstream file/line references. All matched. Brute-force examination of all **925 cross-split pairs** found zero Jaccard-near leakage at the configured threshold; split groups and IDs/rendered sequences were disjoint. SHA-256 comparison confirmed **400 protected files unchanged**, including `.env`, policies, repository inputs, and existing Pipeline 2/3 artifacts. No Pipeline 1 output was created to conceal its absence.

The initial final-code report and audit are retained under `artifacts/pipeline4_partial_validation/_runs/b6997189d36a7d0247296f944688392e1321b462e71361e726e38cc5e96f958c/verification/`. Public [report](artifacts/pipeline4_partial_validation/report.json), [manifest](artifacts/pipeline4_partial_validation/manifest.json), [train](artifacts/pipeline4_partial_validation/train.jsonl), [validation](artifacts/pipeline4_partial_validation/validation.jsonl), and [test](artifacts/pipeline4_partial_validation/test.jsonl) are diagnostic artifacts only.

### Limitations, corrective actions, and research relevance

A complete real three-source run remains **unverified and blocked**. Restore or generate healthy Pipeline 1 accepted exports/reports and finish Pipeline 2 successfully, then rerun the ordinary command. All-three-schema synthetic tests and two-source actual integration do not substitute for that missing full-corpus run. Source scarcity, coarse document/file groups, and forced official holdouts can prevent exact ratios or sufficiently large independent splits. Near comparison is lexical rather than semantic; unmarked benchmark duplication, upstream factual/code correctness, evidence authenticity, sensitive information outside configured patterns, and downstream learning benefit remain unverified.

The pipeline supplies a reproducible, versioned preparation boundary separating policy/code/external provenance and verification evidence from model-facing conversation text. Exact tokenizer measurements, documented selection exclusions, conservative deduplication, held-out isolation, and immutable input/audit hashes support controlled dissertation comparisons without inventing correctness or training results. No fine-tuning, quantization, benchmarking, or security auditing was started.
