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

## Pre-training refactoring — Pipelines 1–4 (2026-10-28)

### Scope and architecture

This task reuses the existing LangGraph pipelines, parsers, generators, source-grounding
checks, runtime evidence checks, SQLite/file checkpoints, deduplication, component
splitting and Qwen tokenizer. It does not implement Pipeline 5 or run model training.
The previous execution findings above remain historical evidence, not claims about
new runs. Existing policy documents, `.env`, repository inputs and existing datasets
were preserved.

The common Pydantic export contract was promoted from Pipeline 4 to
`src/policy_dissertation/data/schemas.py`. Old Pipeline 4 import locations re-export
it for compatibility. Its version remains `unified-qwen-sft-v1`, with `id`, `messages`,
`domain`, `source`, `verification`, `metadata` and optional `tools`. This is inspired
by Nemotron's role/message representation and OpenCodeInstruct's optional test
evidence; it is not a claim of identical NVIDIA schemas. Supported multi-turn and
tool-call/result ordering uses the established protocol validator.

Pipelines 1, 2 and 3 now validate and export this shared representation directly.
Their internal generation/normalization/validation checkpoints retain their legacy
contracts. `data/adapters.py` accepts the three old pipeline versions and the unified
version; it also exposes a reverse legacy adapter and explicit NVIDIA-native adapters
which reuse Pipeline 3 normalization. Legacy files are not rewritten automatically.
The export moves `example_id` to `id` while preserving the old identifier in
`metadata.upstream_example_id`. Original metadata remains in `metadata.provenance`;
messages, tools, source hashes, page/section references, repository line references,
dataset revisions, license declarations and test evidence are retained. Unified
records are checked against their original provenance, including mandatory policy
evidence and the existing coding-execution gates. Pipeline 3 rejects incompatible
unified records before committing accepted decisions.

Verification distinguishes inherited `upstream_local_pass`, `externally_reported`,
`unverified` and `not_applicable`. No external test results become local verification.
Pipeline 4 does not authenticate or rerun upstream tests. The original
`executed_in_pipeline4=false` field remains compatible. No new Docker dependency was
introduced; existing optional Python runtime verification still needs its restricted
container backend, and unsupported coding tasks remain subject to quarantine.

### Publishing, portability and flexible inputs

`data/hub.py` adds opt-in `--push-to-hub OWNER/DATASET` for Pipelines 1 and 2.
Publishing is disabled by default. It requires `HF_TOKEN`, a healthy nonempty local
export and matching hashes, then revalidates records and scans for configured secrets.
Private repositories are requested and existing public targets are refused. A staging
allowlist contains only `training.jsonl`, `manifest.json` and a dataset card. Internal
checkpoints, rejected records, `.env` and original source files are not uploaded.
Accepted conversations still contain source-derived text: privacy does not grant
redistribution rights. Original license references are preserved; unspecified licenses
remain unspecified. Bounded upload retries preserve local artifacts on failure;
existing manifests receive a safe failed-attempt marker. Successful manifests record
the repository and immutable Hub commit SHA.

`data/sources.py` and `configs/datasets.yaml` add typed YAML local/Hub source
configuration. Relative paths resolve against the YAML directory. The existing CLI
and legacy JSON settings remain supported. Explicit CLI output/token/seed values take
precedence. Configured source names and weights drive the existing mixing method;
no source duplication or oversampling was added. The default old interface still
requires healthy Pipeline 1–3 exports, while YAML readiness applies to the explicitly
selected source set.

Local files or directories are hashed; adjacent upstream reports, when present, must
show successful completion and matching export hashes. Standalone files get per-record
validation but cannot establish producer-run completeness. Hub sources pin immutable
revisions, retain source configuration/split and declared license metadata, and cache
checksummed complete snapshots under `_sources/`. Published exports are checked
against their Hub manifest. Other datasets stream into disk snapshots. Optional
`max_records` caps decoded remote rows for testing; native NVIDIA adapters are explicit.
Incomplete snapshots restart at the pinned revision with bounded retries and never
become completed cache hits. Complete snapshots and downstream SQLite decisions are
reused. Final source hashes are checked again before export.

All four CLI commands and Python runners now protect existing exports. A new output
folder or explicit `--overwrite` / `overwrite=True` is required, including resume into
an already exported folder. This intentionally adds a safety requirement to the old
resume procedure without deleting checkpoints or changing original sources.

The portable workflow is Mac generation → optional private Hub publishing → CPU-server
validation/mixing/splitting → future GPU-server training. Only `messages` and optional
`tools` enter the existing Qwen chat template; metadata remains auxiliary. Source
revisions, file/row hashes, exact token counts, component grouping, official holdouts,
seeded splits, secret checks and leakage prevention remain active. Paths in original
provenance need not exist on the consuming machine.

### Files and technology choices

- Added shared `data/{__init__,schemas,adapters,hub,sources}.py`, using existing Pydantic,
  Hugging Face Hub and Datasets libraries rather than a second orchestration framework.
- Updated accepted export boundaries in Pipelines 1–3 and extended Pipeline 4 discovery,
  adaptation and source weighting; retained backward-compatible Pipeline 4 imports.
- Extended `cli.py`, added direct `PyYAML>=6,<7` dependency and updated `uv.lock`.
- Added `configs/datasets.yaml`, `docs/data-portability.md`, and
  `docs/schemas/unified-training.schema.json`; updated README and pipeline guides.
- Added `tests/test_data_portability.py`; updated export assertions in the existing
  tests to exercise the shared schema while preserving legacy semantic assertions.
- Scoped `.gitignore`'s research `data/` exclusion to root `/data/` so the shared
  `src/policy_dissertation/data/` Python package is included in source control/builds.

Primary API references consulted: [Hub uploads](https://huggingface.co/docs/huggingface_hub/guides/upload)
and [Datasets streaming](https://huggingface.co/docs/datasets/stream). Streaming bounds
processing memory; it does not imply zero network transfer or unlimited disk capacity.

### Verification and actual results

Executed `uv sync --locked`, CLI help, `uv run python -m compileall -q src`, `uv build`
and the full `uv run pytest -q --tb=short` regression suite. Final automated result:
**240 passed, 1 skipped, 5 existing SWIG deprecation warnings (8.96 seconds)**. The
single skipped test requires a live Docker daemon, unavailable on this machine.
The 26 new transport tests use synthetic fixtures/mocked Hub calls: they verify schema
round trips, legacy/provenance checks, preservation of external tests, private/allowlisted
publishing, upload failure/retries, source pinning/caching, local and mixed loading,
native adapters, record caps, deduplication, split isolation, overwrite protection and
YAML CLI behavior. These mocked tests are not evidence of real Hub publication or
actual code execution. Initial regression failures during development identified old
export assertions, a rejected-record priority-path error, and simulated execution
counter assumptions; these were corrected before the final passing run.

Real checks are isolated under
`artifacts/pretraining_refactor_integration_20261028_071321/`, not existing output
folders:

| Check | Verified result |
|---|---|
| Live Pipeline 2, one Kotlin explanation unit, one attempt, 30-second timeout | **FAILED**: `endpoint_timeout`; 0 accepted; no live LLM success established |
| Live Pipeline 3, three Nemotron chat + three OpenCode rows | **COMPLETED_WITH_WARNINGS**: 6 processed/read, 6 accepted unified records, 0 rejected; 3 leading marked reasoning blocks removed |
| Live mixed Pipeline 4, six local prepared rows + three streamed native Nemotron math rows | **COMPLETED_WITH_WARNINGS**: 9 input/validated, 0 invalid, 0 duplicates, 3 excluded by mixing, 6 exported |
| Mixed split and quality result | train 4, validation 1, test 1; zero reported component/exact/near leakage; `training_ready=true` for these two configured sources |
| Selected distribution | coding 2, general dialogue 1, reasoning 3; external test evidence 2, unverified 4; no new runtime verification |
| Actual Qwen token counts | min 245, max 720, mean 478, total 2,868; configured maximum 8,192 |
| Independent audit | all 6 shared schemas, original/adapted messages, inherited verification, input row hashes and exact token counts checked; 6 unique IDs and disjoint split groups |
| Resume | load and deduplication checkpoints reused; all four JSONL export hashes unchanged |
| Input preservation | all 400 previously fingerprinted protected files unchanged, including original inputs and existing upstream artifacts |
| Packaging | wheel and source distribution built; all five shared Python modules present; `.env`, input repositories/policies and artifacts absent from wheel |

Pinned live dataset revisions were Nemotron
`9f36020daf067f1a8b39336bf619fe30af30bb02` and OpenCode
`8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`. The original Qwen tokenizer revision was
`40c069824f4251a91eefaf281ebe4c544efd3e18`; no model weights or GPU were required.
`verification.json` in the integration folder records the independent audit, and each
run has its actual `report.json`. No HF token or destination repository was configured,
so a real private upload/authenticated download was **not performed**. The successful
public streaming check does not establish private authentication success.

### Limitations and research relevance

The bounded LLM request timed out; endpoint availability/latency and complete fresh
Pipeline 1/2 LLM runs remain unverified in this refactoring session. Existing server
runs are separate evidence. Docker runtime verification was not exercised here.
Default three-source Pipeline 4 readiness is not established by the successful
explicitly configured two-source smoke test. Automated preparation readiness is not
semantic certification, source-license clearance, test authenticity, or fine-tuning
success. Unmarked reasoning, subtle policy/code errors, and unidentified benchmark
contamination still need appropriate review. General remote snapshots consume disk
proportional to selected input and replay from the beginning after an incomplete
stream; library read-ahead/network bytes are not measured. Unknown native dataset
schemas fail rather than receiving invented answers or metadata. The chosen weights
are illustrative and require later research justification.

The contribution is a traceable, portable training-data interface with reproducible
source revisions and split boundaries across enterprise policies, repository examples
and external data. It separates transport and preparation evidence from semantic and
runtime claims, supporting reproducible dissertation experiments and later deployment
across Mac, CPU and GPU environments. No Pipeline 5 implementation or model training
was started.

<!-- pipeline2-run:16ac11f2d7b7ea5c6edfe3a0a4da4988a10b42dbba72a99668242dc90d7e2ccd -->
## Pipeline 2 execution — 2026-10-29T01:33:13.945238+04:00

- Status: **FAILED**; repositories indexed: 2; selected units: 485.
- Failed units: 4; pending units: 0.
- Generated examples: 482; accepted: 112; rejected/quarantined: 370.
- Accepted runtime-verified: 0; accepted nonexecuting knowledge tasks: 112.
- Rejected/quarantined verification statuses: {"unverified": 308, "failed": 62}.
- Source languages: {"kotlin": 169, "typescript": 234, "python": 82}; generated task types: {"implementation": 69, "explanation": 70, "debugging": 69, "unit_tests": 68, "refactoring": 68, "architecture": 69, "secure_coding": 69}.
- Sandbox: docker_daemon_unavailable; runtime test invocations confirmed this run: 0.
- LLM calls this run: 1011; resumed generation/validation units: 0/0.
- Report: `/Users/mas/Development/DissertationProject/artifacts/pipeline2/report.json`.
- Syntax checks and model review do not establish runtime correctness. Unexecuted coding examples are quarantined; nonexecuting knowledge tasks are explicitly unverified for runtime.
- No repository or generated code was executed directly on the host. Errors: ["mandatory_unit_processing_incomplete"].

No Pipeline 3 or model fine-tuning was performed.
<!-- /pipeline2-run:16ac11f2d7b7ea5c6edfe3a0a4da4988a10b42dbba72a99668242dc90d7e2ccd -->

### Existing accepted-subset publication correction (2026-10-30)

The local Pipeline 2 report contains 112 accepted knowledge examples, 0 pending units,
4 failed units, and no runtime-verified accepted examples. Three failures are truncated
responses and one is an incomplete grounding review. Requiring regeneration solely to
publish these already accepted records was unnecessarily restrictive.

Added `llmops publish_dataset --input ... --repo-id ...` for upload without generation,
and explicit `--allow-incomplete-run` to publish validated accepted records while
preserving producer failure/completeness evidence. Default publishing remains strict.
The original producer report and training bytes are not rewritten; no rejected or
quarantined records are promoted. Record/schema/provenance checks, secret scans,
checksums, nonempty counts, private-target checks and bounded upload retries remain
mandatory. Validation now precedes remote repository creation. Hub manifests/cards and
local publication metadata disclose that this is an accepted subset from a failed run.
Pipeline 4 configured Hub loading retains that provenance and emits a warning.

No actual upload is claimed: a destination namespace and successful authenticated Hub
operation have not been established in this turn. Added mocked tests for standalone
publication, subset disclosure/report preservation, and refusal of invalid records,
count mismatches and checksum mismatches even in subset mode.

Verification for this correction: `uv run pytest -q --tb=short` completed with
**245 passed, 1 skipped** (Docker unavailable), with five existing SWIG warnings.
`publish_dataset --help` executed successfully. Offline checks of the actual local
accepted export validated all **112 records**, including provenance, secret scanning,
record count and matching training checksum. This was read-only validation, not a Hub
upload, new generation run, or semantic certification.

<!-- enterprise-run:a6f2fe718763fb8f3a1c08749742744f1404bd6be5ca0cbceca5edd6bcbafb04 -->
## Enterprise dataset execution — 2026-10-30T10:48:54.414902+04:00

Status: COMPLETED_WITH_WARNINGS; accepted: 112; duplicates: 0; invalid accepted records: 0. Source runs complete: False. Branches: {"code": {"mode": "import", "expected_count": 112, "producer_status": "FAILED"}}. No model training.
<!-- /enterprise-run:a6f2fe718763fb8f3a1c08749742744f1404bd6be5ca0cbceca5edd6bcbafb04 -->

## Unified enterprise workflow — implementation and preservation (2026-10-30)

### Objective and implementation

Added a single `llmops enterprise_data` workflow for corporate policy and repository
training examples, replacing the need to invoke two generation commands manually.
Either `--documents` or `--code` can be omitted; at least one input is required.
The existing parsers, structure-aware chunking, source-grounded LLM generation,
validation and supported isolated code-execution mechanisms are reused unchanged as
internal branches. The old `pip1_data` and `pip2_code` commands remain compatibility
interfaces; source IDs and schema origins retain their existing meanings.

A new LangGraph in `pipelines/enterprise/pipeline.py` orchestrates the optional document
branch, optional code branch and validated merge. Graph state contains stage names,
not document/code bodies. Fresh generation outputs and checkpoints are isolated under
`_branches/documents/` and `_branches/code/`; branches retain their existing fingerprints
and resume rules. The merge uses disk-backed SQLite staging, deterministically ordered
atomic UTF-8 JSONL export, schema/provenance checks, secret scanning, exact conversation
and tool deduplication, and count/hash verification. Matching duplicate provenance is
retained. An ID conflict with different content is rejected; near duplicates, mixing
ratios and split leakage remain Pipeline 4 responsibilities. No source text is silently
truncated by the merge.

The combined output contains one `training.jsonl`, `rejected.jsonl`, `report.json` and
`manifest.json`. The report separates branch acceptance/rejection/quarantine statistics,
merge-invalid records, duplicate occurrences, audit counts, model calls and producer
completion. Every requested branch's failure is visible; a successful branch does not
hide another branch's failure. The schema remains `unified-qwen-sft-v1`; messages,
original IDs, domains, verification and original provenance are preserved, with
additional enterprise acquisition metadata. No new test-execution evidence is invented.

### Existing coding dataset preservation and reuse

The user's existing coding dataset was not deleted, renamed, regenerated or modified.
The actual file is `artifacts/pipeline2/training.jsonl`; any existing `training.json`
files are likewise not targeted. The new workflow refuses an existing non-enterprise
output directory even if `--overwrite` is supplied. Input/output overlap is rejected.

Added read-only `--code-dataset` and `--documents-dataset` imports to merge previous
accepted exports without LLM calls. A raw source and an accepted import cannot both
be supplied for the same branch. Imported datasets require their adjacent producer
report, matching export checksum/count, and per-record revalidation. Imports do not
modify source checkpoints, training records, reports, manifests or historical rejection
files. Imported historical rejection counts remain references to original audits;
only new-branch audits and current merge failures are written to the combined audit.

An incomplete producer requires explicit `--allow-incomplete-import`. This enables
reuse of the 112 accepted local coding examples despite their producer's four failed
units. It preserves `producer_complete=false` and makes the enterprise report
`source_run_complete=false`, with a warning. It never promotes rejected/quarantined
examples or represents those producer failures as resolved. Private publication still
requires an additional explicit `--allow-incomplete-run` for incomplete sources.
The publisher now accounts for inherited enterprise source incompleteness. Pipeline 4
local-source loading retains that disclosure and warns rather than silently losing it.

Existing exports remain protected by `--overwrite`. For source-folder generation,
compatible completed branch checkpoints are reused on explicit resume; imports are the
recommended path when avoiding expensive regeneration is essential. `--limit-documents`
and `--limit-units` apply only to new generation, not to imported record counts.
Optional `--push-to-hub` publishes the single combined accepted dataset through the
existing private, allowlisted publisher. No actual Hub upload was performed here.

### Files and configuration

- Added `src/policy_dissertation/pipelines/enterprise/{__init__,pipeline}.py`.
- Extended `cli.py` with `enterprise_data`, optional folder/import inputs, generation
  test limits, checkpoint resume, optional one-dataset publishing and accurate summaries.
- Extended shared publisher completeness checks and local source provenance disclosure.
- Added `tests/test_enterprise.py` and `docs/enterprise.md`; updated README with the
  preferred enterprise workflow while retaining earlier implementation documentation.
- Added `configs/enterprise-datasets.yaml` for combined enterprise + NVIDIA preparation;
  existing user-customized `configs/datasets.yaml` is untouched. Example weights 0.60
  enterprise / 0.40 NVIDIA are illustrative, not experimentally optimized.

Use the new YAML for Pipeline 4 because the legacy `--input artifacts` interface still
expects the original three output directories. No Pipeline 5 or model training was
implemented. No additional external dependency was required.

### Executed verification and actual outcomes

Full regression run: `uv run pytest -q --tb=short` completed with **264 passed,
1 skipped**, and five existing SWIG deprecation warnings (8.88 seconds). The skipped
case needs a working Docker daemon, unavailable here. The 19 enterprise cases cover
actual parsing/static-indexing branch graphs with mocked LLM responses, either input,
both inputs, deterministic checkpoint resume, read-only imports, imported failed-source
opt-in, duplicate provenance, invalid/checksum/count mismatches, wrong source origins,
independent branch failure visibility, CLI/output protection, publication completeness
and Pipeline 4 compatibility/disclosure. Mocked generation is not live LLM or Docker
execution evidence.

Executed the new CLI against the actual local coding export, isolated to
`artifacts/enterprise_reuse_verification_20261030`, with `--code-dataset` and
`--allow-incomplete-import`. Result: **112 accepted examples, 0 duplicate conversations,
0 invalid records, 0 LLM calls**; status `COMPLETED_WITH_WARNINGS`, correctly retaining
`source_run_complete=false`. This was a read-only accepted-data reuse test, not new
code generation or a full corporate-policy generation run. A second explicit resume
produced the identical accepted export hash. Each of the 112 records was independently
checked against its original messages, verification and provenance, and validated
against the shared schema.

Before implementation, 383 protected files were fingerprinted, including `.env`,
policy/repository source files and top-level coding output artifacts. All 383 remained
byte-identical after verification. The preserved coding `training.jsonl` SHA-256 is
`1a6f281c651cfbced8a1a33db496e20796e98425e34e87a42133801711418487`.
The independent audit is saved in the isolated test output's `verification.json`.
CLI help, source compilation and wheel/source-distribution builds also completed.

### Limitations and research relevance

There was no new live policy/code LLM generation, authenticated Hub publication, or
runtime test execution in this task. The existing accepted coding examples are knowledge
examples, with no runtime-verified accepted coding tasks; unconfigured languages and
Docker limitations are unchanged. A successful enterprise merge proves record and
provenance compatibility, not semantic correctness or full producer completion.
Near-duplicate removal, final balancing and leakage-safe splits remain required later.

The enterprise facade simplifies the user workflow without conflating document and
code validation. Read-only import paths preserve costly earlier experiments and enable
one versioned enterprise dataset for later mixing with independently prepared NVIDIA
sources. Domain labels, original references and inherited verification distinctions
maintain the dissertation's traceability and reproducibility across local and server
execution. Existing experimental findings and source resources are preserved.

<!-- enterprise-run:b55ec43555919a8d4de81fbc342158460d73d885f01e3e374d900e12a61ee633 -->
## Enterprise dataset execution — 2026-10-30T10:55:13.675930+04:00

Status: FAILED; accepted: 112; duplicates: 0; invalid accepted records: 0. Source runs complete: False. Branches: {"documents": {"mode": "generated", "expected_count": 0, "producer_status": "FAILED"}, "code": {"mode": "import", "expected_count": 112, "producer_status": "FAILED"}}. No model training.
<!-- /enterprise-run:b55ec43555919a8d4de81fbc342158460d73d885f01e3e374d900e12a61ee633 -->

<!-- enterprise-run:7a02d28a53d56007528574c30f4817203fedb61b8f651cced0e411e889e38f6d -->
## Enterprise dataset execution — 2026-10-30T10:57:00.191535+04:00

Status: FAILED; accepted: 112; duplicates: 0; invalid accepted records: 112. Source runs complete: False. Branches: {"documents": {"mode": "import", "expected_count": 112, "producer_status": "FAILED"}, "code": {"mode": "import", "expected_count": 112, "producer_status": "FAILED"}}. No model training.
<!-- /enterprise-run:7a02d28a53d56007528574c30f4817203fedb61b8f651cced0e411e889e38f6d -->

<!-- enterprise-preparation:bfe69e96dfffd6fa31db9f85bd972ec8d56d9ea926d47b9526f06a4211a34cdb -->
## Embedded enterprise preparation

Status: COMPLETED_WITH_WARNINGS; enterprise-only training ready: True. Accepted input: 112; prepared: 92; preparation rejected: 20; duplicates: 0; splits: {"train": 82, "validation": 5, "test": 5}. Exact Qwen limit: 4096. Cross-source external audit remains required; no training started.
<!-- /enterprise-preparation:bfe69e96dfffd6fa31db9f85bd972ec8d56d9ea926d47b9526f06a4211a34cdb -->

## Embedded enterprise preparation and direct NVIDIA inputs (2026-10-30)

### Architectural decision and scope

Pipeline 4 is no longer a required user-facing step. Its tested deterministic
preparation engine is retained internally and as a hidden compatibility command,
rather than deleting working algorithms or breaking older experiments. The preferred
workflow is now enterprise generation/merge with automatic finalization, followed by
native external-data loading through shared adapters and a cross-source leakage gate.
The earlier combined enterprise/NVIDIA export and its YAML files remain optional legacy
workflows. No Pipeline 5, optimizer, GPU model load or fine-tuning was introduced.

`enterprise_data` now prepares by default after successful generation/merge. It applies
the pinned original Qwen3-14B chat template, exact conversation token limits, secret and
reasoning/control-token checks, exact/near-format deduplication, source-component
preservation, deterministic seeded splitting and leakage auditing. The existing engine
runs with enterprise weight 1.0, with no cross-dataset mixing quota. Source-generation
`--max-tokens` remains independent from final `--training-max-tokens` (default 4096).
Oversized records are rejected without truncation. Insufficient independent groups
fail the requested three-way split rather than silently exporting empty evaluation
sets. `--skip-preparation` is an explicit generation-only testing mode.

Added `enterprise_prepare --input ... --output ...` to finalize existing accepted
enterprise data without rerunning document/code generation. Its default requires a
new output directory and preserves the original input. The embedded path finalizes
newly generated output in place under the same lock, retaining an immutable original
accepted snapshot under `_accepted/`. Preparation engine checkpoints/diagnostics remain
under `_preparation/`. Prepared output includes `train.jsonl`, `validation.jsonl`,
`test.jsonl`, `preparation_rejected.jsonl` and readiness/quality statistics. The retained
`training.jsonl` is an **all-split audit union**, not the file to train on wholesale.
Reports distinguish generation counts from final prepared counts and retain inherited
incomplete-source warnings. `training_ready` is enterprise-only automated preparation
evidence, not a full-source, semantic or global cross-dataset certificate.

### Native external loading and shared leakage protection

`data/training_inputs.py` supplies direct iterators for native Nemotron and
OpenCodeInstruct records, reusing inspected immutable revisions/licenses, existing
NVIDIA-specific normalization, explicit marked-reasoning rules, source references,
external test evidence and exact Qwen token validation. Messages/tools are the
model-facing fields; verification and provenance stay auxiliary. Records are yielded
in memory; no normalized NVIDIA JSONL dataset or enterprise/NVIDIA merge is required.
The optional cached-export `pip3_datasets` workflow remains available.

A shared SQLite `TrainingInputGate` stores fingerprints/shingles and component IDs,
not conversation bodies. All selected enterprise/external holdouts must be registered
before `seal_holdouts()` permits training records. Cross-split exact, near and component
conflicts are blocked, and same-split duplicate/format-equivalent conversations are
excluded. Late holdout registration is refused. Native records explicitly marked as
holdouts fail a training stream rather than being silently assigned to train. Prepared
split labels survive shared adapters so subsequent consumers do not discard held-out
status. An unseen/unknown benchmark overlap is not certified absent by this mechanism.

Added `check_training_sources` for a bounded CPU-only native-stream preflight against
all enterprise splits. It records pinned sources, tokenization, decisions and audit
fingerprints, without saving NVIDIA training rows or starting training. Its record cap
is per-source decoded rows, not accepted rows or measured network bytes. This is a
reproducible source-order prefix, not uniform random sampling. A small preview cannot
replace full selected-source preflight. A future trainer must register all selected
holdouts before consuming training records through the gate; model loading, optimizer,
source sampling/mixing weights and resumable training cursors remain unimplemented.
Stream errors fail closed, rather than silently dropping source blocks. Native
normalization still has its independently configured byte-size safety guard.

### Publishing and backward compatibility

Prepared enterprise Hub publication now includes declared train/validation/test files,
with hashes/counts and a dataset card pointing to those files. The all-split audit
union is not advertised as the train split. Split membership and disjoint IDs are
checked before upload. Existing published-source loading selects the requested split
file/count rather than accidentally reading the audit union. Legacy unprepared
publication remains compatible. Private upload and accepted-subset disclosure rules
remain in force; no actual Hub upload was performed in this task.

Files added: `pipelines/enterprise/preparation.py`, `data/training_inputs.py`, and
`tests/test_embedded_preparation.py`. Updated CLI defaults/new commands, shared
publisher/source/adapters, README and enterprise/portability/pipeline guides. Existing
user YAML files, original policy documents, `.env`, coding source and existing coding
training outputs were preserved. No new dependency was required.

### Actual verification evidence

- Full regression command `uv run pytest -q --tb=short`: **278 passed, 1 skipped**,
  five existing SWIG deprecation warnings (8.84 seconds). The skip is the existing
  real Docker test, unavailable on this host.
- Fourteen added tests cover embedded finalization/CLI defaults, existing-input
  preservation, insufficient-group failure, mandatory holdout registration,
  cross-source exact/near leakage, direct native schema/test evidence, duplicate and
  oversized filtering, declared holdout handling, prepared split publication/loading,
  and preservation of holdout labels/auxiliary metadata. Synthetic tokenizers and
  mocked Hub calls are explicitly not live model/Hub verification evidence.
- Real `enterprise_prepare` used the existing 112-record coding-only enterprise reuse
  snapshot, isolated to `artifacts/enterprise_prepared_verification_20261030`. With the
  actual Qwen tokenizer and 4096-token bound, **92 records were accepted** and **20
  were rejected as oversized**, with no truncation. Splits: **82 train / 5 validation /
  5 test**. Status `COMPLETED_WITH_WARNINGS`; enterprise preparation ready true;
  inherited source-run completeness remains false. No LLM generation occurred.
- Actual token statistics: min 2716, max 4020, mean 3214.8695652173915, total 295768.
  The tokenizer revision remains `40c069824f4251a91eefaf281ebe4c544efd3e18`.
- Independent read-only audit rechecked all 92 shared schemas, unique IDs, original
  messages/verification/provenance, exact token counts and rendered token hashes;
  no cross-split source-component overlap was found. The original 112 coding records
  remain intact, including the 20 excluded from this smaller final token budget.
- Real `check_training_sources`, isolated to `artifacts/direct_nvidia_check_final_20261030`,
  preloaded all 92 enterprise split fingerprints and streamed **3 native Nemotron chat
  + 3 native OpenCode records**. All 6 passed the bounded check with no detected
  cross-source conflicts/rejections. Three closed leading reasoning blocks were removed
  under the configured normalization rule. No NVIDIA JSONL export was created.
  The final index contained 5 validation, 5 test and 88 train records (82 enterprise
  plus 6 external). Status `COMPLETED`; no training started.
- Native sources were pinned to Nemotron
  `9f36020daf067f1a8b39336bf619fe30af30bb02` and OpenCode
  `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`, retaining their declared licenses.
- All 383 previously fingerprinted protected files remained byte-identical. The
  independent audit is in the prepared test output's `verification.json`; original
  coding `training.jsonl` was not deleted, overwritten or regenerated.
- CLI help confirms `pip4_prepare` is hidden but callable for compatibility. New help,
  source compilation and package builds completed; the wheel includes the new shared
  native-loading and embedded-preparation modules. No model weights were downloaded.

### Limits and research relevance

This verification used existing coding knowledge examples, not new policy generation
or runtime-verified code tasks. It does not resolve the original four failed coding
producer units or unavailable language/Docker backends. Six streamed external examples
are a bounded integration check, not a full-corpus contamination audit or proof of
semantic correctness. Original NVIDIA structures remain on the Hub, but adapters and
quality checks still run before any record is eligible as model input. Fingerprint
index disk use scales with the compared token volume and needs production planning.
Networking errors require a fresh/replayed preflight; no training-progress resume is
claimed. License declarations and external tests are preserved without certifying
legal suitability or local test execution.

This change removes an unnecessary manual orchestration step while preserving exact
model-input checks and explicit source/split boundaries. It supports separately
versioned enterprise and NVIDIA data, traceable runtime adaptation, and reproducible
future experiments without requiring a physically merged corpus. All previous report
findings remain historical evidence; no model training was started.

<!-- enterprise-run:5c8379b5ca905c2194f2c87bb9bf7b52e9cc5c39b67c13bdb8bebd8fb14b48f4 -->
## Enterprise dataset execution — 2026-10-30T11:45:46.611782+04:00

Status: COMPLETED_WITH_WARNINGS; accepted: 226; duplicates: 0; invalid accepted records: 0. Source runs complete: False. Branches: {"documents": {"mode": "generated", "expected_count": 114, "producer_status": "COMPLETED_WITH_WARNINGS"}, "code": {"mode": "import", "expected_count": 112, "producer_status": "FAILED"}}. No model training.
<!-- /enterprise-run:5c8379b5ca905c2194f2c87bb9bf7b52e9cc5c39b67c13bdb8bebd8fb14b48f4 -->

<!-- enterprise-preparation:cdcbc9228ff7136284c4b589bfe031da71fef98ab296ee6a3578b92031f10297 -->
## Embedded enterprise preparation

Status: COMPLETED_WITH_WARNINGS; enterprise-only training ready: True. Accepted input: 226; prepared: 206; preparation rejected: 20; duplicates: 0; splits: {"train": 186, "validation": 9, "test": 11}. Exact Qwen limit: 4096. Cross-source external audit remains required; no training started.
<!-- /enterprise-preparation:cdcbc9228ff7136284c4b589bfe031da71fef98ab296ee6a3578b92031f10297 -->

## Final `enterprise-sft` dataset structure (2026-10-30)

This section documents the final prepared enterprise dataset, combining corporate
policy examples and repository-derived coding knowledge in one shared conversational
schema. The following counts were verified against local
`artifacts/enterprise_ready/` JSONL files and their recorded export checksums.
The local publication manifest records private repository **`sallout/enterprise-sft`**
and immutable revision **`28c64f71a8917f1c09af94fbd94c738643062ca5`**. That publication
record was inspected locally; this documentation update did not independently query
or modify the remote Hub repository.

### Published file layout

Application-managed files in the prepared Hub export are:

```text
enterprise-sft/
├── README.md           # Dataset card with explicit Hugging Face split mapping
├── manifest.json       # Schema, split counts, hashes, tokenizer and source-run disclosure
├── train.jsonl         # Model training examples only
├── validation.jsonl    # Held-out development / model-selection examples
├── test.jsonl          # Held-out final evaluation examples
└── training.jsonl      # All-split accepted audit union; never train on it wholesale
```

The `default` dataset configuration maps the Hub `train`, `validation` and `test`
splits to their corresponding files. The audit union is excluded from that split
mapping to avoid introducing validation/test records into training. Original source
files, `.env`, API tokens, checkpoints, rejection audits and internal fingerprint
indexes are not part of the upload allowlist. Additional local-only artifacts include
`report.json`, `rejected.jsonl`, `preparation_rejected.jsonl`, `_accepted/`,
`_branches/` and `_preparation/`, where applicable.

### Verified contents and split distribution

| Split | Corporate compliance | Coding knowledge | Total |
|---|---:|---:|---:|
| Train | 104 | 82 | 186 |
| Validation | 0 | 9 | 9 |
| Test | 10 | 1 | 11 |
| **All accepted records** | **114** | **92** | **206** |

The original merged accepted input contained 226 records: 114 policy and 112 coding
examples. Exact Qwen preparation at 4096 tokens excluded 20 oversized coding records
without truncation, retaining 206 records. Those exclusions do not delete the original
coding dataset or the original accepted snapshot. Verified accepted token statistics
are min 283, max 4020, mean 1636.2621359223301, total 337070. The pinned Qwen tokenizer
revision is `40c069824f4251a91eefaf281ebe4c544efd3e18`.

Splits preserve source-component isolation rather than forcing exact percentages.
The current validation split contains coding records only; corporate-policy validation
coverage is therefore not established by that split. The test split does contain 10
policy examples. Group-wise allocation does not guarantee that each domain appears
in every split; any later evaluation design must account for this observed limitation
without moving related source groups across boundaries arbitrarily.

### Canonical record schema

Each UTF-8 JSONL line is one `unified-qwen-sft-v1` record:

| Field | Type / meaning |
|---|---|
| `schema_version` | Literal `unified-qwen-sft-v1` |
| `id` | Deterministic 64-character SHA-256 identifier |
| `messages` | Ordered conversation objects with validated roles and content |
| `domain` | `corporate_compliance` or `coding` for the current enterprise export |
| `source` | Original pipeline origin, retained for traceability |
| `verification` | Typed verification status, optional tests and inherited evidence |
| `metadata` | Original provenance, enterprise acquisition and preparation/split information |
| `tools` | Optional function definitions when supported tool conversations require them |

The shared schema permits system/user/assistant/tool messages and validated multi-turn
or tool-call/result interactions; it does not invent absent interactions. Metadata
retains the original pipeline identifier and example ID, original source provenance,
input row/file hashes, enterprise branch/import status, and source-component grouping.
Policy provenance retains document IDs, source paths, section/page references and
exact evidence. Coding provenance retains repository/snapshot identity, file/symbol/
line references, source quotes and validation evidence. Prepared metadata additionally
records `training_split`, `split`, `split_group` and exact token/template information
under `preparation`. Optional fields depend on the actual record; missing tests are
not fabricated.

For model input, use only `messages` and optional `tools`, applying the pinned Qwen
chat template without silent truncation. `metadata`, `verification`, IDs and domain
labels are auxiliary fields, not extra model-facing conversation text. The future
trainer must implement appropriate assistant-target loss masking; no training success
or loss-mask implementation is claimed here.

### Verification status and intended use

The local prepared export reports `COMPLETED_WITH_WARNINGS`, enterprise-only
`training_ready=true`, and `source_run_complete=false`. Its publication is explicitly
marked `accepted_subset_only=true`, preserving the incomplete coding producer status.
All 206 current records have verification status `not_applicable`: the coding subset
contains accepted knowledge examples, not newly runtime-verified implementation
examples. This status does not mean that semantic correctness has been proven.

The prepared manifest records accepted counts, disjoint split membership, export
hashes, tokenizer identity/template settings, declared source restrictions and source
completion disclosure. The local publication metadata records repository/revision and
the audit-union hash. Use the immutable revision for reproducible subsequent loading.
NVIDIA datasets remain separate external sources, with runtime adapters and the shared
cross-source leakage gate required before future training. The enterprise-only dataset
readiness flag does not certify the combined enterprise/NVIDIA corpus.

This documentation update changed no datasets or split assignments. It used read-only
JSONL counts/domain/verification inspection and split checksum comparisons; no LLM
calls, Hub upload, model training or new experimental results were performed.

<!-- native-nvidia:9652b2e7417277f6886c18d0db51a94e1829c7ff239b9acc3ba5dbf7ad45a61c -->
## Native NVIDIA preparation execution — 2026-10-30T13:57:04.828054+04:00

Status: COMPLETED_WITH_WARNINGS; examined 20; accepted 20; rejected 0; decoded rows this invocation 22; resumed decisions 0. No model training or external test execution. Sources: [{"alias": "nemotron", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data", "config": "chat", "split": "train", "revision": "9f36020daf067f1a8b39336bf619fe30af30bb02", "license": "nvidia-open-model-license", "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/", "quota": 10, "scan_limit": 30, "processed": 10, "accepted": 10, "rejected": 0, "stop_reason": "target_met"}, {"alias": "opencode", "dataset": "nvidia/OpenCodeInstruct", "config": "train", "split": "train", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "license": "cc-by-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "quota": 10, "scan_limit": 30, "processed": 10, "accepted": 10, "rejected": 0, "stop_reason": "target_met"}]. Quality reasons: {}. This is selection-scoped structural/token/leakage evidence, not semantic certification.
<!-- /native-nvidia:9652b2e7417277f6886c18d0db51a94e1829c7ff239b9acc3ba5dbf7ad45a61c -->

<!-- native-nvidia:7c6aa59b41fa7e062577c78cff29f6d8ea3357b6c7ba47e5005161293094202e -->
## Native NVIDIA preparation execution — 2026-10-30T14:03:29.768210+04:00

Status: COMPLETED_WITH_WARNINGS; examined 20; accepted 20; rejected 0; decoded rows this invocation 22; resumed decisions 0. No model training or external test execution. Sources: [{"alias": "nemotron", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data", "config": "chat", "split": "train", "revision": "9f36020daf067f1a8b39336bf619fe30af30bb02", "license": "nvidia-open-model-license", "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/", "quota": 10, "scan_limit": 30, "processed": 10, "accepted": 10, "rejected": 0, "stop_reason": "target_met"}, {"alias": "opencode", "dataset": "nvidia/OpenCodeInstruct", "config": "train", "split": "train", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "license": "cc-by-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "quota": 10, "scan_limit": 30, "processed": 10, "accepted": 10, "rejected": 0, "stop_reason": "target_met"}]. Quality reasons: {}. This is selection-scoped structural/token/leakage evidence, not semantic certification.
<!-- /native-nvidia:7c6aa59b41fa7e062577c78cff29f6d8ea3357b6c7ba47e5005161293094202e -->

<!-- native-nvidia:b89931e0114246effcc3927bc8ea47ab4ef1a2e1fbe269a04017f86c21aea1bf -->
## Native NVIDIA preparation execution — 2026-10-30T14:10:13.355228+04:00

Status: COMPLETED_WITH_WARNINGS; examined 20; accepted 20; rejected 0; decoded rows this invocation 0; resumed decisions 20. No model training or external test execution. Sources: [{"alias": "nemotron", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data", "config": "chat", "split": "train", "revision": "9f36020daf067f1a8b39336bf619fe30af30bb02", "license": "nvidia-open-model-license", "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/", "quota": 10, "scan_limit": 30, "processed": 10, "accepted": 10, "rejected": 0, "stop_reason": "target_met"}, {"alias": "opencode", "dataset": "nvidia/OpenCodeInstruct", "config": "train", "split": "train", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "license": "cc-by-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "quota": 10, "scan_limit": 30, "processed": 10, "accepted": 10, "rejected": 0, "stop_reason": "target_met"}]. Quality reasons: {}. This is selection-scoped structural/token/leakage evidence, not semantic certification.
<!-- /native-nvidia:b89931e0114246effcc3927bc8ea47ab4ef1a2e1fbe269a04017f86c21aea1bf -->

<!-- native-nvidia:a806212ea7e74e359b04893c49697dbd019b319816599961ec1259ac3095bc42 -->
## Native NVIDIA preparation execution — 2026-10-30T14:49:10.223198+04:00

Status: COMPLETED_WITH_WARNINGS; examined 20; accepted 20; rejected 0; decoded rows this invocation 0; resumed decisions 20. No model training or external test execution. Sources: [{"alias": "nemotron", "dataset": "nvidia/Nemotron-Cascade-2-SFT-Data", "config": "chat", "split": "train", "revision": "9f36020daf067f1a8b39336bf619fe30af30bb02", "license": "nvidia-open-model-license", "license_url": "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/", "quota": 10, "scan_limit": 30, "processed": 10, "accepted": 10, "rejected": 0, "stop_reason": "target_met"}, {"alias": "opencode", "dataset": "nvidia/OpenCodeInstruct", "config": "train", "split": "train", "revision": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "license": "cc-by-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "quota": 10, "scan_limit": 30, "processed": 10, "accepted": 10, "rejected": 0, "stop_reason": "target_met"}]. Quality reasons: {}. This is selection-scoped structural/token/leakage evidence, not semantic certification.
<!-- /native-nvidia:a806212ea7e74e359b04893c49697dbd019b319816599961ec1259ac3095bc42 -->

## External Data — NVIDIA: completed native preparation stage and local verification (2026-10-30)

### Objective and implemented workflow

Completed the external-data stage as `llmops nvidia_data`, retaining NVIDIA data in
its native Hub storage rather than requiring another full normalized JSONL export.
The workflow prepares a reusable, reproducible approved selection for the future
training loader. It does not implement model training, source-mixing weights,
benchmarking, deployment, or a security-audit pipeline.

A new LangGraph orchestrates metadata/license/configuration inspection and source
pinning, enterprise split indexing, native streaming/adaptation/quality filtering,
transactional selection checkpoints, and source-plan/statistics publication. Native
Nemotron and OpenCode adapters, protocol checks, byte guards, marked-reasoning rules,
secret scans, exact Qwen token counting and cross-source fingerprint logic reuse the
existing implementations. Graph state contains stage/artifact references, not records.
No LLM endpoint or generated answers are used.

The total accepted quota is configurable across both datasets; default allocation is
equal, with explicit `--allocation ALIAS=count` support. Dataset selection,
Nemotron/OpenCode configuration, immutable source revisions, seed, bounded shuffle
buffer, exact token bound and a small-test candidate cap are configurable. The
sampler is a seeded prefix buffer, explicitly not uniform over the entire corpus.
Quota/exhaustion/candidate-limit shortfalls are reported; a positive-quota source
with zero valid examples fails the stage rather than inventing records.

The stage now pins the Qwen tokenizer to the prepared enterprise export's immutable
revision automatically. Both local enterprise artifacts and downloaded prepared Hub
files (manifest plus splits, without generation report.json) are supported. Every
enterprise split's hash, count, model/template identity, readiness and row labels
are checked before native training candidates are accepted. Revoked enterprise
readiness is not silently ignored on resume. Inherited incomplete-source status is
retained as a warning, not reclassified as complete producer success.

### Recovery, integrity and server reuse

Added `pipelines/native_datasets/{__init__,pipeline,replay}.py`, extended
`data/training_inputs.py`, exposed `nvidia_data`, added `tests/test_native_datasets.py`
and `docs/nvidia-data.md`, and updated README. No dependency was added.

The output contains:

```text
artifacts/nvidia_local_verification_20261030/
├── source_plan.json
├── report.json
├── _metadata/                 # Checksummed inspected metadata/cards
├── _tokenizer/                # Pinned tokenizer/config assets only
├── _runs/<fingerprint>/
│   └── fingerprints.sqlite3   # Global fingerprint index and native source decisions
└── verification/audit.json    # Independent local integration evidence
```

Approved/rejected decisions store source position/ordinal, raw hash, normalized ID,
exact token count/hash, transformation hashes and reasons. Full questions, answers,
tests and reasoning traces are not materialized as a new native training dataset.
Fingerprint indexes contain hashes/shingles rather than conversation bodies.

Gate mutations and decisions commit in one SQLite transaction. Matching completed
profiles reuse their decisions without rereading source rows. Interrupted/network
reads restart pinned streams and verify the committed seeded prefix before
continuing. Stream retries are bounded; shortened/changed replay and finalized
checkpoint corruption fail closed. Resume fingerprints include source/configuration
pins, enterprise split hashes, normalization/quality settings, seed, adapter code and
runtime versions. Errors are safe categories, not provider response bodies or API
credentials. Research-report failures revoke usable plan readiness. Earlier
fingerprinted development runs remain isolated rather than being appended together.

`iter_prepared_native()` provides the consumption path for future training: it
validates the plan/database/selection digest, matching code/dependencies, original
Qwen tokenizer/template, the exact enterprise split hashes and the global near-match
threshold. It replays only approved native coordinates, verifying raw content, IDs,
model token counts/hashes and the fresh shared leakage gate before yielding a record.
All selected enterprise/other evaluation sets must be indexed before training rows.
Externally reported test evidence is preserved and is never upgraded to locally
executed verification. Network errors during consumption propagate; this is not an
optimizer/training-cursor implementation.

For server reuse, copy the source-plan artifact directory with its referenced
`_runs/` decision database and the enterprise split files/manifest, install the same
project and locked dependencies, and use authenticated/public Hub access as applicable.
Paths to the decision database are relative and comparisons use content hashes.
Do not copy the Mac virtual environment. Native bodies remain on the Hub: replay
requires network access or a matching source provider/cache. A new output and larger
accepted quota can prepare a server-scale selection; no complete corpus download
or GPU is required by default.

### Executed tests and actual local results

Full `uv run pytest -q --tb=short`: **302 passed, 1 skipped**, with five existing
SWIG deprecation warnings (10.46 seconds). The single skip remains the unavailable
real Docker test from the repository pipeline, not a native-data failure. The 24
new native-stage cases cover total/source quotas, schema/license retention, native
replay, external test preservation, resume with zero new reads, interrupted-run and
network recovery, changed/shortened replay, cross-source heldout duplicates,
malformed/oversized/duplicate records, candidate limits, zero allocations, cache
corruption, research-report/access errors, missing local generation reports,
immutable tokenizer pinning and rejection of weakened replay thresholds. Synthetic
streams/tokenizers and mocked APIs are not presented as live integration evidence.

The live local command was:

```bash
uv run llmops nvidia_data \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/nvidia_local_verification_20261030 \
  --max-samples 20 \
  --max-records-per-dataset 30 \
  --shuffle-buffer 2 \
  --max-tokens 4096
```

Verified selected-source results:

| Source/configuration | Native candidates processed | Accepted | Rejected |
|---|---:|---:|---:|
| NVIDIA Nemotron / chat / train | 10 | 10 | 0 |
| NVIDIA OpenCodeInstruct / train / train | 10 | 10 | 0 |
| **Total** | **20** | **20** | **0** |

- Initial bounded selection read 22 records into the sampler, including two buffered
  rows beyond the 20 committed candidates. Reader/Arrow/HTTP read-ahead and actual
  network transfer bytes are not measured by these counters.
- Status **COMPLETED_WITH_WARNINGS**, selection-scoped `data_prepared=true`.
  Twelve closed-leading-think transformation occurrences were recorded. The warning
  retains `enterprise_accepted_subset_from_incomplete_producer`; it is not a new
  native processing failure and does not resolve the original coding failures.
- All 206 prepared enterprise records were indexed first: 186 train, 9 validation,
  11 test. The final fingerprint index contains 206 train (186 enterprise + 20 native),
  9 validation and 11 test. No cross-source conflict was detected in this selected
  20-record sample; this is not a statement about the full NVIDIA datasets.
- Repeating the identical command resumed all 20 decisions, accepted the same 20,
  and read **0 new source rows**. Approved selection digests matched across the
  bounded initial and final hardened implementation profiles; initial evidence is
  retained separately from the cache-only resume report.
- Live approved-selection replay independently checked all **20 unique IDs**, shared
  schemas and exact Qwen token counts/rendered-token hashes against pinned native
  rows. Selected token counts range **196–2833**, below 4096. All **10 OpenCode**
  records retained original external test/judgment metadata and `locally_verified=false`.
  The audited decision database hash was unchanged after read-only replay.
- **399 protected resource/data files remained byte-identical**, including `.env`,
  policy/repository sources and enterprise/coding dataset artifacts. No native
  training JSONL was created. No LLM calls, external test execution, model weights,
  training or Hub publication occurred.
- Evidence lives in `initial_final_profile_report.json`, `report.json`,
  `source_plan.json` and `verification/audit.json` under the isolated native output.
  The final-profile initial and resume reports share the same run ID;
  `initial_verification_report.json` separately retains an earlier development run.
  Development code hardening produced distinct fingerprint directories; approved
  row hashes/selection digests were compared rather than claiming those code profiles
  were identical.

Pinned sources remain Nemotron `9f36020daf067f1a8b39336bf619fe30af30bb02` and OpenCode
`8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`, with their inspected NVIDIA Open Model
License and CC BY 4.0 declarations respectively. The Qwen tokenizer is
`40c069824f4251a91eefaf281ebe4c544efd3e18`. Verified runtime versions are Datasets
4.8.5, PyArrow 26.0.0, Pydantic 2.14.1, Transformers 5.20.2, Tokenizers 0.23.3 and
LangGraph 1.2.17. Original license/source references and test statuses remain metadata;
no new redistribution rights or locally verified test results are inferred.

### Remaining scope and research relevance

The local integration is successful, but a small seeded prefix does not prove
semantic correctness, full-corpus contamination absence, coverage of every Nemotron
configuration, server throughput/disk requirements or GPU fine-tuning success.
Unmarked reasoning and unidentified benchmark overlap remain review risks. The
approved plan is scoped to the chosen 20 records and current enterprise holdouts;
new sources/holdouts require corresponding shared auditing before optimization.
Fingerprint storage grows with token/shingle volume. Network interruptions during
training-time consumption need future trainer-owned cursor/checkpoint handling.
Training mixing weights, model loading and optimization remain unimplemented.

This completes a reproducible external data engineering component: native datasets
stay separately versioned, accepted coordinates are durable and auditable, schema
adaptation is verified on replay, and enterprise/external isolation checks remain
mandatory. It provides concrete local evidence before a larger CPU-server run while
preserving the dissertation's distinction between structural data preparation,
external test claims, semantic correctness and actual model-training results.

## Interactive CLI workspace — implementation and verification (2026-10-30)

### Objective and user-facing entry point

Implemented an interactive terminal workspace so an operator can extract and validate
enterprise/external data without assembling pipeline commands manually. `uv run llmops`
now opens the menu; `uv run llmops menu` is an explicit alias. Existing direct commands,
help behavior and pipeline error exit codes remain available for automation. The new
entry point does not start a pipeline until an operator selects an action, supplies
its settings and confirms the run.

The header displays the requested **Bridging the Cusem** title, **By Mohammed Sallout**,
and the user-provided clickable profile **https://www.linkedin.com/in/msallout/**.
Branding is configurable via `UI__TITLE`, `UI__AUTHOR`, `UI__LINKEDIN_URL`; the title's
original supplied spelling was retained rather than silently renamed.

The menu contains Enterprise data extraction, Enterprise data validation, External
dataset installation, External dataset validation, Fine-tuning, Security check,
Prompt injection check, Model customization, Env setup and Exit. Fine-tuning and the
three future inspection/customization actions are explicitly unavailable, greyed out
and non-executable. This task did not implement those future pipelines, rent a GPU,
load model weights or start training.

### Interface architecture and behavior

Added `interface/{__init__,dashboard,environment,validation}.py`, UI branding settings,
no-argument/menu entry points and the `nvidia_validate` automation command. The menu
calls the existing Typer command dispatcher using an argument list in the current
process; it does not construct shell strings or duplicate processing logic. Rich
provides headers/panels/forms/report summaries, while Questionary supplies terminal
arrow/number navigation, path completion and hidden credentials. Non-TTY sessions
use a numbered prompt fallback. Ctrl+C/EOF/cancellation and invalid input are handled
without accidentally launching work.

Enterprise extraction can read either optional raw folder or import an existing
accepted branch dataset. Reusing the existing code dataset requires no new code
inference. Wrong/mixed branch imports are identified before processing; incomplete
producer subsets require explicit consent. Supported document formats accurately
reflect the existing PDF/Markdown/TXT parser. Suggested source budget is 4048 UTF-8
byte-token units, separate from final exact Qwen limits and LLM response budgets.
Small/custom/full generation scopes, repository task types, output location and
explicit resume/rewrite are collected interactively. Source generation requires
configured endpoint/key/model; imports do not.

The extraction menu defaults to generation/merge only so the separate enterprise
validation action can be exercised. Operators can alternatively include preparation
in extraction. The existing direct `enterprise_data` command retains its automatic
preparation behavior. Validation calls the existing preparation service without
regenerating data; exact Qwen token bound and split seed remain controllable. Small
fixtures lacking independent groups can still legitimately fail a three-way split;
no safeguards were relaxed to make those fixtures appear ready.

Enterprise validation also provides numeric report inspection and optional private
Hub publication of an existing export. The publication action requires an actual
owner/dataset identifier and explicit subset permission when applicable; it does not
mistake successful publication for successful completion of the original producer.
No actual publication was performed during UI verification.

External installation collects sources, total accepted cap, candidate cap, Qwen token
limit, seed/buffer and optional revision/configuration/allocation, then uses the existing
resumable `nvidia_data` workflow. It creates a native source plan, not an eager full
corpus download or another normalized JSONL dataset. External validation uses a new
read-only action around the existing approved-selection replay API, fresh enterprise
holdout indexing and source/ID/token/integrity checks. Optional partial checks are
explicitly marked `full_plan_validated=false`; full checks mean the approved selection,
not the complete upstream NVIDIA corpora. Existing input artifacts remain untouched.

Live backend progress and actual warning/failure status remain visible. Successful
output paths are remembered for subsequent forms; job operation/output/exit code and
duration are recorded under Git-ignored `.llmops/`. Input credentials and full form
answers are not stored in menu history. A history-write error does not change the
reported processing outcome. Interactive session exit is independent of the last job
exit code; automation can continue using the direct commands' nonzero process codes.

### Environment handling and safety

Env setup supports read-only configured/missing status without exposing values or
claiming authentication/connectivity success. Credential/endpoint entry is hidden and
requires an interactive terminal; piped secret entry is refused. Blank entries keep
existing values. An explicit save validates a staged `.env`, preserves unrelated
settings/comments, creates a unique owner-only backup and atomically replaces the
file. New file/backup permissions are 0600. Symlink editing, unsupported keys,
multiline/control/interpolated edits and invalid URLs/configuration are rejected.
Process overrides are reported without revealing their values.

The tuning screen controls existing document/code response budgets, request timeouts,
bounded attempts, external retries/timeout/candidate multiplier and logging level.
It warns that changed generation settings can invalidate checkpoints, recommending
accepted-code imports when avoiding regeneration. No project `.env` was edited in
this implementation run; environment writing/backup/repair tests use temporary fixtures.
Configuration presence is not proof that an endpoint or token has working permissions.

Direct UI dependencies are Rich 15.0.0, Questionary 2.1.1 and python-dotenv 1.2.4;
Questionary uses prompt-toolkit 3.0.53. `pyproject.toml`/`uv.lock` were updated. Primary
API references consulted were [Questionary question types](https://questionary.readthedocs.io/en/stable/pages/types.html)
and [Rich prompts](https://rich.readthedocs.io/en/stable/prompt.html). Backend data
engines, original schemas, source grounding, privacy filters, checkpoint checks,
private publication defaults and split leakage rules were reused rather than rebuilt.

### Actual verification

- Full `uv run pytest -q --tb=short`: **336 passed, 1 skipped**, five existing SWIG
  deprecation warnings (11.35 seconds). The skip is the existing real-Docker repository
  test, not a UI/native-validation failure.
- Thirty-four added test cases cover no-argument/menu launch, correct branding,
  disabled stages, cancel/EOF/bad-input behavior, existing coding reuse, wrong-branch
  imports, generation prerequisites, test limits, output protection, source settings,
  full/sample native validation, actual nested command execution, masked credentials,
  owner-only backups, env rejection/repair, history/defaults and backend error handling.
  Data-generation and Hub-dependent fixture calls are mocked; they are not presented
  as new teacher generation, real upload or training evidence.
- Real PTY terminal checks exercised number navigation into Env setup, read-only
  configuration presence, return to the main menu, five down-arrow moves skipping
  disabled stages, and clean Exit. Header/profile link and disabled stages rendered
  correctly. No secrets were printed and no `.env` save was invoked.
- A real numbered-menu run selected External dataset validation and called the actual
  `nvidia_validate` backend against the existing approved native selection and
  `artifacts/enterprise_ready`. Result: **COMPLETED, 20/20 records checked,
  full_plan_validated=true, 0 LLM calls, no training**. It reused the approved source
  plan without installation/regeneration. The audit is at
  `artifacts/menu_native_validation_20261030/report.json` with a fresh fingerprint index.
  This is live selected-row replay evidence, not verification of all NVIDIA data.
- All **404 fingerprinted existing resource/export files remained byte-identical**,
  including `.env`, policy/repository source inputs, enterprise/coding datasets and
  top-level native plan/report artifacts. No existing coding training file was deleted
  or regenerated. The menu wrote only its own state/new validation audit and the
  explicitly requested project implementation/documentation files.
- Source compilation, source/wheel packaging and `git diff --check` completed.
  README and `docs/interactive-cli.md` explain launching, controls, stages, environment
  handling, partial/full validation, recovery and limits. Original CLI help/regression
  behavior remains covered.

### Limits and research relevance

This is a synchronous terminal UI, not a background scheduler or remote GPU controller.
Fine-tuning, security/prompt-injection checks and model customization remain unavailable
and cannot be launched. No new live document/code LLM generation, credential write,
private Hub upload or GPU test was performed. Endpoint permissions still need actual
service access; existing inherited incomplete-source and evaluation-coverage limitations
remain documented. Success on 20 approved native records is not semantic correctness
or full-corpus contamination certification.

The interface makes pre-training experiments reproducible and operable from a single
entry point while retaining the established execution services and audit artifacts.
It separates extraction, preparation, native installation/selection and validation,
prevents accidental costly regeneration, keeps credential values outside logs/history,
and exposes actual evidence before committing to later rented-GPU training work.

## Published enterprise dataset restoration — 2026-10-30

### Objective and implementation

A new server must be able to reuse the already published enterprise dataset without
rerunning document/code generation. Added `interface/restoration.py`, the
`download_dataset` Typer command, and a **Download an existing enterprise dataset
from Hugging Face** action under Enterprise data validation. No parsing, generation,
training or publication service was rewritten. Existing exports are protected unless
replacement is explicitly requested; the interactive preview still requires confirmation.
The menu remembers the restored output as a suggested enterprise preparation/input path.

Uses the existing Hugging Face Hub client: resolves owner/name and branch/tag to an
immutable commit, then calls `snapshot_download` with an explicit file allowlist and
bounded retries. Private access uses configured HF_TOKEN without displaying it; public
repositories can be downloaded without a token. Official API reference:
https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download .
No additional dependency was required. Hub caching supports transfer reuse on retry.

Before committing exports, checks every required SHA-256 against the remote manifest,
adapts/validates every conversation with the existing canonical adapter, rejects duplicate
record IDs, verifies accepted counts, and for prepared exports verifies exact record
membership/content, disjoint split IDs, split labels/counts and full split coverage.
Preserves downloaded data bytes and manifest. Creates a local report carrying the pinned
repository revision, tokenizer provenance, original readiness and source-completeness
status. Missing/unknown completeness remains a warning. An accepted-only export remains
unprepared. These checks do not repeat code execution, tokenization or semantic review.

### Actual verification

- Targeted restoration/menu tests: **39 passed**, 5 existing SWIG warnings.
- Full `uv run pytest -q`: **346 passed, 1 skipped**, 5 existing SWIG warnings,
  11.93 seconds. Skip: existing real Docker test, daemon unavailable.
- Ten new cases cover actual service file-copy/checksum/schema/count/split behavior
  using synthetic mocked Hub access, protected existing exports, inherited incomplete
  status, menu routing without local source inputs, and CLI routing. Mock tests are
  distinct from the following real private-Hub integration.
- Real command executed:
  `uv run llmops download_dataset --repo-id sallout/enterprise-sft --revision 28c64f71a8917f1c09af94fbd94c738643062ca5 --output artifacts/enterprise_hub_restore_verification_20261030`.
  Result: **206 accepted records, training_ready=true,
  status=COMPLETED_WITH_WARNINGS**. Six allowlisted remote files fetched, existing
  train/validation/test preserved; no LLM calls or model training. The warning is
  inherited incomplete producer evidence, not a failed download. Original local
  enterprise/coding exports and `.env` were not replaced.
- README and interactive CLI guide now describe the server restoration path and
  equivalent command. Dataset source files/checkpoints and the separate native NVIDIA
  source plan are not restored by this enterprise download. Further semantic correctness
  or full upstream completion is not established by successful restoration.
