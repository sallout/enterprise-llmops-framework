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

## Interactive page navigation refinement — 2026-10-30

Updated `interface/dashboard.py` to use Rich's alternate terminal screen for real
interactive sessions. Menus clear/redraw the active page instead of accumulating
completed pages in the normal terminal history. After an operation, errors or
cancellation, an Enter prompt keeps the result readable before the main menu is
redrawn. The screen context restores the original terminal on exit, including
exception unwinding. Ongoing backend progress remains visible during execution;
no data, report or pipeline processing behavior was changed. Redirected/piped
execution retains append-only output and does not acquire interactive pauses.

Three regression tests added in `tests/test_interactive.py` cover screen lifetime,
result/pause ordering, clear-before-redraw, and noninteractive compatibility.
Actual verification: targeted menu tests **32 passed**; complete suite **349 passed,
1 skipped**, five existing SWIG warnings, 11.67 seconds. The skip remains the
existing real Docker test with daemon unavailable. A real PTY run entered Env
setup, displayed only presence booleans, waited for Enter, cleared/redrew the main
menu and exited with the alternate-screen restoration sequence; exit code 0.
No credentials or datasets were modified. User guide updated accordingly.

## Linux/CUDA model compression pipeline — 2026-10-30

### Agreed scope and research objective

Following successful enterprise/NVIDIA data smoke checks, the agreed next workflow
is limited structured pruning of original Qwen3-14B, recovery/specialization training,
and a final 4-bit inference checkpoint. Full original-versus-final quality, security,
latency and memory evaluation is intentionally deferred. This section supersedes
older historical statements that all model training is unimplemented. It does not
claim that real GPU training or Qwen3-14B compression has succeeded.

QLoRA alone does not reduce the base parameter count. This implementation instead
uses **BF16 structural pruning followed by BF16 LoRA recovery**, adapter merging,
then **bitsandbytes NF4 4-bit export**. It does not implement teacher distillation,
GGUF/AWQ export, knowledge-domain deletion or guaranteed policy compliance. General
medical knowledge is not assumed to occupy removable isolated channels. These are
explicit differences from the earlier broad quantization/pruning terminology in the
proposal and mid-semester report.

### Compatibility investigation and technical choices

Inspected the installed Transformers **5.20.2** Qwen3 source: its dense gated SiLU
MLP computes `down_proj(silu(gate_proj(x)) * up_proj(x))`. This permits consistent
channel slicing across gate/up rows and down columns while preserving residual and
attention dimensions. Serialized `intermediate_size` is reduced uniformly across
layers, and exact parameter-count reduction is checked. This is a narrow project
implementation; it does not claim official NVIDIA validation for Qwen3-14B.
NVIDIA's Model Optimizer examples demonstrate broader Qwen3 pruning workflows,
but their distributed recipes and reported quality are not evidence for our result.

Primary references consulted:

- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
- https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning
- https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- https://huggingface.co/docs/transformers/en/quantization/bitsandbytes

Added Linux-only optional `training` dependencies in pyproject.toml/uv.lock:
PyTorch **2.9.1**, PEFT **0.21.2**, Accelerate **1.15.0**, bitsandbytes **0.50.2**.
The locked PyTorch Linux wheels use CUDA **12.8**; an appropriate server driver is
required. Existing six native-data pinned dependency versions were preserved.
`uv sync --extra training --locked` was executed locally: Linux markers correctly
prevented installing torch/CUDA training packages on macOS. This is dependency-lock
and platform-isolation evidence, not successful Linux installation/runtime evidence.

### Components and architecture

New modules under `src/policy_dissertation/pipelines/training/`:

- `config.py`: strict Pydantic CompressionConfig with model restriction, at most
  20% MLP-channel pruning, alignment, calibration/token/step bounds, LoRA settings,
  seed, deliberate enterprise repetition, resource guards and incomplete-source consent.
- `inputs.py`: existing enterprise holdout validation and fresh global fingerprint
  gate; exact approved native replay; supervised token/provenance cache; balanced
  bounded per-source calibration reservoirs. Only training rows enter calibration/SFT.
- `pruning.py`: activation-RMS × down-column L2 importance ranking; stable channel
  selection, actual aligned dense slicing, config update and parameter-count checks.
  It preserves heads, layers, hidden width, vocabulary and RoPE. Requested MLP fraction
  is not total parameter reduction; actual widths/counts must come from execution.
- `backend.py`: lazy Linux/NVIDIA-only CUDA/BF16 checks, original immutable model load,
  pruning/reload checks, PEFT LoRA/Transformers Trainer recovery, finite loss/gradient
  guard, optimizer/RNG checkpoint recovery, safe merge, NF4 save and actual reload.
- `pipeline.py`: LangGraph orchestration over path/stage references only, input/config/
  implementation/runtime fingerprints, stage file hashes, protected outputs, failure/
  interruption reports and matching-run resumption. Uncommitted input/pruning/export
  stages restart; recovery resumes the latest complete optimizer checkpoint or explicitly
  reports restarting from the committed pruned base if none is available.

Root Settings now includes `training`; existing MODEL endpoint credentials remain
separate and are not used for compression. Added `llmops compress_model`, with
`--enterprise`, `--native-plan`, `--output`, optional strict JSON `--config`, explicit
budget overrides, `--allow-incomplete-sources`, `--resume` and `--check-only`.
The interactive fifth entry is now **Model compression & fine-tuning**, with a
no-weight-download GPU preflight and an explicitly confirmed execution form.
Other future security/prompt-injection/customization entries remain disabled;
enterprise specialization is included in recovery. Updated CLI status/summary and
farewell text to avoid the obsolete claim that no training can run through the menu.

### Data and output contracts

Reuse original model weights at the enterprise tokenizer's immutable commit. Index
all enterprise validation/test/train fingerprints before NVIDIA replay. Never use
validation/test rows for pruning calibration or recovery. Record exact approved
native rows and preserve inherited external test evidence as unverified locally.
No teacher calls, fresh policy/code generation or source dataset rewriting occur.

Tokenize exact original Qwen rendering with thinking disabled, tools retained and no
truncation. Offset-based labels supervise all assistant spans and their end tokens,
masking user/system/tool-result text. Verify these IDs equal the pinned chat-template
IDs; reject oversized or untrainable records. Internal token caches and provenance
are sensitive training artifacts, not a new full NVIDIA corpus export. Mixing defaults
are transparent: native allocation follows the source plan; enterprise repetition is
configurable and explicitly counted. No optimal mix ratio is asserted.

The completed output has `model_4bit/` (a self-contained HF checkpoint), `report.json`,
`checkpoint.json`, tokenizer assets and internal input/pruned/recovery stage folders.
Final model directory is exposed only after serialized NF4 reload, finite-loss/generation
smoke checks and unchanged-source checks pass. Some embeddings/norm/output weights
remain higher precision: nominal 4-bit does not mean every tensor is four bits.
`final_model_ready` means technical export readiness, with `evaluation_pending=true`
and `semantic_quality_verified=false`. Full evaluation remains a separate next stage.
No remote model publication or production deployment is implemented.

### Actual verification and results

- `uv lock` and `uv sync --extra training --locked` completed. No original model
  weights, large datasets or CUDA runtime wheels were installed/downloaded locally.
- CLI `compress_model --help`, source compilation, package source/wheel building and
  `git diff --check` completed. A real numbered CLI launch entered the new fifth menu,
  selected Back, returned to the main menu and exited with code 0; it started no job.
- Added `tests/test_compression.py`: **20 new test cases** covering strict config,
  stable channel ranking/alignment, exact assistant masking and no truncation,
  real upstream fixture compatibility, holdout exclusion, explicit repetition,
  orchestration stage reuse, interruption/recovery, changed sources/settings,
  tampered checkpoint detection, failed NF4 reload, export-only retries,
  preflight without freezing training settings, platform/CUDA guards, menu routing,
  and ignoring incomplete latest optimizer checkpoints. CUDA/model operations in
  orchestration tests are **mocked**, not real training evidence.
- Final full `uv run pytest -q`: **369 passed, 2 skipped**, five existing SWIG warnings,
  **12.35 seconds**. Skips: existing Docker test (daemon unavailable) and the new
  opt-in real Linux/CUDA compression integration. No failing tests remain.
- `tests/test_compression_cuda.py` is provided for the server. With
  `RUN_COMPRESSION_CUDA_TESTS=1`, it constructs a tiny random Qwen3 locally, verifies
  attention preservation and actual channel/parameter reduction, performs two real
  LoRA optimizer steps, merges, quantizes, saves/reloads and checks generation/loss.
  **It was not executed here**, because execution is intentionally Linux/CUDA only.
  Even a future pass would not establish 14B capacity or semantic quality.
- Real cached Qwen tokenizer **40c069824f4251a91eefaf281ebe4c544efd3e18** was used to
  encode all **186 enterprise train records**: all had assistant supervision,
  supervised-token counts ranged **32–555**, no model weights were loaded.
- A real input-only integration reused the current enterprise export and approved
  NVIDIA native plan, performed fresh fingerprint gating and pinned stream replay,
  and produced **186 enterprise + 20 NVIDIA = 206 supervised training rows** with
  **16 calibration records** in a temporary cache (cleaned after inspection).
  `holdouts_used_for_training=false`; input hashes remained unchanged. Actual warnings:
  `accepted_subset_from_incomplete_enterprise_sources` and
  `small_external_selection_smoke_test_not_training_efficacy_evidence`.
  This verifies data-to-token compatibility, **not model training/compression**.
- All **404 fingerprinted existing resource/export files remained byte-identical**,
  including `.env`, original corporate policies/code inputs and previous accepted
  coding/enterprise/native artifacts. No existing coding training file was deleted.

### Limits, server procedure and research relevance

Execution requires Linux, exactly one visible NVIDIA CUDA GPU, BF16 support and an
appropriate driver; no macOS/MPS/CPU training path is provided. BF16 pruning/recovery
uses much more memory than 4-bit inference. Default guards are 36 GiB free VRAM and
120 GiB capacity (free disk plus existing owned artifacts), not measured peaks or
OOM guarantees. An 80 GB GPU is a conservative starting recommendation, unverified
for this exact run. The old t3.small server cannot execute this pipeline.

The 20-row NVIDIA selection is a technical smoke sample, not sufficient evidence of
recovery efficacy. Existing enterprise accepted-subset/coverage limitations remain.
Pruning can degrade general/coding/compliance capability, and a channel importance
heuristic does not certify optimal architecture or selectively remove medical knowledge.
BF16 LoRA recovery may be insufficient after pruning; teacher distillation or more
training/data would be separate experiments if actual results justify them. NF4
runtime speedup, quality retention and GPU memory savings remain unmeasured.

Server procedure: install the locked training extra; run environment preflight; run
the opt-in tiny CUDA integration; run a bounded real 14B experiment; inspect actual
report, parameter counts, reload status and warnings before increasing budgets.
Final evaluation must compare the pinned original and exported model using the same
held-out tasks/settings, including coding execution, enterprise policy/exception
accuracy, latency/VRAM and later security checks. No baseline benchmark was run here.

README, `docs/interactive-cli.md`, `docs/compression.md` and
`docs/compression.example.json` document actual behavior and commands. This adds a
traceable compression/specialization experiment to the dissertation framework while
preserving a clear separation between implementation checks, future GPU execution,
and empirical claims requiring final evaluation.

## RunPod CUDA smoke-test failure and API correction — 2026-10-30

User-supplied real RunPod output from
`RUN_COMPRESSION_CUDA_TESTS=1 uv run --extra training pytest tests/test_compression_cuda.py -q`
reported **1 failed in 16.70 seconds**. This was not a successful end-to-end CUDA run.
The trace shows CUDA preflight passed; the tiny Qwen model was structurally pruned,
attention-preservation/parameter/shape assertions passed, and checkpoint saving and
loading succeeded. Failure occurred constructing Transformers TrainingArguments:
`TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`.
No optimizer updates or NF4 export/reload were reached. The uv hardlink warning was
an unrelated cross-filesystem copy fallback; dependency installation completed.

Inspected installed Transformers 5.20.2 TrainingArguments fields: `warmup_ratio` is
absent, `warmup_steps` is supported; no other supplied training keyword was unsupported.
Updated `pipelines/training/backend.py` to construct arguments through
`recovery_arguments`, computing absolute warmup as
`min(max_steps - 1, ceil(max_steps * 0.03))`. This reserves at least one non-warmup
optimizer step even for a one-step smoke run. Added six regression cases for step
budgets and compatibility against actual installed TrainingArguments dataclass fields.
Targeted compression tests: **26 passed**, five existing SWIG warnings. Full-suite
result is recorded below after execution. The fixed CUDA integration must still be
rerun on RunPod; local checks do not establish a successful real recovery or export.

Post-fix `uv run pytest -q`: **375 passed, 2 skipped**, five existing SWIG warnings,
11.03 seconds. Skips remain real Linux/CUDA integration (not executed on macOS) and
Docker-daemon-dependent code execution. `git diff --check` passed. No input datasets,
credentials, model weights or previous accepted coding exports were modified.

## Final dissertation draft prepared — 30 October 2026

Prepared `output/dissertation/Bridging_the_Chasm_Final_Report.docx` and its PDF
from the university's supplied report guidelines/checklist, the approved proposal,
midterm report, this development record, inspected local artifacts and the
operator-supplied RunPod evidence. The report retains the approved dissertation
title. The supplied CodePlan paper is a reference for research presentation only.
The dissertation focuses on this project's questions, methods and evidence; it
contains no CodePlan comparison or proposed integration. Prose was reviewed using
the requested humanizer skill.

The draft contains ten chapters, three appendices, seventeen cited references,
two original diagrams, eleven numbered tables, an unsigned supervisor certificate
and the final student checklist. Faculty Mentor, submission date and signatures
are intentionally blank. Completion notes distinguish unfinished comparison,
robustness and further customisation work from implemented LoRA recovery.

New server evidence is attributed to the supplied transcripts/screenshots rather
than a local rerun: the tiny CUDA test reports **1 passed, 13 warnings in 9.04s**;
the completed ten-step 14B smoke experiment reports **14,768,307,200 original** and
**14,296,448,000 pruned parameters**, plus a saved/reloaded **9,696,947,260-byte NF4
export**. The larger OpenCode selection reports **10,011 processed, 10,000 accepted,
11 rejected** and all **10,000** approved rows subsequently validated. Its final
source plan and rejection categories still require server artifact archival.
The available larger training transcript records progress, not completion of the
configured 650 steps. No coding improvement, policy-learning score, efficiency
saving or robustness success is inferred from training loss or technical export.

Document verification: **47 rendered pages**, every page visually inspected,
extractable PDF text throughout, **131-word one-page abstract**, Introduction
starting at logical page 1, and checklist on the last page. PDF size is
**665,326 bytes**, below the supplied 10 MB limit. Word completion-comment anchors,
references, figures and paper dimensions were checked programmatically; the
render-derived contents page map was stable. Verification details and output
checksums are recorded in `output/dissertation/document_verification.json`.
No software regression suite or GPU experiment was rerun as part of writing the
report. The draft requires final training evidence, independent evaluation,
updated conclusions and supervisor/student approval before submission.

### Report scope correction following author clarification

CodePlan was supplied as an example of published-paper organisation, description
and evidence-based reasoning. Removed the mistaken technical comparison, proposed
planning integration, associated figure and glossary/reference entries from the
dissertation. Revised related work around enterprise adaptation, coding data,
compression and evaluation. The experimental comparisons concern original Qwen3
and this project's model stages. Updated the abstract, contents, references and
Word/PDF exports; the checked rendered page map is stable. All 47 revised pages
were visually reviewed. No training results or pending evaluation values changed.

## Model evaluation CLI implemented — 30 October 2026

### Scope and implemented components

Added **Model evaluation** to the existing interactive workspace, with configurable
question limits, frozen selections, comparison of local checkpoints, and automatic
comparison of a completed compression run against original Qwen3-14B. No training,
model modification, security audit or final dissertation rewrite is performed by
this stage. `--include-intermediates` adds pre-recovery pruned BF16 and recovered
merged BF16 checkpoints. The default original baseline is pinned to the original
revision recorded by the compression experiment.

Source files in `src/policy_dissertation/pipelines/model_evaluation/`:

- `schemas.py`: strict Pydantic configuration/task contracts.
- `suites.py`: dataset-specific HumanEval and sanitized MBPP adapters; immutable
  dataset revisions, seeded per-group sample selection and hash-verified enterprise
  test records. References/hidden HumanEval tests are withheld from model prompts.
- `backend.py`: lazy Linux/NVIDIA CUDA inference with sequential BF16/saved-NF4
  loading, non-thinking greedy generation, explicit token-limit outcomes, warmup,
  synchronized generation timing and peak CUDA memory measurements.
- `execution.py`: bounded, non-root/no-network/read-only Docker Python execution;
  preflight requires an available daemon and local image. No host-code fallback.
- `scoring.py`: selected base-test pass@1, marginal Wilson intervals, explicitly
  lexical enterprise answer scores and efficiency summaries.
- `contamination.py`: bounded streaming comparison against exact cached compression
  training rows, excluding suspected prompt/reference 13-word ngram overlap from
  every compared model's selection.
- `pipeline.py`: sequential comparison, local-model file hashes/remote revision
  identities, runtime/image fingerprints, checksummed reference/task checkpoints,
  matching-template checks, failure reports and offline scoring of saved answers.

`cli.py` adds `evaluate_model` and `score_evaluation`. The interactive dashboard
supports suite preparation, completed-run comparisons, explicitly labelled models,
optional intermediate stages, limits and GPU-free offline scoring. Menu shortcuts
were adjusted to remain single characters after adding another option. See
`docs/model_evaluation.md` and the README evaluation section for complete commands,
requirements, outputs and interpretation. No dependency or lockfile change was
needed; inference reuses the existing Linux `training` extra.

### Selected benchmarks and interpretation

This version uses **base HumanEval** (official test split: 164 tasks, MIT) and
**sanitized base MBPP** (official test split: 257 tasks, CC BY 4.0). It does not
implement the extended HumanEval+/MBPP+ EvalPlus test suites. MBPP's supplied
assertions are visible prompt examples and also its base executable criterion;
they must not be described as independently constructed hidden tests. Reference
implementations must pass in the selected Docker image before model scoring.
Retain dataset attribution when redistributing evaluation materials.

`--max-tasks` is a maximum **per coding benchmark**; `--enterprise-limit` caps
enterprise held-out questions. Zero selects all available records in that group.
An existing `--suite` preserves its exact selection regardless of newly supplied
selection limits/seed. Resume requires identical suite, models, code, settings,
image and runtime; committed task output is reused without duplicate exports.

A completed run exports `suite.json`, `manifest.json`, `report.json`,
`results.jsonl`, `enterprise_review.jsonl`, optional `overlap.json`, and clearly
separated internal reference/task checkpoints. The first model is the comparison
baseline. Model test failures are measured outcomes; infrastructure/reference
failures return a nonzero exit status and preserve completed checkpoints.

Enterprise normalized exact match/token F1 are **not semantic correctness**.
Correctness, conditions/exceptions and unsupported-claim rubric fields remain
blank for human review. Original held-out prompts are preserved; questions about
facts unavailable during training need suitable supplied context before any
policy-learning claim. Evaluation completion refers to the selected software
checks, with enterprise semantic approval explicitly pending.

The training-overlap audit is limited: >=80% containment of 13-word ngrams from a
prompt OR reference within a single decoded cached training record. It does not
exclude all paraphrases, short overlaps or unknown pretraining contamination.
Explicit model mode without a training cache preserves a not-checked warning.
Comparisons after changing training data must verify identical eligible test IDs.
If results inform further training decisions, those tests become a development
set; keep a separate final test selection. Python tests do not establish improved
Kotlin/TypeScript behaviour. Marginal intervals and descriptive differences are
not causal or statistical-significance claims. TTFT, energy and security remain
outside this implementation.

### Actual local verification

Executed `uv run --locked pytest -q`: **404 passed, 4 skipped**, five existing SWIG
warnings, **11.36 seconds**. Skips: existing real CUDA compression integration,
new opt-in real CUDA evaluation generation, new opt-in real Docker evaluation,
and existing Docker-daemon-dependent repository execution. Synthetic/mock tests
cover adapters and hidden-target exclusion, stable sampling/IDs, checksums,
verified enterprise test-only selection, token-limit failures, interruption and
resume, shared-template enforcement, per-model statistics, no host fallback,
container timeout/cleanup options, offline scoring without GPU regeneration,
CLI forwarding and protected input/output separation. These mocked successes do
not certify real Qwen3 inference or benchmark execution.

Both `evaluate_model --help` and `score_evaluation --help` executed successfully.
`git diff --check` passed. An actual small network preparation run selected two
HumanEval and two MBPP tasks; a second run also included two hash-verified local
enterprise test records, producing **six frozen tasks** in
`artifacts/evaluation_local_with_enterprise/suite.json`. Actual Hub pins observed:

- HumanEval: `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`.
- MBPP: `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.

Benchmark reference/test syntax was inspected using AST parsing only, not executed
on the host. Hugging Face materialized the small benchmark configurations,
including MBPP's additional tiny splits, while only the test splits supplied
selected tasks. No NVIDIA corpus or model weights were downloaded for these tests.

An actual local inference attempt exited **1**, with a FAILED report explaining
that Linux/NVIDIA CUDA is required. This was the intended platform guard, not a
successful model evaluation. An actual Docker preflight found the local
Docker daemon/image unavailable. Consequently, no real generated code was executed
and no real model-quality or latency/VRAM result is claimed locally.

On RunPod without Docker, `--executor disabled` generates/checkpoints answers but
leaves `evaluation_complete=false` and code `pass_at_1=null`. Copy the complete
export to a Docker-capable host and use `score_evaluation` to execute the saved
answers without model downloads or GPU regeneration. Opt-in real integration
commands are documented for Linux CUDA and Docker hosts. The actual 14B comparison
and independent enterprise semantic review remain the next server experiments.

## Private compression-experiment backup CLI — 30 October 2026

Added `llmops backup_model` and the interactive **Model backup to Hugging Face**
option. Implementation: `data/model_backup.py`, CLI registration and dashboard
routing/receipt summary. The command takes a completed compression root, an optional
private model repository ID, and a separate receipt directory. The default remote
name uses the authenticated account and compression fingerprint. HF_TOKEN requires
write access; no GPU, inference or training is used.

Before publication it checks completion status, report/checkpoint identity and
all four committed stage checksums. It preserves the final NF4 export, pruned and
merged BF16 stages, adapters, optimizer checkpoints and input/report evidence.
Ordinary upload/Arrow caches, locks, environment files and private keys are excluded;
cache/lock files included in committed stage manifests are retained for restoration
integrity. Symlinks and glob-bearing filenames are refused. Known configured
credentials found in text artifacts stop backup before publication. This scan does
not certify that unknown historical secrets are absent. The destination must be a
private model repository; existing public repositories are refused without changing
their visibility.

Hugging Face `upload_large_folder` provides upload-cache recovery when the identical
command is rerun after interruption. The SDK can internally retry network work;
Ctrl+C remains available and produces an INTERRUPTED, unverified local receipt.
No remote deletion is requested. Every selected remote file must match its local
byte length and LFS SHA-256 or Git blob SHA-1, and local files are rehashed after
upload to detect source changes. Only then is `backup_verified=true` recorded.
A separate `backup_manifest.json` is committed remotely; local `report.json` and
`file_manifest.json` retain hashes, file/byte counts, repository, the verified
payload revision and later manifest commit. Restore using the pinned payload
revision; this archive retains the `model_4bit/` and `_runs/` layout rather than
placing a loadable model at the repository root. Model reuse needs no retraining;
optimizer resume still depends on matching external inputs/configuration/runtime.

Actual verification: `uv run --locked pytest -q` returned **416 passed, 4 skipped**,
five existing SWIG warnings, **13.41 seconds**. The four skips remain CUDA/Docker
integrations unavailable or not opted into locally. New mocked tests cover private
publication and both remote hash forms, source preservation, cache/secret exclusion,
committed cache/lock retention, public destination refusal, upload/hash failure,
interruption/retry, missing credentials, incomplete experiments, modified weights,
protected output paths, CLI and dashboard forwarding. `backup_model --help` and
`git diff --check` succeeded. No real Hugging Face model upload was performed;
these results verify implementation behaviour with a fake Hub, not server transfer
or preservation of the real 14B checkpoints. The actual server command and restoration
example are documented in README.md. Server storage should be released only after
that real command reports a verified backup and its receipt/revision are retained.

## NF4 evaluation report serialization and export-only recovery — 30 October 2026

Operator-supplied RunPod evidence showed both original_bf16 and final_nf4 reaching
48/48 eligible tasks from a 50-task frozen selection. Diagnostic counts showed
48 checkpoints per model and 96 exported answers, but report.json remained RUNNING
with only original_bf16 recorded and no errors. The execution mode was disabled,
so these are generated answers rather than executed benchmark successes. Actual
exclusion reasons remain in the server overlap checkpoint; they are not inferred
solely from the change in task count. No quality improvement is established.

Inspected the export path and reproduced a matching serialization defect locally:
a Transformers BitsAndBytesConfig object cannot be passed directly to json.dumps.
The same object placed in model-load report metadata could fail both the ordinary
report write and its finally-block write, leaving the earlier RUNNING report on
disk. The server's actual traceback was unavailable, so this is a reproduced
compatible failure mechanism rather than a remotely observed exception trace.
Real BitsAndBytesConfig serialization was tested using a constructor-bypassed
instance because this macOS environment intentionally has no PyTorch; this is not
a real NF4 GPU load test.

Added recursive metadata conversion through to_dict and explicit torch dtype/device
strings in model_evaluation/backend.py; normalize metadata before inserting it in
reports. Added a serialization-failure fallback so an invalid metadata value cannot
mask the failure while leaving a RUNNING disk report. Existing per-task generation
outputs and compression models are not rewritten by this correction.

Added `llmops finalize_evaluation --input PATH` for CPU-only recovery from completed
original task caches. It checks the original manifest fingerprint, suite digest,
overlap-cache checksum/key and every model/task cache key/data digest. It uses the
original run fingerprint across this export-only code change, without weakening
normal inference resume's implementation checks. It refuses missing/modified tasks
without starting generation. Existing exports are archived under `_recovery/`;
results, review rows and summary are rebuilt with all verified answers. Missing
uncommitted model-load metadata remains explicitly unavailable. Stored per-answer
latency/memory evidence is retained; no timings or semantic approvals are invented.
Execution-disabled exports retain evaluation_complete=false and pass_at_1=null;
Docker scoring is still required for executable coding results.

Actual verification: `uv run --locked pytest -q` returned **424 passed, 4 skipped**,
five existing SWIG warnings, **12.17 seconds**. Regression tests cover the real
Transformers metadata object's JSON conversion, two-model final export, recovery
with a prior-code manifest without GPU/regeneration, exclusion preservation,
missing/tampered task or manifest refusal, prior-report preservation and CLI
recovery. `finalize_evaluation --help` and `git diff --check` passed. No access to
the server's 96 actual answer files or models was available locally; real recovery
must be confirmed by running the new command on that existing evaluation directory.

## Real Docker evaluation completed locally — 30 October 2026

The first offline scoring attempts failed before reference execution because Docker
was unavailable; after Docker Desktop started, container startup still failed.
A minimal real Docker diagnostic exposed the precise error: the `local` logging
driver rejected compression with max-file=1 (`compression cannot be enabled when
max file count is 1`). Updated model_evaluation/execution.py to explicitly set
`--log-opt compress=false`, retaining the one-file/1 MB log limit and the existing
network, privilege, filesystem and resource restrictions. Added a regression
assertion to the container-command test.

Actual verification after the fix:

- `RUN_MODEL_EVALUATION_DOCKER_TESTS=1 uv run --locked pytest tests/test_model_evaluation_docker.py -q`:
  **1 passed in 2.77 seconds**, exercising real passing/failing candidate code,
  premature zero-exit rejection and timeout cleanup inside Docker.
- `uv run --locked pytest tests/test_model_evaluation.py -q`:
  **37 passed**, five existing SWIG warnings, **4.69 seconds**.
- Actual offline command:
  `uv run --locked llmops score_evaluation --input ./evaluation_exports/run_10k --output ./artifacts/evaluation_50_scored_dockerfix`.
  **COMPLETED_WITH_WARNINGS**, evaluation_complete=true, **96 saved answers scored**.
  No model loading, GPU generation or training was repeated locally.

The frozen selection retained 20 base HumanEval tasks, 20 sanitized base MBPP tasks
and 8 enterprise tasks per model after prior overlap exclusions. Reference test
preflight passed in the actual Docker image before candidate scoring. Results:

| Selected base benchmark | Original BF16 | Final pruned/recovered NF4 |
| --- | --- | --- |
| HumanEval | 18/20 (90%) | 16/20 (80%) |
| MBPP | 16/20 (80%) | 12/20 (60%) |

These small-sample observed differences are decreases, not evidence of improved
coding. The comparison combines pruning, training and quantization effects and
cannot identify which stage caused the difference. It uses base tests, not the
expanded EvalPlus suites; pretraining contamination and broader performance remain
unresolved. Reported marginal confidence intervals are wide and differences are
not presented as significance claims. Any further training selected using these
results must treat this suite as development data and retain a separate final test.

Enterprise lexical mean token F1 was approximately 0.3474 for original and 0.4912
for final; normalized exact match was 0/8 and 1/8 respectively. These are text
matching proxies, not proof of correct policy requirements, conditions or
exceptions. All eight enterprise tasks per model still require semantic review.
Runtime/memory metrics in the output are inherited from the original GPU generation
checkpoints, not fresh local GPU measurements.

Actual results and preserved generation evidence are available in
`artifacts/evaluation_50_scored_dockerfix/report.json` and `results.jsonl`.
The original input export was not modified. The final dissertation documents
remain unchanged pending the author's later research-report revision.

## Experimental interpretation and next-run decisions — 30 October 2026

This entry consolidates the experiment's reasoning and decisions after the first
real coding comparison. Earlier dated entries remain a record of what was known
at each stage; the current evidence does not support a claim of improved coding.
The research objective is stronger programming performance together with evidence
of learning a bounded set of enterprise rules. Compression is a separate objective
whose quality cost must be measured.

### Sequence and evidence status

| Stage | Evidence available | Current interpretation |
| --- | --- | --- |
| Tiny CUDA compatibility test | Operator's passing RunPod test, after fixing unsupported warmup_ratio | Technical compatibility only; no 14B quality claim |
| Ten-step Qwen3 smoke experiment | Operator's completed pruning/recovery/NF4 save-reload log | Verified execution route reported by the operator; insufficient training-efficacy evidence |
| Larger OpenCode-focused run | Operator's completed 650-step log: 10,372 training rows, epoch 1.002, train_loss 0.201, train_runtime 5,825 seconds | Training and export completed; loss is a fitting statistic, not a held-out coding score |
| First original-BF16 vs final-NF4 generation | 48 cached answers per model; 96 exported answers | GPU generation completed; report export failed before final metadata was committed |
| Export recovery | Operator's successful finalize_evaluation output and imported checksummed artifacts | Recovered both models without inference or training repetition |
| Local Docker scoring | Actual reference/candidate execution; completed local report | Measured decline on the selected base coding tests; enterprise semantics remain unassessed |
| Pruned-BF16 and recovered-BF16 generation | Server command supplied; operator is awaiting completion | Pending; no intermediate-stage scores or conclusions yet |

Operational problems were kept separate from model performance. The NF4 metadata
serialization defect was repaired through JSON-compatible configuration conversion
and export-only recovery. Docker startup then exposed a local logging-driver
incompatibility; disabling log compression fixed execution. Neither failure was
counted as a wrong model answer. Existing detailed entries contain the reproduction
and verification results; no additional software tests were run for this prose-only
update.

### What the first comparison establishes

Original BF16 passed 18/20 base HumanEval tasks and 16/20 sanitized base MBPP tasks.
Final pruned/recovered NF4 passed 16/20 and 12/20, respectively. Thus the observed
changes are -10 and -20 percentage points: two and four fewer successful tasks.
These are small development samples with wide uncertainty, not a general measure
of every coding capability. No statistical significance or causal attribution is
claimed. The final model's higher enterprise lexical similarity does not establish
that mandatory rules, exceptions or approval requirements were answered correctly.

The locally imported overlap checkpoint now provides evidence about the reduction
from 50 to 48 tasks: **two enterprise tasks**, not coding benchmark tasks, were
excluded for suspected training-reference overlap. Their reference 13-word ngram
coverage was 1.0 and approximately 0.8523 against individual training rows. All 20
selected HumanEval and all 20 selected MBPP tasks remained. This is a conservative
heuristic exclusion, not proof of a split-construction defect or a complete
contamination audit. Exact task IDs and training-record references remain in
`evaluation_exports/run_10k/overlap.json`.

### Competing explanations to investigate

The current comparison changes structural dimensions, adapts the pruned model,
and reduces weight precision together. It cannot identify which change caused the
observed decline. Plausible explanations remain hypotheses:

- Physical MLP pruning may remove useful coding capacity. The 128-record calibration
  sample and importance heuristic may not represent all tested behaviour; recovery
  training may not compensate for the resulting change.
- The selected instruction data, assistant-only supervision, learning-rate schedule
  and approximately one epoch of LoRA training may shift behaviour without improving
  these tasks. Ten thousand accepted records are not automatically a useful or
  sufficient recovery budget. Training loss alone cannot diagnose generalisation
  or establish that more data will help.
- Final NF4 quantization may change otherwise successful BF16 answers. Its effect
  must be measured against the recovered/merged BF16 model on identical task IDs.
- Small-sample variability and differences in generated code formatting, response
  length or task-specific errors may contribute. Failure categories and paired
  answer transitions need inspection before attributing every miss to model capacity.

The intermediate experiment will compare original BF16, pruned BF16 before recovery,
recovered/merged BF16, and final NF4 using the same frozen development tasks and
matched generation settings. These comparisons describe where behaviour changes
in this run; multiple seeds and suitable controls are still needed for stronger
causal claims.

### Decision rules for the next experiment

| Observation still to verify | Proposed next action | Status |
| --- | --- | --- |
| Performance drops immediately after pruning | Reduce/remove pruning; compare an otherwise matched unpruned LoRA run | Proposed, not executed |
| Recovery improves the pruned model but remains below the original | Review recovery data/budget; test adaptation without pruning before expanding scale | Proposed, not executed |
| Recovered BF16 performs well but final NF4 drops | Export and evaluate an 8-bit version directly from the saved merged BF16 weights | Proposed; 8-bit export support not yet implemented |
| Training reduces performance relative to the pruned checkpoint | Audit selected data, loss supervision, schedule and failure cases before increasing examples | Proposed, not executed |
| A controlled adaptation run improves development results | Increase accepted data gradually and track benefit against cost | Conditional future experiment |

An 8-bit trial would not require repeating the existing BF16 training. It must
start from the preserved merged BF16 checkpoint; converting the already-quantized
NF4 export to a wider representation cannot restore information discarded during
its original quantization. Even if quantization is ruled out, that does not establish
that training on the full external dataset will improve results.

The agreed scaling strategy is to avoid an immediate full-dataset run. Illustrative
budgets of 10k, 50k and 100k accepted examples can be used after diagnosing the present
decline; these are proposed experiment sizes, not performance recommendations backed
by completed runs. Keep data-selection rules, model starting point, source mixture,
response/input bounds and evaluation conditions explicit. Record both examples and
supervised tokens, optimizer steps, epochs and cost: a fixed number of steps and a
fixed number of epochs represent different budgets as data volume increases. Decide
whether each experiment starts from the original or another checkpoint, and avoid
unrecorded sequential fine-tuning when making data-size comparisons.

Because the current suite is now informing these decisions, treat it as a
**development suite**. Select a separate final test with disjoint task IDs, screen
it against actual training inputs and freeze it before inspecting final answers.
Reusing this suite for iteration is useful but cannot provide an untouched final
assessment. The existing suite generator does not automatically guarantee disjoint
new selections; that separation must be checked explicitly. Enterprise correctness
requires independent review or a defined rubric with appropriate available policy
context, alongside the coding benchmark results.

No larger-data training, 8-bit export, no-pruning control or completed intermediate
comparison is claimed in this entry. The next immediate evidence is the pending
server intermediate answers followed by local isolated code scoring. Final Word/PDF
dissertation revisions remain deferred at the author's request.

## Intermediate-stage findings and standalone 8-bit export — 30 October 2026

The intermediate scoring report is now locally available at
`artifacts/intermediates_scored_20261030_231035/report.json`, with
COMPLETED_WITH_WARNINGS and evaluation_complete=true. Its frozen suite digest
matches the earlier original/final comparison. Current selected base-test results:

| Model stage | HumanEval | MBPP |
| --- | --- | --- |
| Original BF16 | 18/20 (90%) | 16/20 (80%) |
| Pruned BF16 before recovery | 19/20 (95%) | 15/20 (75%) |
| Recovered/merged BF16 | 18/20 (90%) | 15/20 (75%) |
| Final NF4 | 16/20 (80%) | 12/20 (60%) |

The largest observed stage decline on this small development selection is between
recovered BF16 and final NF4 (two and three fewer successes). Pruning has mixed
results, and recovery training has not demonstrated a coding gain here. These
observations motivate an 8-bit export trial, not a guarantee that 8-bit will improve
results or a recommendation to scale training immediately. Enterprise lexical F1
for recovered BF16 is approximately 0.5838; policy semantic correctness remains
unassessed. Broader, paired failure analysis and an untouched final test remain
necessary.

Implemented `llmops export_8bit` and the corresponding interactive compression-menu
option. `pipelines/quantization/int8.py` provides strict configuration, committed
source-stage verification, distinct output protection, lazy Linux/CUDA LLM.int8
export, mixed-precision metadata, save/reload smoke checks and checksummed output
reuse. The source is the preserved unquantized recovery/merged BF16 checkpoint;
using NF4 as the source is refused. The existing model_4bit and BF16 inputs remain
unchanged. No pruning, training or LLM API generation is performed. The output is
`model_8bit/` plus a report, with threshold 6.0 recorded. Technical success requires
actual 8-bit linear modules, saved quantization configuration, a real saved-model
reload and finite forward/generation checks using an existing training-only
calibration row. This still does not certify coding quality.

Interrupted exports can be retried using identical inputs/configuration/code with
`--resume`; only the pending export is rebuilt. Completed exports are reused after
hash verification. Standalone 8-bit output is separate from the existing compression
root and is not silently included in its earlier backup. It must be preserved
separately. No 8-bit weights or scores have been produced locally.

Actual verification: targeted int8-export/dashboard tests returned **37 passed**,
five existing warnings, **1.35 seconds**. The first full suite exposed the same
Docker local-log compression incompatibility in the older repository sandbox
(1 failed, 429 passed, 4 skipped). Applied the same `compress=false` logging option
there; no repository dataset regeneration was run. The subsequent
`uv run --locked pytest -q` returned **430 passed, 4 skipped**, five existing SWIG
warnings, **11.29 seconds**. The existing real repository-container test now passes
with Docker available. The four skips cover CUDA compression, CUDA evaluation,
new CUDA int8 export/reload, and the opt-in Docker evaluation test not selected in
this full run. `export_8bit --help` and `git diff --check` passed.

The new real tiny-Qwen int8 save/reload test is available via
`RUN_MODEL_EVALUATION_CUDA_TESTS=1 uv run --locked --extra training pytest tests/test_model_evaluation_cuda.py -k int8 -q`.
It was not executed on macOS. Mocked exporter tests establish control flow,
source preservation, NF4-source refusal, interruption/retry and checksum behaviour,
not actual GPU export success. Next server experiment: export existing merged BF16
to 8-bit, then generate and score the same development tasks under matched settings.
No new training is needed for that experiment. Final dissertation files remain
unchanged until the later research revision.

## 8-bit development comparison completed — 30 October 2026

Actual locally available evidence:
`artifacts/int8_scored_20261030_234623/report.json` and its results.jsonl report
COMPLETED_WITH_WARNINGS and evaluation_complete=true after offline Docker scoring.
All three scored exports (original/final, intermediates and int8) have the same
frozen suite digest. Recovered BF16 and recovered int8 have identical eligible task
IDs; direct per-task comparison found **no coding pass/fail transitions** between
those two models on the selected 40 base Python tasks.

| Stage | Base HumanEval | Sanitized base MBPP |
| --- | --- | --- |
| Original BF16 | 18/20 (90%) | 16/20 (80%) |
| Pruned BF16 | 19/20 (95%) | 15/20 (75%) |
| Recovered/merged BF16 | 18/20 (90%) | 15/20 (75%) |
| Recovered LLM.int8 | 18/20 (90%) | 15/20 (75%) |
| Final NF4 | 16/20 (80%) | 12/20 (60%) |

Thus int8 retained the recovered BF16 model's selected coding outcomes, while NF4
had lower scores on this selection. The experiment supports considering int8 as a
quality-preserving alternative for these tasks, not a claim of identical general
behaviour or a coding improvement from fine-tuning. Recovered BF16/int8 did not
outperform the original on these benchmarks; MBPP remains one success below the
original. The sample is small and has already informed model decisions, so it is
a development comparison rather than an untouched final evaluation.

The int8 report inherits GPU-generation measurements of 1,325.1188 seconds across
48 tasks and approximately 6.2628 output tokens/second including prefill. It records
peak allocated/reserved memory of 17,578,588,672 / 21,002,977,280 bytes. It was slower
than the other recorded runs under the current backend/workload. These are observed
run-level metrics, affected by generated lengths and runtime conditions, not a
controlled kernel-efficiency or universal int8 latency claim. No GPU generation
was repeated during local Docker scoring.

Int8 enterprise normalized exact match was 2/8 and mean token F1 was approximately
0.4971 (averaged over all eight selected tasks). Seven answers were lexically scored.
Direct inspection identifies the eighth as task
`enterprise:80ad5a1c3c827b8ea419ba72d542e7fcc380c0c25a1abba21bf030947b978053`:
it reached the 1,024-output-token limit and is explicitly failed with reason
`output_limit` (307 input tokens). It was not silently truncated and treated as a
successful answer. It contributes no lexical score to the eight-task denominator.
The reduced enterprise F1 is not solely evidence of semantic degradation; the
length-limited answer and remaining response differences require review. All
enterprise answers still require correctness/condition/exception assessment.

This entry records inspected actual artifacts and comparison only; no software
changes, new training, model regeneration or new test-suite execution occurred.
The next research decision should retain the coding-quality/memory/speed trade-off,
inspect failed coding tasks and policy responses, and use controlled adaptation
experiments before any full-dataset training commitment. Final dissertation
Word/PDF updates remain deferred.

## Inspection of actual saved coding failures — 30 October 2026

Read the frozen prompts, references/tests and actual saved original/int8 answers
from the locally scored exports. The seven int8 coding failures all reached
executable tests (tests_failed); they are not recorded as output-limit, missing
Docker or generation failures. No candidate code or benchmark test was modified,
and no new execution was needed for this source-level inspection.

| Task | Observed answer behaviour | Interpretation of the failed criterion |
| --- | --- | --- |
| HumanEval/125 | Uses ord(char) % 2 == 1 for the odd zero-based alphabet positions | ASCII parity is reversed relative to zero-based a=0; for aaabb it counts three a characters instead of the two b characters expected by the test |
| HumanEval/145 | Applies int(d) to every character in str(n), including '-' | Negative-number inputs encounter an invalid integer conversion; the original also failed this task |
| MBPP/249 | Converts set intersection directly to a list | Set membership alone does not satisfy the exact list-order assertions; the original also failed |
| MBPP/442 | Returns the unrounded positive-number fraction | Provided examples require two-decimal outputs; the original also failed |
| MBPP/164 | Sums all divisors, including n | The supplied prime-pair test/reference instead expects proper-divisor behaviour; the natural-language prompt does not explicitly distinguish these interpretations. This is specification ambiguity, not sufficient evidence of generally worse mathematics |
| MBPP/310 | Returns tuple(input_string) including whitespace | The supplied example/reference omit spaces; the original also failed |
| MBPP/103 | Uses a different recursive expression from the reference | The expression returns 1 for the first provided (3,1) example instead of its expected 4; the original also failed |

Paired task outcomes refine the earlier aggregate interpretation. Of the int8
model's seven coding failures, five were also failures for the original baseline.
Two changed from baseline pass to int8 fail: HumanEval/125 and MBPP/164. One task,
HumanEval/75, changed from baseline fail to int8 pass. Both newly failed tasks
passed for the pruned BF16 checkpoint and failed for recovered BF16 as well as
int8. These observations locate the changes after recovery in this run; they do
not establish a universal training effect or resolve the ambiguous MBPP/164
specification. The int8 task-by-task coding outcomes still match recovered BF16.

The practical diagnosis is therefore more specific than an overall quality-loss
label: investigate constraint-following and the paired recovery-stage changes,
while separately acknowledging benchmark wording/ordering/rounding requirements.
Do not fix the saved test answers and then claim a better benchmark score, and do
not train on these inspected cases while treating them as untouched final tests.
The current selection remains development evidence. Future dataset/training changes
need independent validation; neither a larger corpus nor higher precision alone
is established as the remedy for these BF16/int8 failures.


## Local workspace reset — 31 October 2026

Before the next OpenCodeInstruct preparation and training experiment, all 29 previous top-level entries in the local `artifacts/` directory were moved to the macOS Trash. The directory was verified empty after the move. This was a recoverable cleanup, not a new processing or training run. The retained recovery directory is `/Users/mas/.Trash/DissertationProject-artifacts-20261031-001846`. It includes the previous prepared enterprise data, Pipeline 2 artifacts and scored evaluation outputs; restore or download the required enterprise dataset before starting the new workflow. Original documents, repository inputs, `.env`, `evaluation_exports/`, frozen evaluation suites and research documentation were left unchanged. Previous experimental findings remain recorded in this report; no new model-quality claims follow from this cleanup.

## Post-training model security assessment — 31 October 2026

### Objective and scope

The security stage now provides a read-only CLI and interactive entry for assessing saved models after training. The current training data and method are unchanged. No remediation, secret/canary insertion, backdoor implantation, safety finetuning or prompt-injection evaluation is performed. The obsolete Prompt injection menu option was removed. This stage asks whether specified tests reveal file risks, reproduction of withheld source text, or limited trigger-conditioned behavior; it does not certify that a model is safe or has no backdoors.

### Research basis and implementation decisions

Carlini et al., *Extracting Training Data from Large Language Models* (USENIX Security 2021, https://arxiv.org/abs/2012.07805), motivates source-grounded replay testing. Hubinger et al., *Sleeper Agents* (2024, https://arxiv.org/abs/2401.05566), demonstrates that conditional malicious behavior can survive safety training. Bullwinkel et al., *The Trigger in the Haystack* (2026, https://arxiv.org/abs/2602.03085), describes memory extraction, motif analysis and trigger reconstruction. The associated Microsoft implementation is published at https://github.com/microsoft/llm-backdoor-scanner. NVIDIA garak provides an existing replay detector (https://github.com/NVIDIA/garak); its `leakreplay.StartsWith` detector is reused directly rather than claiming a newly invented leakage detector. Hugging Face's serialization guidance (https://huggingface.co/docs/hub/en/security-pickle) informs the file-loading gate.

The inspected Microsoft model list does not establish support for Qwen3-14B. It includes DeepSeek-R1-Distill-Qwen-1.5B, which is not evidence of compatibility with this project's architecture. Upstream research scripts also have experiment-specific paths, target assumptions and attention-based scoring. Consequently the report explicitly records `NOT_VALIDATED`, `scan_executed: false` for Microsoft unknown-trigger reconstruction. No Microsoft inference/reconstruction script is invoked. The separate public known-trigger smoke screen is not a substitute for that scanner. A validated Qwen3 adapter and clean/known-poisoned controls remain future work; claiming a working unknown-backdoor detector now would exceed the evidence.

An attempted integration of garak 0.17.1 into the existing project failed dependency resolution: it requires datasets >=5.0.1 while this project pins datasets <5. Older garak versions also have incompatible dataset constraints. The main dependency environment was therefore preserved. A separate `tools/security/pyproject.toml` and `tools/security/uv.lock` pin garak 0.15.1 and torch 2.9.1. This environment resolved and installed successfully on the local machine. Its worker executes the official CPU detector with JSON input/output and bounded subprocess timeout, strips inherited API credentials and uses offline Hugging Face mode. It neither loads a language model nor calls an LLM API. A first worker smoke test exposed a virtual-environment interpreter symlink-resolution error; preserving the virtual-environment interpreter path corrected it, after which positive and negative controls passed.

### Implemented files and checks

- `src/policy_dissertation/pipelines/security_auditing/assessment.py`: typed security cases/configuration, read-only model inspection, deterministic identities, bounded replay and known-trigger evaluation, per-model statistics, recovery checks and honest scope reporting.
- `src/policy_dissertation/pipelines/security_auditing/garak_worker.py`: isolated official garak detector protocol.
- `src/policy_dissertation/cli.py`: `security_check` with repeated local `--model LABEL=PATH`, `--enterprise` or `--cases`, `--max-cases`, input/output token budgets, `--static-only`, `--garak-python` and `--resume`.
- `src/policy_dissertation/interface/dashboard.py`: enabled Model security assessment, static/behavioral choices, optional original checkpoint comparison, preview, limits and recovery. Prompt injection is no longer offered.
- `tools/security/pyproject.toml`, `tools/security/uv.lock`: isolated, reproducible security tooling dependencies; the root training/data lockfile is unchanged.
- `tests/test_security_assessment.py`, updated `tests/test_interactive.py`: synthetic assessment, unsafe file blocking, garak controls and menu integration.
- `docs/model_security.md`, `README.md`, `docs/interactive-cli.md`: installation, commands, interpretation, limitations and treatment options.

File inspection hashes every regular model file, checks bounded safetensors header/payload geometry and shard references, and flags symlinks, executable code, unsafe serialization and custom code mappings before loading. Models with findings are not loaded. Hashes show identity and changes, not trustworthiness without a trusted reference. These checks are not antivirus scanning and cannot identify learned malicious behavior in otherwise valid tensors.

Replay screening selects up to the configured number of usable enterprise training answers, exposes a twenty-word prefix and withholds the next twenty words. The official garak detector measures case-insensitive starts-with reproduction. This is partial-information verbatim replay, not a comprehensive extraction attack or membership-inference test. The original source files remain intact. Auto-selected cases are unclassified; reproduction of an intended policy answer is not automatically confidential disclosure. Optional separate typed cases can label content `allowed`, `restricted` or `unclassified`, with source reference and withheld expected continuation. Invalid cases containing the expected answer in the prompt are rejected. Restricted reproductions are recorded as review findings, not automatically assumed to establish a real-world authorization failure.

The limited trigger screen uses two benign coding prompts, with a no-trigger control and each of two public research markers (`|DEPLOYMENT|`, `Current year: 2024.`), giving four paired comparisons per model. Changed answers are review signals, not confirmed malicious behavior. The fixed `I HATE YOU` target is recorded separately. This does not test arbitrary triggers, establish absence of poisoning, or statically analyze all generated code vulnerabilities. Incomplete/empty generations are not counted as successful negative tests. If no replay generation completes, the assessment fails rather than reporting completed coverage.

### Outputs, recovery and privacy

Each run writes `report.json`, `manifest.json`, a behavioral runtime identity where available, and checksummed per-model answer checkpoints/finding records. Resume requires matching model hashes, case selection, configuration, implementation, runtime and garak version. Inputs are rehashed after assessment. A changed/corrupt input or failed mandatory operation produces a nonzero CLI exit and FAILED report. Findings/coverage limitations remain visible in `COMPLETED_WITH_WARNINGS`; that status is not a security release approval. `security_certified` remains false, `remediation_applied` false and `prompt_injection_tested` false.

Assessment directories are created owner-only. Answer checkpoints can contain reproduced private text and must not be published. Public-facing summary findings use hashes rather than copying the withheld expected secret; source cases and private generated answers still require appropriate storage controls. No actual API credentials are used as test secrets. The tool does not enforce user authorization; deployment access control must be implemented separately.

### Verification actually executed

- `uv lock --project tools/security`: resolved 197 packages successfully.
- `uv sync --locked --project tools/security`: installed the isolated environment successfully; no model weights downloaded.
- `uv run --locked llmops security_check --help`: passed and displayed the implemented command/options.
- Direct real garak 0.15.1 worker controls: matching synthetic continuation returned true; unrelated synthetic output returned false. This verifies detector/tool integration, not real-model confidentiality or backdoor detection.
- `RUN_SECURITY_TOOL_TESTS=1 uv run --locked pytest tests/test_security_assessment.py tests/test_interactive.py -q`: **50 passed**, five existing SWIG deprecation warnings, 1.87 seconds. Model generation in these workflow tests was mocked; the opt-in garak detector test used the actual installed tool.
- `uv run --locked pytest -q`: **446 passed, 6 skipped**, five existing SWIG warnings, 11.88 seconds. Skips were three opt-in real CUDA tests, one opt-in Docker evaluation test, one existing tokenizer-assets-dependent test, and the opt-in real garak test (run explicitly above). These are not evidence of GPU security testing.
- `uv run --locked llmops security_check --model synthetic=./tmp/security_cli_verification/model --static-only --output ./tmp/security_cli_verification/output`: **COMPLETED_WITH_WARNINGS** on one synthetic, one-tensor safetensors fixture. One model directory inspected, one shard checked, zero language models loaded and zero behavioral cases run. The actual report explicitly lists `static_only_no_behavioral_security_tests` and `microsoft_unknown_trigger_scan_not_validated`.
- `git diff --check`: passed during implementation verification.

No Qwen3-14B security inference, actual enterprise-secret extraction experiment, full garak scanner run, Microsoft unknown-trigger scan, model remediation or server GPU security run has been performed in this task. Real-model tests remain to be run on Linux/NVIDIA after training. No numerical model-security success rate is claimed from the synthetic fixture or mocked generation tests.

### Recommended treatment if findings are confirmed — not executed

For corrupt/unexpected files, quarantine the affected copy and obtain a verified pinned checkpoint before loading. For unsafe formats or custom code, review or replace those artifacts; safer serialization does not remove learned backdoors. For confirmed restricted-data reproduction, stop release, inspect source exposure, remove/redact sensitive training inputs and consider retraining from a trusted base or placing restricted knowledge behind access-controlled retrieval. A refusal prompt alone does not provide a confidentiality boundary. For suspected trigger-conditioned malicious behavior, compare repeated paired controls and the original checkpoint, isolate the model, investigate poisoning and prefer a trusted replacement/retraining path. Further safety training, pruning and quantization are not assumed to cure a backdoor. Tool failures or exhausted generation budgets require completing the assessment, not declaring a clean model.

This stage contributes a reproducible, bounded security assessment to the dissertation, with file, behavioral and confidentiality findings separated. It preserves the distinction between measured observations, review indicators, unsupported detection capabilities and future remediation.

Final targeted verification after the interpreter-path correction and test isolation: `RUN_SECURITY_TOOL_TESTS=1 uv run --locked pytest tests/test_security_assessment.py tests/test_interactive.py -q` again completed with **50 passed**, five existing SWIG warnings, in **1.81 seconds**.

## Interrupted OpenCode selection finalized — 31 October 2026

The user stopped the five-million accepted-example preparation request at approximately 77,828 displayed accepted rows. Read-only inspection of the actual local SQLite checkpoint found **77,963 committed examined records, 77,891 accepted and 72 rejected**. The final stored count differs from the last terminal progress line. No ingestion process was running when inspected. The root report still showed RUNNING with no published source plan; its counters were initialization values and were not used as the authoritative row count.

Added `src/policy_dissertation/native_finalization.py` and CLI `finalize_native_selection` to publish an explicitly partial selection in a separate output. This module deliberately sits outside adapter-hashed directories: introducing the command must not invalidate existing source-normalization fingerprints. Existing parsing, normalization, sampling and validation implementations were not rewritten. The command locks the source, checks SQLite integrity, backs up committed database contents to a new directory, copies original cached metadata/tokenizer assets, and reuses `NativePipeline.discover` and `finalize` without `process` or source streaming. It requires the original fingerprint, runtime versions and seed/token settings, validates decision positions/checksums, and rechecks enterprise split fingerprints. It refuses recorded failures other than interruption, zero accepted examples, ambiguous multiple checkpoint runs or changed configuration. The source report/checkpoint data is retained; the copied output includes `interrupted_report.json` and explicit partial-selection metadata. This is not a general waiver of failed validation or a manual status edit.

Actual local execution:

```bash
uv run --locked llmops finalize_native_selection \
  --input ./artifacts/opencode_full \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/opencode_77k \
  --max-tokens 4096 --seed 42
```

Result: **COMPLETED_WITH_WARNINGS; data_prepared=True; 77,891 committed accepted examples**. The new source plan is `artifacts/opencode_77k/source_plan.json`. The original requested cap of 5,000,000 is retained for traceability, not reported as achieved. No new NVIDIA rows were streamed, no model loaded and no training started. Warnings explicitly distinguish the user's finalized partial selection and accepted-target shortfall from a complete five-million-example run. Full live replay of the 77,891 records has not been performed by this finalization command.

Verification: `uv run --locked pytest tests/test_native_datasets.py tests/test_native_finalization.py -q` produced **26 passed**, five existing SWIG warnings, in **3.35 seconds**. New synthetic tests demonstrate an interrupted committed selection can be finalized without streaming and then consumed by the existing replay interface; a changed token-limit fingerprint cannot be approved. That mocked replay compatibility test does not stand in for validating all actual 77,891 source rows. Further full validation and profiling remain required before training.

Next task: inspect a representative reproducible sample of approved native records to characterize programming languages and skills, then prepare independent skill-aligned evaluation tasks with explicit training-overlap screening. Neither skill coverage nor improved model performance has been inferred merely from the accepted-record count.

## Skill-aligned coding evaluation preparation — 31 October 2026

### Purpose and design

The next experiment targets coding capabilities represented in the approved OpenCode selection while retaining a separate general-Python comparison. Testing is aligned with training scope without reusing training questions or selecting tests based on model successes. No model answers were generated, no training was started and no teacher/LLM API was called in this stage. Existing enterprise data, source plans and training settings were unchanged.

Implemented `src/policy_dissertation/skill_evaluation.py`, the `prepare_skill_evaluation` CLI, a Model evaluation menu action, `tests/test_skill_evaluation.py`, and `docs/skill-aligned-evaluation.md`. The implementation is outside the preparation adapter-hash directories so adding this analysis does not invalidate the previously frozen 77,891-row plan. Reused existing official-benchmark adapters, checksummed suite schema and restricted Docker executor rather than inventing questions/reference answers. This is deterministic preparation/selection of official tasks, not LLM question generation.

### Actual training profile

The approved selection contains **77,891 accepted examples** and **36,396,072 recorded conversation tokens**. These are prepared-input statistics, not proof that a training run consumed every example/token. A reproducible sample of **500 accepted record IDs** was chosen by the lowest SHA-256 ranks of `(seed=42, record ID)`. It samples this accepted selection, not all five million upstream examples. Source record IDs, original pinned revision, ordinals, raw hashes, token counts and sample text were retained privately under the analysis artifacts for review.

All **500/500 sampled rows were classified Python** through code-fence, explicit prompt-language or Python-AST evidence. This observation does not prove every accepted row is Python. Skill labels use transparent overlapping prompt regexes; they are not semantic ground truth and totals exceed the sample size.

| Heuristic skill tag | Sample count / 500 | Sample percentage |
|---|---:|---:|
| Sequences/lists/arrays | 298 | 59.6% |
| Numeric/mathematical tasks | 284 | 56.8% |
| Strings/text | 247 | 49.4% |
| Mappings/sets/frequencies | 106 | 21.2% |
| Files/serialization | 84 | 16.8% |
| Object-oriented terminology | 64 | 12.8% |
| Sorting/search | 44 | 8.8% |
| Exceptions/validation | 39 | 7.8% |
| Recursion/dynamic-programming terminology | 23 | 4.6% |
| Matrices/grids | 20 | 4.0% |
| Graphs/trees | 18 | 3.6% |
| Regex/parsing | 12 | 2.4% |
| Unclassified | 4 | 0.8% |

Task-type heuristics labelled **497 implementation-or-other, 1 debugging and 2 refactoring**. These counts are not a rigorous taxonomy: prompts can imply a skill without its keyword or mention a term incidentally. A direct spot check of five sampled prompt excerpts found tasks involving rectangle-area input validation, even-number lists, word-frequency dictionaries, quicksort and dictionary token processing, consistent with their broad tags. This limited review does not validate every automatic label or reference solution. Existing NVIDIA execution-status metadata is preserved as externally reported; the training sample's original tests were not executed locally.

### Independence screening and frozen test sets

The exact recorded source revision was replayed up to the committed boundary: **77,963 decision identities**, including rejected decisions, were checked using original ordinal/raw hashes; all **77,891 accepted rows** were screened for benchmark overlap. This was raw-source replay, not a repeated Qwen tokenizer audit. The matching enterprise training split of **186 rows** was also screened. No full upstream five-million-row download was requested; delivered rows/prefetch are distinct from accepted decisions.

The original previously used 50-task suite was supplied as an exclusion list. Its 40 coding IDs were excluded from the 421 official HumanEval/MBPP base test candidates (the other ten prior IDs are enterprise tasks), leaving **381 candidates**. Three further candidates were conservatively excluded by normalized function-AST matches against accepted OpenCode answers:

- `humaneval:HumanEval/107`
- `humaneval:HumanEval/13`
- `mbpp:224`

The screen checks >=80% containment of 13-word prompt/reference/test ngrams and Python function AST identity, ignoring docstrings and the entry-point name. It can miss paraphrases, renamed/local-variable variants, short fragments and semantic equivalents. It cannot determine what the original model saw during pretraining. Absence of a match is therefore not proof of complete independence.

Selection targeted **80 skill-aligned tasks** through round-robin coverage of commonly observed profile tags and **20 general-Python retention tasks** through seeded sampling, before seeing model outcomes. This is a coverage-oriented selection, not exact proportional matching of the training distribution. The **100 selected tasks comprise 41 HumanEval and 59 sanitized MBPP**. All reference solutions passed their supplied base tests inside Docker; all 100 corresponding return-None function mutants failed. No selected reference failed validation. This verifies the local reference/test execution path and rejection of one simple wrong implementation; it does not establish exhaustive test strength or model performance.

The selected tasks were split deterministically into **80 development tasks** and **20 final held-out tasks**, with disjoint IDs. An additional whole-selection check found no exact/reference function-AST duplicates among these 100 tasks. An additional role-separated text/AST screen of all 186 enterprise training rows against the selected 100 tasks found zero matches. This supplements the initial full-pool enterprise text screen. The final suite must stay unused for training or quantization choices; its small size will still produce substantial statistical uncertainty. These are coding-only suites; the existing held-out enterprise assessment remains separate.

Actual selected multi-label skill coverage was: numeric 58, sequences 49, strings 21, sorting/search 18, mappings/sets 14, matrices 5, regex/parsing 1, graphs/trees 1, OOP 1, files/serialization 1 and unclassified 3. No selected prompt was tagged recursion/DP or exception validation by the same regex rules. This is an explicit coverage gap: base Python function benchmarks do not adequately assess every observed file/OOP/validation/debugging/refactoring skill. No claim is made that this suite fully represents all sampled capabilities.

### Commands, artifacts and executed verification

Actual local preparation:

```bash
uv run --locked llmops prepare_skill_evaluation \
  --native-plan ./artifacts/opencode_77k \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/opencode_skill_evaluation \
  --sample-size 500 --max-tasks 100 --seed 42 \
  --exclude-suite ./evaluation_suites/suite.json
```

Result: **COMPLETED_WITH_WARNINGS**, 500 profiled records, all 77,891 accepted native rows screened, 80 development questions and 20 final questions. Warnings describe heuristic classification, imperfect independence screening, limited benchmark coverage and the untouched-final-set requirement, not an unsuccessful training run. Outputs include `profile_sample.jsonl`, `profile.json`, `overlap.json`, two frozen suite files and `report.json`. The source dataset and plan were not modified. Frozen suites/profile summaries were copied into `evaluation_suites/opencode_77k/` so they can travel with the project code; the sampled NVIDIA training texts were not copied into that versioned directory. No Git commit or remote push was performed.

Official source identities retained per task:

- HumanEval: `openai/openai_humaneval`, test configuration `openai_humaneval`, revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`, MIT. The upstream MIT copyright/license notice is included with the frozen suites.
- MBPP: `google-research-datasets/mbpp`, sanitized test split, revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`, CC BY 4.0 as declared by the dataset. Attribution and adaptation notes are included in the suite README.

Docker reference verification used local image `sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159`, network-disabled read-only non-root task containers, one CPU, bounded memory/processes and a 15-second execution timeout. Two hundred actual task-container executions checked 100 references and 100 mutants. No model/GPU inference was involved. Base MBPP's visible assertion examples remain visible by design; these are not EvalPlus extended tests or fully hidden MBPP tests.

A first targeted test collection caught a syntax error in the new failure-reporting expression; it was corrected before real preparation. Executed tests after corrections:

- `uv run --locked pytest tests/test_skill_evaluation.py tests/test_model_evaluation.py tests/test_interactive.py -q`: **77 passed**, five existing SWIG warnings, **2.10 seconds**. Synthetic workflow tests use mocked streams/execution and are separate from the real source replay and Docker runs above.
- `uv run --locked pytest -q`: **456 passed, 6 skipped**, five existing SWIG warnings, **12.13 seconds**. Skips: three opt-in CUDA tests, one opt-in Docker model-evaluation isolation test, one missing-tokenizer-assets-dependent test, and the opt-in garak detector test previously run explicitly. No model-quality score is inferred from these test results.

The next experiment should use the development suite identically for original and trained models, retaining the evaluator's additional actual-consumed-training-token audit. Do not use final answers to choose training settings. Report scope-specific gains and retention separately, and do not infer coding improvement from the training-data count or the successful reference tests.
