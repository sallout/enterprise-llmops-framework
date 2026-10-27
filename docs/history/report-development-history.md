# Dissertation Project Report

## 1. Project overview

This repository will support the final dissertation research deliverable through modular Python pipelines. Development began from a clean scaffold on 25 October 2026. No previous source code, experimental scripts, or legacy architecture was reused. The existing `.env` and ten synthetic policy documents under `dumy_policies/` are preserved as local configuration and future Pipeline 1 inputs.

## 2. Initial architecture and decisions

- **Python 3.12+ with uv:** a pinned development interpreter series, isolated environment, reproducible dependency lock, and a standard package build workflow. Verification used CPython 3.12.14 and uv 0.12.5.
- **src package layout:** `policy_dissertation` is installed before use, reducing accidental imports from an uninstalled repository root.
- **Hatchling:** a minimal standards-based build backend for source and wheel distributions.
- **Pydantic Settings:** typed configuration with environment precedence over a local `.env`, nested `MODEL__*` keys matching the preserved configuration, and `SecretStr` for the API credential. Credentials are excluded from model representations and masked in JSON serialization. Settings are loaded at command execution, not import or help time.
- **Typer:** discoverable CLI commands and automatic help; one status command establishes a working entry point without implementing research functionality.
- **Standard Python logging:** an application-owned stderr logger, configurable severity, and repeatable initialization without duplicate handlers.
- **Error boundary:** safe expected errors and generic unexpected failures return code 1; invalid configuration returns code 2 without printing validation values. Normal help and status return 0.
- **pytest:** small offline checks for scaffold behavior, isolated synthetic configuration, and secret masking.
- **Pipeline isolation:** separate empty namespaces reserve policy transformation, training, quantization, security auditing, benchmarking, and deployment. Only the first is named Pipeline 1; later numbering and interfaces remain undecided.

No LLM client dependency is installed. Endpoint configuration is retained for future use without making network calls.

## 3. Files and directories created

| Path | Role |
| --- | --- |
| `pyproject.toml` | Package metadata, CLI entry point, dependency groups, build backend, pytest configuration |
| `uv.lock` | Resolved development and runtime dependency versions |
| `.python-version` | Python 3.12 development selection |
| `.gitignore` | Secrets, environments, caches, generated datasets, models, artifacts, editor state |
| `README.md` | Project scope, setup, configuration, commands, architecture, limitations |
| `report.md` | Architecture and research record |
| `src/policy_dissertation/__init__.py` | Package identity |
| `src/policy_dissertation/__main__.py` | Python module entry point |
| `src/policy_dissertation/cli.py` | Typer application, status command, error boundary |
| `src/policy_dissertation/config.py` | Validated settings and nested model configuration |
| `src/policy_dissertation/logging.py` | Stderr logging setup |
| `src/policy_dissertation/errors.py` | Expected application error type |
| `src/policy_dissertation/pipelines/__init__.py` | Pipeline namespace |
| `src/policy_dissertation/pipelines/policy_transformation/` | Pipeline 1 placeholder |
| `src/policy_dissertation/pipelines/training/` | Training placeholder |
| `src/policy_dissertation/pipelines/quantization/` | Quantization placeholder |
| `src/policy_dissertation/pipelines/security_auditing/` | Security auditing placeholder |
| `src/policy_dissertation/pipelines/benchmarking/` | Benchmarking placeholder |
| `src/policy_dissertation/pipelines/deployment/` | Deployment placeholder |
| `tests/test_scaffold.py` | Six scaffold tests |

Each pipeline directory contains only a documented `__init__.py`. uv initialized Git metadata because the directory had no Git repository. Local generated `.venv/` and `dist/` are ignored. Existing resources were neither rewritten nor relocated.

## 4. Implementation progress and verification

Initialization is complete. Runtime dependencies installed: `pydantic-settings` 2.16.1 and `typer` 0.27.3; development dependency: `pytest` 9.1.1. Pydantic 2.14.1, python-dotenv, and other transitive dependencies are recorded in `uv.lock`.

Commands actually executed:

| Command | Observed result |
| --- | --- |
| `uv init --package --name policy-dissertation --python 3.12 --build-backend hatch` | Project initialized successfully |
| `uv add pydantic-settings typer` | Local editable package and runtime dependencies installed |
| `uv add --dev pytest` | Test dependencies installed |
| `uv sync --locked` | Lock resolved and installed environment checked successfully |
| `uv run policy-dissertation --help` | Exit 0; help lists status and standard options |
| `uv run policy-dissertation status` | Exit 0; reports initialized state and unimplemented pipelines; existing local settings load successfully |
| `uv run python -m policy_dissertation --help` | Exit 0; module entry point works |
| `uv run pytest` | **6 passed in 0.19s**, no failed or skipped tests |
| `uv build` | Source distribution and wheel successfully built |

Test coverage includes help independent of invalid settings, status behavior, nested dotenv loading and environment precedence, masked secrets, invalid configuration without input disclosure, expected/unexpected error exit codes, and nonduplicated stderr logging. Tests do not access real endpoints or transform documents.

Additional checks executed with a local Python verification script confirmed SHA-256 equality for `.env` and all ten policy files against a pre-initialization baseline. Archive inspection confirmed `.env` was absent from both source and wheel distributions. `git check-ignore .env .venv/example data/example models/example .pytest_cache/example` confirmed those paths are ignored. `git status --short` confirmed the synthetic policy inputs remain eligible for version control while secrets and generated outputs are excluded.

## 5. Current limitations

The scaffold provides structure, not research results. Pipeline 1 and all other pipelines are unimplemented. No document ingestion, transformation schemas, LLM requests, training, quantization, security auditing, benchmarking, or deployment exists. Endpoint availability, authentication, model compatibility, policy parsing, and transformation quality remain unverified. Paths are relative to the current working directory unless explicitly overridden. Reproducibility was checked on this macOS Python environment; other operating systems have not been tested. Dependency installation emitted a nonfatal warning about an upstream version specifier; resolution, installation, tests, and builds succeeded.

## 6. Research documentation structure

Future work will extend this report with:

1. Research questions, objectives, scope, and success criteria.
2. Input provenance, synthetic-policy characteristics, data contracts, and transformation methodology.
3. Per-pipeline implementation decisions and reproducibility instructions.
4. Experiment protocols, configuration records, baselines, and evaluation measures.
5. Results with evidence, error analysis, and threats to validity.
6. Discussion, limitations, conclusions, and references.

These are planned sections, not completed studies. Do not report placeholder modules as implemented research functionality.

## 7. Next planned task

**Pipeline 1 — Corporate Policy Transformation.** The next implementation instruction should define transformation outputs and acceptance criteria, using the existing `dumy_policies/` documents and securely loaded endpoint settings. No Pipeline 1 functionality was implemented during initialization.

## 8. Pipeline 1 — Task 1: Document Ingestion & Parsing

### 8.1 Objective, scope, and dissertation contribution

Implemented on 25 October 2026. This milestone converts the existing synthetic corporate policy sources into structured, traceable document artifacts for the Corporate Data Transformation phase. It establishes the source references, byte-level provenance, page-level evidence, and failure reporting needed before downstream processing can be studied. It does not establish transformation quality or any training outcome.

Scope is limited to discovery, document parsing, schemas, artifact storage, CLI integration, tests, and documentation. No semantic chunking, training-example generation or validation, fine-tuning, other pipelines, LLM requests, LangGraph execution, dataset downloads, or standalone model downloads were implemented or performed. Sections 1–7 above preserve the historical initialization record; this section describes the current ingestion capability.

### 8.2 Components and files

Created within `src/policy_dissertation/pipelines/policy_transformation/`:

- `schemas.py`: Pydantic document, extraction metadata, page, artifact-reference, and batch-summary models.
- `parsers.py`: strict UTF-8 text/Markdown parsing and per-page PDF Markdown/raw-text extraction.
- `ingestion.py`: recursive sorted discovery, source-derived identifiers, SHA-256 hashes, failure isolation, atomic JSON writes, and reference-only summary.

Also created `tests/test_ingestion.py` with 22 test cases (including parametrized cases). Modified the policy transformation package description, `cli.py`, `config.py`, `pyproject.toml`, `uv.lock`, the existing status test, `README.md`, and this report. The `llmops` executable is an alias for the existing CLI. `PARSED_OUTPUT_DIR` is a new optional setting. Existing `.gitignore` rules already exclude `artifacts/`, so no change was needed there. The existing `.env` and all ten source policy files remain unchanged, verified against SHA-256 baselines captured before implementation.

### 8.3 Parsing strategy and dependency choices

Final locked extraction dependencies are **PyMuPDF 1.26.7** and **PyMuPDF4LLM 0.0.27**, constrained to `>=1.26,<1.27` and `>=0.0.27,<0.1` respectively. PyMuPDF provides local PDF access, physical page boundaries, raw text, and test-fixture creation. PyMuPDF4LLM produces Markdown with inferred headings, list formatting, and tables. The lightweight pre-layout release was selected to avoid model-based layout inference and its additional dependencies. The initial unconstrained resolution selected 2.0 and layout dependencies; these were replaced and removed from the active environment with `uv sync --locked`. No layout inference was run.

Official API reference consulted: [PyMuPDF4LLM API](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html). Because that reference tracks newer versions, the installed 0.0.27 API and actual extraction behavior were also inspected and tested.

TXT (`.txt`) and Markdown (`.md`, `.markdown`) suffixes are case insensitive. UTF-8 and UTF-8 BOM are accepted; source text and line endings are retained. No headings, tables, or page numbers are invented for these formats. Invalid UTF-8 and NUL-containing inputs fail explicitly. PDF processing reads the same bytes used for hashing, identifies headings across the document, and extracts one physical page at a time. This page separation is provenance capture, not semantic chunking. OCR, image export, and embedded-image extraction are absent. Raw text is additionally retained on each PDF page for comparison with Markdown.

### 8.4 Document schema and artifact contract

`DocumentArtifact` contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Currently `1.0` |
| `document_id` | SHA-256 of normalized absolute source file URI |
| `source_path` | Absolute source location |
| `source_type` | `pdf`, `markdown`, `txt`, or `unsupported` |
| `content_hash` | SHA-256 of original bytes; null when the source was not read |
| `extracted_content` | Combined PDF Markdown or unchanged decoded text |
| `extraction_metadata` | Parser, versions, format, encoding, page count, page records, warnings, error |
| `processing_status` | `parsed`, `failed`, or `unsupported` |

Each PDF page record contains `page_number` (1-based physical index), `extracted_content`, `raw_text`, and `warnings`. Printed page labels are not treated as physical page indexes. TXT and Markdown use an empty page list and null page count. Unsupported files are reported without reading their content, so their hashes remain null; unreadable files also have null hashes. Readable files that fail parsing retain their source-byte hash.

Identifiers remain stable for edits at the same normalized absolute location; content hashes change with source bytes. Identical bytes at different locations have distinct identifiers and identical hashes. Moving the repository changes document identifiers; they are location identities, not portable content identities.

Outputs are UTF-8 JSON:

```text
artifacts/pipeline1/parsed/
├── documents/<document_id>.json
└── summary.json
```

The summary stores counters, discovery warnings, and source/artifact references only. The ingestion API returns this lightweight summary, avoiding large document content in any future orchestration state. Each document is processed and written before the next. Reruns atomically replace artifacts for the same identities and replace the summary; orphaned artifacts from removed sources are not deleted. Consumers must use the current summary as the authoritative manifest.

Input and output trees may not overlap. Symbolic links are not followed. Unreadable directories produce discovery warnings; inaccessible contents cannot be counted as discovered documents. Per-document failures do not stop other files. Output-write failures stop the batch with an explicit application error, because an artifact cannot be reliably persisted. A run with failed documents or discovery errors returns exit 1 after writing the available summary; unsupported files alone do not change exit 0. Invalid configuration returns 2.

### 8.5 Executed commands and actual results

| Executed command/check | Observed result |
| --- | --- |
| `uv add pymupdf pymupdf4llm` | Initial latest-version installation succeeded; subsequently replaced with lightweight versions |
| `uv add 'pymupdf>=1.26,<1.27' 'pymupdf4llm>=0.0.27,<0.1'` | Required parser versions installed and lock updated |
| `uv sync` | CLI alias installed successfully |
| `uv sync --locked` | Final environment and lock verified; no unresolved installation errors |
| `uv run pytest` (first implementation run) | 25 passed in 2.64s |
| `uv run pytest` (after raw-text and additional failure/table tests) | 28 passed in 0.23s |
| `uv run pytest` (final verification) | **28 passed in 0.27s**, 5 upstream SWIG deprecation warnings; no failures or skipped tests |
| `uv run llmops policies parse --input dumy_policies/` | Exit 0; 10 discovered, 10 parsed, 0 failed, 0 unsupported |
| `uv run llmops policies parse --input dumy_policies/ --output artifacts/pipeline1/parsed` | Final run exit 0; same counts, 40 PDF pages |
| `uv run llmops policies parse --help` | Exit 0; input/output options displayed |
| `uv run policy-dissertation policies parse --help` | Exit 0; original executable exposes the same command |
| Local artifact verification script | All 10 JSON documents and summary validated with Pydantic; all content hashes matched sources; all 40 raw page records matched direct source extraction |
| Preserved-resource hash verification | `.env` and all ten policy PDFs unchanged |

The final 28-test suite comprises the original six scaffold tests and 22 ingestion cases. It verifies a generated two-page PDF fixture, a table PDF, source/page references, UTF-8 Markdown/TXT with preserved line endings, empty and whitespace files, corrupt PDFs, invalid/binary text, unsupported files, simulated read denial, encrypted PDFs, deterministic IDs and hashes, recursive statistics, JSON round trips, reference-only summaries, successful CLI parsing, partial-failure CLI behavior, overlapping/missing directory guards, blank PDFs, symlink skipping, output-write failures, and retained content after a simulated page-extraction failure. Fixtures are small, generated locally under pytest temporary directories, and require no corpus downloads or network access.

Actual final processing statistics:

| Metric | Result |
| --- | ---: |
| Discovered files | 10 |
| Parsed documents | 10 |
| Failed documents | 0 |
| Unsupported documents | 0 |
| Physical PDF pages | 40 |
| Document JSON artifacts | 10 |
| Summary JSON artifacts | 1 |

These are execution counts, not an accuracy evaluation. A `parsed` status indicates nonempty extraction without page failures, not verified semantic or visual fidelity. The parser prints an upstream suggestion about the optional layout package; this is informational and did not prevent execution.

### 8.6 Artifact inspection and known limitations

Inspected artifact:

`artifacts/pipeline1/parsed/documents/9daa94ffcde30b00c64b5f98fb05ad3b5cd31084a98869e7348b1ac6b17f718d.json`

This references the four-page `POL-AI-005` source policy. The first original page was rendered to a temporary PNG and visually inspected. Its title, numbered Objective/Scope/Governance sections, metadata table, and mandatory policy language appear in the Markdown. Programmatic checks verified the heading, Objective section, table syntax, and a policy rule; raw text retained table labels such as `Policy Owner` and `Chief AI Officer / CISO / Data Protection Officer`. Page records are physical pages 1–4. A wider check confirmed direct raw-text equality on every page of all ten sources.

Observed inaccuracies: first-row table text loses spaces (for example, `PolicyOwner`), and visible bullet symbols decode as `G` in both Markdown and direct raw text. Paragraph spacing is uneven, and heading levels are inferred from typography. These issues are not automatically repaired because that would risk fabricating structure or content. Metadata explicitly warns about layout, glyph, and table-whitespace uncertainty. Raw text offers an additional reference but shares font-encoding limitations and cannot replace the original PDF.

Other limitations: no OCR for image-only scans; no password handling; no non-UTF-8 fallback; no DOCX support; no guarantee for complex or merged tables, reading order, columns, embedded images, or malformed fonts. A blank page or extraction error conservatively marks the document failed while retaining other pages, so intentional blank pages may require review. Files are read into memory one at a time without an enforced size limit. Native parser hangs/resource exhaustion are not isolated in subprocesses. Absolute source paths are stored for traceability and may need redaction before sharing. Cross-platform behavior has not been tested. The five upstream SWIG deprecation warnings remain nonblocking.

### 8.7 Next step

Document ingestion and parsing is complete for this task. Await the next implementation instruction before adding semantic chunking, LLM transformation, training-example schemas, validation, or any other pipeline capability.

## 9. Pipeline 1 — Task 2: Parsing Quality and Structure-Aware Semantic Chunking

### 9.1 Objective and scope

Implemented on 25 October 2026. This stage consumes the existing ingestion manifest and parsed document JSON artifacts, assesses extraction quality, creates a separate deterministic normalized representation, and groups content into section-aware policy chunks. The original PDFs, `.env`, and parsed artifacts are unchanged. No parsing component was reimplemented or source artifact overwritten.

This is a structural approximation of semantic policy units. No embedding model, generative model, training-example generation, training-example validation, fine-tuning, or other pipeline functionality was added. Earlier report sections remain a historical record; statements that chunking was unimplemented describe those earlier milestones.

### 9.2 Identified extraction limitations and diagnostics

The earlier PDF inspection established that visible bullet glyphs can decode as `G`, table words can lose spaces, and Markdown paragraph spacing can be irregular. Current privacy-policy inspection additionally identified a breach-notification clause spanning physical pages 2–3, interrupted by the repeated controlled-document header and synthetic-template footer. Naive fixed-size or page-based splits could detach its 72-hour condition and risk-based exception.

Structured quality findings identify:

- Leading `G` as an ambiguous possible bullet; actual words are retained.
- Unambiguous Unicode bullet markers eligible for safe Markdown-marker normalization.
- Markdown tables without valid separators or with inconsistent column counts.
- Potential merged table words using lowercase-to-uppercase transitions and, where available, a comparison to raw page text. These are suspicions, not definitive spelling corrections.
- Repeated lines at page edges, distinguishing allowlisted boilerplate from uncertain or normative repeated content.
- Excess blank lines, excessive internal spaces, and replacement glyphs.
- Unfinished prose at page boundaries, distinguishing an eligible lowercase continuation from uncertain boundaries.
- Gaps, duplicates, orphaned numbers, missing numbers on peer headings, absent numbering, and inconsistent numeric/Markdown hierarchy.
- Uncertain clause boundaries, marked conditional continuations, and indivisible oversized units.
- Original parser warnings, retained as unresolved document-wide context rather than discarded.

Every finding includes a deterministic local ID, code, message, resolution, and original text spans when locatable. Local warning IDs are scoped to the document/report; they are not global identifiers. The checks are heuristic and may generate false positives. They are not legal, semantic, or training-example validation.

### 9.3 Safe normalization and provenance

Original and normalized representations are stored separately. Each normalized line retains its exact original text, normalized text, original page and half-open character offsets, join separator, exclusion flag, and action list. Offsets reference a physical page's original extracted Markdown; without physical pages they reference `document.extracted_content`. Raw PDF page text remains in the original parsed artifact for comparison. No page numbers are invented for TXT or Markdown.

Permitted automatic changes:

1. Normalize CRLF line endings, blank-line runs, and single trailing spaces. Markdown hard breaks, table spacing, code fences, and indented code receive conservative handling; arbitrary internal word spacing is not rewritten.
2. Convert explicit line-leading Unicode bullet characters (`•`, `●`, `▪`) followed by whitespace into a Markdown `-` marker. The real corpus's ambiguous `G` markers do not satisfy this evidence standard and are never automatically replaced.
3. Exclude repeated edge boilerplate only after it appears on at least three pages and at least 75% of document pages. Eligible text is restricted to the known controlled-document header pattern, plain page labels, and the exact synthetic-template footer pattern, allowing its page number to vary. Unknown or normative repeated text is retained and flagged. Excluded text remains fully recoverable in normalization reports and original artifacts.
4. Join unfinished noncode/nontable prose to a lowercase continuation on the next physical page when no heading intervenes. Hyphenated endings and other uncertain cases remain unjoined and are flagged. No missing words, punctuation, or numbers are invented.

The runtime invariant compares each retained source line's nonwhitespace lexical content with its normalized version, allowing only the recorded explicit bullet substitution. Exclusions must satisfy the narrow boilerplate allowlist. The original Pydantic artifact must remain unchanged. Chunk coverage separately verifies that every normalized nonheading body line occurs exactly once and in order. These checks preserve language mechanically; they do not prove that source extraction or clause interpretation is semantically correct.

`MUST`, `MUST NOT`, `SHOULD`, `MAY`, and `Strictly Prohibited` retain their case and words. Conditions, exception language, thresholds, approval requirements, and other lexical content are not rewritten. Phrase whitespace may differ where harmless whitespace or a recorded page continuation is normalized. Some PDF line/paragraph artifacts intentionally remain unresolved.

### 9.4 Selected chunking method and token budget

The chunker follows Markdown headings and conservatively recognized plain numbered policy headings. It stores the current section and parent contexts; explicit numeric ancestry takes precedence over inconsistent inferred Markdown depth, with a warning. It does not invent section numbers or missing parents.

Preferred units are complete list items, including all subsequent paragraphs until a new clause or heading. A marked conditional item beginning with wording such as `Unless` or `Provided` stays with the preceding requirement and is flagged for review. Uninterrupted prose runs remain together conservatively rather than being split at every extracted blank line. This protects many locally attached exceptions but does not resolve all dependencies across separate clauses or sections.

Adjacent units pack into chunks within the same section only. There is no overlap, truncation, or duplicated body text. Section/parent context is metadata rather than repeated body content. Reliable page continuations stay in the same unit and carry both page references. A unit that cannot fit safely is retained in full with `oversized=true` and an `oversized_clause` finding.

Section-aware grouping was selected because page and fixed-size boundaries can split mandatory clauses from conditions or exceptions, while the observed artifacts contain recognizable policy sections and list starts. This is an architectural justification, not an experimentally demonstrated superiority claim. No comparative downstream model evaluation has been conducted.

The selected documented estimation method is `utf8_bytes_upper_bound_v1`: `token_count = len(content.encode('utf-8'))`. Default `--max-tokens` is 2048 byte units. This is a conservative content-token upper bound for ordinary byte-BPE encodings, not an actual count for a selected LLM tokenizer. Metadata, section context serialization, prompt/chat wrappers, and future training formatting are outside this budget. No tokenizer vocabulary or model was downloaded. Oversized behavior is covered by tests, though no corpus unit exceeded the default budget.

Chunk IDs are SHA-256 over the algorithm version (`structure-v1`), document identity, source-byte content hash, exact chunk content, ordered source-line IDs, and section context. The same source version/configuration produces the same IDs and output bytes. A changed limit keeps IDs when grouping/content/context do not change. A changed source-byte hash changes IDs even if some local clause wording stays the same.

### 9.5 Implemented components and files

New files under `src/policy_dissertation/pipelines/policy_transformation/`:

| File | Responsibility |
| --- | --- |
| `chunk_schemas.py` | Canonical typed chunks, source spans, normalized lines/pages/reports, quality findings, batch references/statistics |
| `structure.py` | Shared heading, section-number, clause-marker, and Markdown-table recognition |
| `normalization.py` | Quality assessment, conservative normalization, original/normalized mappings, lexical invariants |
| `chunking.py` | Hierarchical section context, clause grouping, token estimate, stable identifiers, complete nonoverlapping coverage |
| `chunk_batch.py` | Existing ingestion-manifest consumption, JSONL chunks, separate audit reports, document failure isolation, statistics |

Created `tests/test_chunking.py` (30 test cases, including parametrized cases). Modified the existing `cli.py`, `config.py`, policy-transformation `__init__.py`, `README.md`, and this report. `CHUNK_OUTPUT_DIR` defaults to `artifacts/pipeline1/chunks`; the existing `PARSED_OUTPUT_DIR` remains the default input. No new runtime or development dependencies were required, and `uv sync --locked` verified the existing environment successfully.

Output structure:

```text
artifacts/pipeline1/chunks/
├── documents/<hash-of-document-id>.jsonl
├── normalization/<hash-of-document-id>.json
├── quality/<hash-of-document-id>.json
└── summary.json
```

Each document identity is hashed again for a safe artifact filename because incoming JSON identifiers are untrusted path material. `summary.json` maps original document IDs to these files, and retains counts/references rather than document text. All outputs are UTF-8. JSON/JSONL files are replaced atomically per file. Outputs are not transactional across the whole batch; a write failure can leave a partially refreshed output directory and a previous summary. Consumers should accept only a successful completed run and its manifest. Old unreferenced files are retained rather than deleted.

### 9.6 Canonical policy chunk schema

| Field | Type / meaning |
| --- | --- |
| `schema_version`, `algorithm_version` | Artifact contract and grouping version |
| `chunk_id` | Deterministic SHA-256 identifier |
| `document_id`, `source_content_hash`, `source_path` | Source identity, source-byte hash, absolute source reference |
| `section_id`, `section_title`, `section_heading` | Optional source section number, title, exact normalized heading |
| `parent_sections` | Typed parent contexts containing ID, title, original heading depth/text, and source span |
| `section_source_spans` | Original spans for the current heading and its ancestors |
| `page_numbers` | Sorted physical pages represented in body content; empty for unpaged sources |
| `content` | Complete normalized clause/prose content |
| `token_count`, `token_method`, `max_tokens` | UTF-8 byte upper-bound count, documented method, configured limit |
| `oversized` | Whether an indivisible unit exceeds the limit |
| `line_ids`, `source_spans` | Ordered line references and original page-character offsets for content reconstruction |
| `normalization_status` | `unchanged`, `normalized`, or `review_required` |
| `quality_warnings` | Relevant typed unresolved/corrected findings plus document-wide parser warnings |

Heading-context spans can reference earlier pages than the chunk body; `page_numbers` lists body-content pages. Chunk content can be reconstructed exactly from normalization-report lines and their recorded join separators. Original source lines can then be recovered from the original spans, including excluded boilerplate in the report. This is traceability to the extracted document version, not a guarantee of correspondence to every visible PDF glyph.

### 9.7 Commands executed and actual test results

Implemented CLI:

```sh
uv run llmops policies chunk --input artifacts/pipeline1/parsed --output artifacts/pipeline1/chunks --max-tokens 2048
```

The original executable also exposes `policy-dissertation policies chunk`. The batch consumes ingestion `summary.json`, skips entries that were not parsed successfully, and reports invalid artifacts without stopping valid documents. Failed documents produce exit 1 after reporting; quality warnings and oversized chunks alone remain reviewable successful output. Output-write failures stop with an explicit application error. Directory trees may not overlap, manifest references must remain inside the input directory, and manifest/source identities must agree.

Verification actually executed:

| Command/check | Actual outcome |
| --- | --- |
| `uv run pytest` during initial implementation | 51 passed in 0.30s |
| `uv run pytest` after additional regression checks | 56 passed in 0.33s |
| `uv run pytest` after section-hierarchy checks | 58 passed in 0.31s |
| `uv run pytest` final verification | **58 passed in 0.35s**, 5 upstream SWIG deprecation warnings; no failures/skips |
| `uv sync --locked` | Existing locked environment verified, no missing dependencies |
| `uv run llmops policies chunk --input artifacts/pipeline1/parsed --output artifacts/pipeline1/chunks --max-tokens 2048` | Exit 0; all 10 documents processed; 134 chunks |
| `uv run llmops policies chunk --help` | Exit 0; input/output/budget options displayed |
| Repeated corpus command with output SHA-256 comparison | All 31 JSON/JSONL files byte-identical across reruns |
| Artifact inspection/verification script | All 134 chunks reconstructed exactly from normalized line IDs; source spans valid; body lines complete/in order/nonduplicated; token accounting and limits verified |
| Protected-language occurrence comparison | Modal phrase occurrence counts, including deduplicated heading context, matched original extracted content for every document |
| Preserved-resource SHA-256 check | All 22 protected files unchanged: `.env`, 10 policy PDFs, 10 parsed document artifacts, and ingestion summary |

The full suite comprises 30 new chunking/quality cases, 22 ingestion cases, and six scaffold cases. New tests cover section detection, parent ancestry, numbering gaps/duplicates/orphans/unnumbered peers, inconsistent heading depth, modal/condition/threshold preservation, the cross-page privacy breach pattern, ambiguous `G` and explicit Unicode bullets, malformed tables and merged-word regressions, conservative repeated/unique footer handling, uncertain headers, stable/version-sensitive IDs, token limits and oversized units, line-span reconstruction, unpaged text, code and Markdown hard breaks, uncertain hyphenated continuations, conditional list items, abnormal formatting, no overlap, manifest errors, unchanged inputs, and end-to-end PDF/Markdown parsing-to-chunking CLI compatibility.

### 9.8 Actual corpus statistics and correction accounting

| Metric | Actual result |
| --- | ---: |
| Parsed documents referenced / processed | 10 / 10 |
| Skipped / failed documents | 0 / 0 |
| Generated chunks | 134 |
| Oversized chunks at 2048 byte units | 0 |
| Total quality findings | 517 |
| Findings marked corrected | 64 |
| Findings remaining unresolved | 453 |
| Chunks marked review required | 134 |
| Output files | 31: 10 JSONL, 10 normalization JSON, 10 quality JSON, 1 summary JSON |

Finding breakdown:

| Code | Count | Resolution / interpretation |
| --- | ---: | --- |
| `suspicious_bullet` | 340 | Unresolved; original `G` retained |
| `table_spacing_suspect` | 21 | Unresolved; merged-word suspicion, no inferred spacing inserted |
| `clause_boundary_uncertain` | 76 | Unresolved; includes short list items without punctuation, possible false positives |
| `conditional_list_continuation` | 1 | Unresolved review finding; conservatively kept with preceding clause |
| `cross_page_uncertain` | 5 | Unresolved; no inferred reconstruction |
| `parser_warning` | 10 | Unresolved original extraction caveats, propagated to every relevant document's chunks |
| `cross_page_continuation` | 4 | Corrected join operations, words unchanged |
| `excessive_whitespace` | 40 | Corrected page-level blank-line findings |
| `repeated_header_footer` | 20 | Corrected document/pattern groups, covering 80 excluded line instances |

The action audit records 731 extra blank-line collapses, 80 repeated-boilerplate line exclusions, and four cross-page joins. Findings and actions have different units: one grouped repetition finding covers several source lines; one whitespace finding can cover many blank lines. No real-corpus `G` markers or table words were automatically repaired. No malformed table syntax or numbering faults were detected in this run; tests verify these detectors on synthetic cases. Absence of a finding is not proof that no defect exists.

All 134 chunks remain `review_required`, including mechanically clean chunks, because original parser caveats are retained as unresolved document-wide context. This is intentional and does not classify the chunks as approved training examples.

### 9.9 Privacy-policy inspection and actual example

Inspected the privacy policy's generated chunks and matching normalization/original artifacts, focusing on:

- Section 6: the `MUST` breach-reporting and assessment requirements, continuation across physical pages 2–3, `72 hours after becoming aware`, and `unless the breach is unlikely to result in risk` are in the same chunk, with both page references. The normalized join reads `containment, and regulatory/individual notification obligations`; no words were supplied.
- Section 4: the prohibition on production personal data in development/training remains in the same chunk as its anonymization, pseudonymization, and documented-approval exceptions.
- Section 5: the TLS 1.2 `MAY` clause remains with `only when required for interoperability` and approved-cipher conditions.
- Section 9: lawful/necessary exception conditions, DPO/Privacy approval, Security approval where controls are reduced, fixed expiry, the prohibition on waiving statutory obligations, and `MUST NOT` permanent-policy-change language remain together.

Actual section 9 chunk:

- JSONL file: `artifacts/pipeline1/chunks/documents/7ba3c11be82e580434f08114dfcc31645a8d019b0afa86711d9e1a1daae79628.jsonl`, line 11.
- Chunk ID: `60e4d08a066f1790705748fab9b86d878074db5ff439ed0dd5c034bd064ac610`.
- Section: `9`, `Exceptions and Risk Acceptance`.
- Parent context: `Data Privacy and GDPR Compliance Policy`.
- Body pages: `[3]`; byte-token count: `637`; normalization status: `review_required`.

Its exact generated content is:

```text
Privacy exceptions are permitted only where lawful and necessary. An exception MUST identify the specific

requirement, legal basis, processing purpose, affected data, risk assessment, safeguards, accountable owner,
DPO/Privacy approval, Security approval where technical controls are reduced, and a fixed expiration date. No

exception may waive statutory rights or regulatory obligations.

An exception MUST NOT be interpreted as permanent policy change. The control owner may require additional

monitoring, narrower scope, remediation milestones, or earlier review. Expired exceptions automatically cease to

authorize the deviation.
```

These targeted checks verify retention of selected source requirements and their local context. They do not certify legal validity, original PDF extraction fidelity, or semantic correctness of the entire corpus.

### 9.10 Remaining risks and dissertation relevance

Unresolved risks include ambiguous glyphs, false-positive table/formatting checks, missing table spacing, irregular within-clause paragraph breaks, heuristic heading/list interpretation, unknown boilerplate intentionally retained, uncertain continuation boundaries, and cross-section/global exception dependencies. Lowercase continuation is evidence for a join, not a semantic proof. Plain numbered-heading recognition is deliberately narrow. Long prose runs can exceed the budget rather than being aggressively split. Protected-language occurrence checks cannot prove that every condition is interpreted correctly. No semantic quality score, training readiness, or fine-tuning benefit has been established.

The token method is conservative and model independent; a selected model's exact tokenizer and final prompt budget will be needed later. Absolute source paths may need redaction before sharing. Character spans refer to extracted text, not PDF bounding boxes. IDs are version-sensitive and source-location-derived through the ingestion stage. Rerun outputs are deterministic but storage has no whole-batch transaction, stale-artifact cleanup, subprocess isolation, or resource limits. Verification was performed on the existing macOS/Python environment only. The five upstream SWIG warnings remain nonblocking.

This stage supports future fine-tuning dataset generation by making source clauses, section context, normalization actions, uncertainty, and provenance explicit and recoverable before examples are generated. It provides auditability rather than claiming dataset validity.

### 9.11 Recommended next task

Conduct a focused human parsing/chunk-quality review and define explicit acceptance criteria for unresolved bullets, table spacing, global exception context, and chunk eligibility. After that review, wait for explicit instruction before implementing LLM-based training-data generation. No generation work has begun.

<!-- pipeline1-run:15e9b6a9baf0e2484dd37ad70f0314d1e3c65027694cbf2b130d4906c9a7e4e4 -->
## Pipeline 1 automated execution — 2026-10-25T22:03:59.416078+04:00

- Run ID: `15e9b6a9baf0e2484dd37ad70f0314d1e3c65027694cbf2b130d4906c9a7e4e4`; status: **FAILED**.
- LangGraph stages: parse → normalize/chunk → generate → validate → export.
- Documents processed: 1; chunks: 13.
- Generated examples: 5; accepted: 0; rejected: 0.
- Failed chunks: 13; LLM calls this invocation: 6.
- Validation: Pydantic schema, source identity, complete extractive passage, model-assisted grounding/conditions/scope, global exact question/answer deduplication.
- Accepted means these automated gates passed; semantic/legal correctness and downstream fine-tuning performance remain unverified.
- Output report: `/Users/mas/Development/DissertationProject/artifacts/pipeline1_integration/report.json`.
- Errors: ["pipeline_stage_failed:KeyboardInterrupt", "mandatory_chunk_processing_incomplete", "no_accepted_training_examples"].

No Pipeline 2 or model fine-tuning was performed.
<!-- /pipeline1-run:15e9b6a9baf0e2484dd37ad70f0314d1e3c65027694cbf2b130d4906c9a7e4e4 -->

<!-- pipeline1-run:6d5dfa4e45a7acdc3c5cd4a685fb11eefbe97b6d3cbbec3625def4d9cfa08474 -->
## Pipeline 1 automated execution — 2026-10-25T22:09:54.516800+04:00

- Run ID: `6d5dfa4e45a7acdc3c5cd4a685fb11eefbe97b6d3cbbec3625def4d9cfa08474`; status: **COMPLETED_WITH_WARNINGS**.
- LangGraph stages: parse → normalize/chunk → generate → validate → export.
- Documents processed: 1; chunks: 13.
- Generated examples: 13; accepted: 12; rejected: 1.
- Failed chunks: 0; LLM calls this invocation: 0.
- Validation: Pydantic schema, source identity, complete extractive passage, model-assisted grounding/conditions/scope, global exact question/answer deduplication.
- Accepted means these automated gates passed; semantic/legal correctness and downstream fine-tuning performance remain unverified.
- Output report: `/Users/mas/Development/DissertationProject/artifacts/pipeline1_integration/report.json`.
- Errors: [].

No Pipeline 2 or model fine-tuning was performed.
<!-- /pipeline1-run:6d5dfa4e45a7acdc3c5cd4a685fb11eefbe97b6d3cbbec3625def4d9cfa08474 -->

## 10. Complete End-to-End Pipeline 1 — Corporate Policy Data Workflow

### 10.1 Objective and implemented scope

Integrated the existing parsing, quality assessment, normalization, and section-aware chunking components into one automated command. The workflow now generates source-grounded conversational records through the configured LLM, validates them, exports accepted and rejected data separately, and writes actual execution statistics and a research-report entry. The earlier instruction not to generate data was superseded by the explicit end-to-end implementation request. Historical sections and failed-attempt evidence are preserved.

```sh
uv run llmops pip1_data --input ./dumy_policies --output ./artifacts/pipeline1 --max-tokens 2048
```

There is no implicit document/chunk limit and no need to run intermediate commands. Input/output paths are honored. No Pipeline 2, model download, Qwen tokenizer conversion, fine-tuning, benchmarking, or deployment was implemented. The full ten-document corpus was not submitted to the live LLM during this task; real verification used the requested small subset.

### 10.2 Architecture and technology choices

A compiled **LangGraph StateGraph** orchestrates five stages:

1. `parse`: call existing recursive ingestion; preserve source documents and report unsupported/failed inputs.
2. `chunk`: call existing quality/normalization and structure-aware chunking; preserve original artifacts and warnings.
3. `generate`: call the configured endpoint through **LangChain ChatOpenAI**; ask for a specific question and a complete extractive source answer.
4. `validate`: validate candidate schema, identity, whole-passage support, relevance, conditions, and scope; retain per-chunk decisions.
5. `export`: perform global question/answer deduplication, publish accepted/rejected JSONL and report, and update this research report.

Graph state contains only stage names and artifact paths. Document content, source contexts, candidates, and checkpoints stay on disk or inside local node execution, not in persistent graph state. Durable checksummed per-chunk files provide recovery. LangGraph's SQLite checkpointer is not used; reruns reconstruct the graph and reuse completed generation/validation work. Parsing/chunking rerun deterministically.

Added runtime dependencies: `langchain-openai` **1.7.2** and `langgraph` **1.2.17**, with `langchain-core` **1.7.0** and other transitive dependencies locked in `uv.lock`. Existing Pydantic, Pydantic Settings, Typer, PyMuPDF, and PyMuPDF4LLM remain in use. LangChain provides the chat-completions adapter; LangGraph makes stages explicit and testable. Pydantic provides candidate, verdict, and conversational-record contracts.

References consulted: [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api), [LangChain ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai), and [Qwen3-14B's official model card](https://huggingface.co/Qwen/Qwen3-14B). Only the compatible chat-completions interface is assumed. Provider-native JSON-schema/tool support is not required; JSON-only prompting and Pydantic parsing work with the configured compatible endpoint.

### 10.3 Source files and outputs

New files in `src/policy_dissertation/pipelines/policy_transformation/`:

| File | Role |
| --- | --- |
| `data_schemas.py` | Candidate/envelope, strict grounding verdict, fixed three-message conversational record, validation outcome |
| `data_io.py` | Atomic JSON/JSONL writes, deterministic hashes, checksummed checkpoint loading/saving |
| `generation.py` | LangChain adapter, source passage choices, JSON prompting, bounded attempts, provider error classification and credential suppression |
| `training_validation.py` | Schema/reference/evidence gates, model-assisted grounding, record metadata, duplicate keys |
| `pipeline.py` | LangGraph orchestration, source/configuration fingerprints, resume, output locking, publication, execution reporting |

Created `tests/test_pipeline1.py`, `docs/pipeline1.md`, and `docs/schemas/pipeline1-training.schema.json`. Modified CLI, settings, package description, `pyproject.toml`, `uv.lock`, README, and this report. Existing parser and chunker algorithms were reused rather than rewritten.

The selected output directory receives `training.jsonl`, `rejected.jsonl`, and `report.json`. Intermediate artifacts and per-chunk checkpoints are under `_runs/<run_id>/parsed`, `chunks`, `generation`, and `validation`. Root JSONL exports are rebuilt from the current run's validated checkpoints rather than appended. Existing earlier parsed/chunked artifacts are not deleted.

### 10.4 Canonical training schema and Qwen compatibility

The canonical schema version is `pipeline1-extractive-v1`. Each accepted JSONL object contains:

- `example_id`: deterministic SHA-256 over source chunk/passage identity and conversation.
- `messages`: exactly three ordered role/content objects: system, user, assistant.
- `metadata`: application-built source/evidence references and validation evidence.

The system message contains the policy title plus recognized policy-wide Scope and Exceptions text. The user message is the generated instruction. The assistant answer must reproduce a complete selected source passage after whitespace normalization. Source passage selection retains clause continuation paragraphs and marked conditional continuations; arbitrary sentence truncation is not allowed.

Metadata includes document ID, source byte hash/path, chunk ID, section ID/title, parent sections, physical pages, source spans, passage ID, exact evidence quote and chunk-character offsets, policy-wide context references, original quality warnings, and the actual grounding verdict. The exported JSON Schema is in `docs/schemas/pipeline1-training.schema.json`; the detailed contract is in `docs/pipeline1.md`.

This role/content format can later be passed to Qwen3-14B's chat template after selecting the messages. Metadata must be retained separately or excluded from model loss. No template application, tokenization, thinking-mode decision, train/evaluation split, label masking, or Qwen fine-tuning has been performed. Format compatibility is not evidence of successful training or improved model behavior.

### 10.5 Validation and quality strategy

Required gates are:

1. Pydantic candidate validation with exact allowed fields, required references, and string length bounds. A malformed batch response receives bounded retries; invalid individual candidates are rejected without discarding valid siblings.
2. Exact document/chunk identity and passage-ID membership.
3. Equality of the answer to the entire selected source passage after whitespace normalization. This prevents unsupported answer wording, changed thresholds/negations, and sentence truncation inside that passage.
4. Credential-like output rejection/suppression.
5. A separate model-assisted review of the full chunk, proposed question/answer, and recognized policy-wide scope/exceptions. All four strict boolean fields must pass: grounding, answering the question, condition preservation, and scope correctness. A negative/uncertain verdict is rejected. Verification failure after retry exhaustion is a mandatory processing failure.
6. Global exact question/answer deduplication after whitespace normalization and case folding. Provenance differences do not permit duplicate committed pairs.

An exact quote alone does not prove that it answers the question completely. The live rejected example demonstrated this distinction: a supported quotation could omit other requirements relevant to the question. The review call is separate but uses the same configured model, so correlated errors remain possible. Accepted means automated gates passed; every record explicitly retains `automated_checks_passed_not_human_verified` metadata.

Formatting-only warnings are carried into metadata rather than becoming a blanket rejection. Ambiguous `G` markers and uncertain table words are not guessed away. Existing whitespace/boilerplate/continuation normalization remains active. No unsupported repair of policy wording was introduced. This permits valid records while preserving extraction uncertainty for later audit.

### 10.6 Configuration, status, recovery, and security

The existing `.env` is reused unchanged via `MODEL__BASE_URL`, `MODEL__API_KEY`, and `MODEL__MODEL_NAME`. New optional nested settings are:

| Setting | Default | Purpose |
| --- | ---: | --- |
| `PIPELINE1__RESPONSE_MAX_TOKENS` | 4096 | Provider output limit for generation/review |
| `PIPELINE1__MAX_ATTEMPTS` | 3 | Maximum attempts per model JSON call |
| `PIPELINE1__REQUEST_TIMEOUT` | 60 seconds | Per-request timeout |

The CLI's `--max-tokens` remains the existing source-chunk **UTF-8 byte-token upper bound**, default 2048; it is never substituted for a model response limit. An indivisible oversized chunk is retained by chunking but not sent for generation: it receives an operational failure and the run fails mandatory processing. No silent truncation occurs.

A deterministic run fingerprint includes input locations/hashes, chunk limit, endpoint fingerprint, model/configuration, prompt hash, schema version, and dependency versions. Credentials are excluded. Completed generation and terminal validation decisions are reused per chunk; corrupt/incomplete checkpoints are recomputed and failed calls can be retried. The version must be advanced for future validation/passage-selection semantic changes not represented by prompt/configuration identity.

An output-directory lock prevents concurrent writers to the same destination. Files are atomically replaced individually, and `report.json` is published last with hashes of both JSONL files. This is not a whole-directory transaction: consumers must check final status and hashes after an interruption. A provider call interrupted before its checkpoint commits may be repeated on resume; provider-level exactly-once billing is not claimed. Older fingerprint artifacts remain for audit. Report updates for concurrent runs in different output directories are not globally coordinated.

`COMPLETED` and `COMPLETED_WITH_WARNINGS` exit 0. Accepted data plus rejected candidates or retained source warnings uses the latter. Supported-document failures, failed mandatory chunk generation/verification, all-rejected output, or report-update failure produce `FAILED` and exit 1. Configuration errors exit 2. `rejected.jsonl` distinguishes actual candidate rejections (`kind=example`) from operational chunk failures (`kind=chunk_failure`); an API failure does not fabricate a generated example.

API credentials and raw provider exception bodies are never printed or serialized. HTTP status/timeout/connection categories are recorded safely. Optional model-call tracing is disabled. Actual source documents and `.env` were checked against pre-task SHA-256 baselines after integration and remain unchanged.

### 10.7 Automated tests actually executed

Executed:

```sh
uv add 'langchain-openai>=1,<2' 'langgraph>=1,<2'
uv run pytest tests/test_pipeline1.py
uv run pytest
uv sync --locked
uv run llmops pip1_data --help
```

Observed results:

- First end-to-end mocked suite: **18 passed in 4.92s**.
- Full suite after initial integration: **76 passed in 2.64s**.
- Expanded suite with schema-prompt regression, interruption recovery, scope/exception context, PDF-to-training, output lock, and recursive multi-document checks: **82 passed in 2.73s**.
- Final verification after documentation/status and interruption-accounting changes: **82 passed in 2.79s**, five upstream SWIG deprecation warnings, no failures or skipped tests.
- `uv sync --locked` succeeded. CLI help exposed the required single-command interface and input/output/chunk-limit options.

The 24 new end-to-end test cases use explicit mocked API responses. They cover the actual compiled graph; full accepted/rejected exports; candidate and response schema failures; source-reference and extractive-evidence rejection; separate grounding rejection; bounded JSON/transport retries; transport circuit breaking; all-rejected failure; mixed accepted/rejected success; exact global duplicate rejection; formatting warnings allowed; distinct response budget; resume without extra calls or duplicate output; changed inputs; corrupt checkpoints; interruption recovery; document failures; oversized clauses; progress/error exit codes; credential suppression; policy-wide context; nested multi-document discovery; a synthetic PDF; and output locking.

These mocked tests do not prove real endpoint compatibility. That was checked separately below.

### 10.8 Real LLM integration: failed initial attempt, correction, and completed run

An unchanged copy of the existing four-page privacy policy was placed in `artifacts/integration_inputs/privacy_policy/`. The original under `dumy_policies/` was not modified. Executed the full real CLI:

```sh
uv run llmops pip1_data --input ./artifacts/integration_inputs/privacy_policy --output ./artifacts/pipeline1_integration --max-tokens 2048
```

The first live attempt revealed a prompt/schema mismatch not exposed by mocks: the endpoint generated `question` instead of the canonical `instruction` key because the original envelope's object schema was too generic. The run was deliberately interrupted before validation/acceptance. It recorded **FAILED**, five saved generated candidates, zero accepted records, six started model calls, and 13 incomplete mandatory chunks. It did not complete validation; the original interrupted report's zero rejected-example count does not mean those candidates passed. The failed attempt's report and exports were preserved under `artifacts/pipeline1_integration/attempt1/`.

The generation prompt was corrected to include the exact nested Candidate schema and explicitly require `instruction`. A regression test now inspects that prompt contract. The prompt hash selected a fresh run, so malformed first-attempt candidates were not reused. Interruption export accounting was also strengthened so future generated-but-unvalidated candidates are explicitly rejected as incomplete.

The corrected real run completed with:

| Metric | Actual result |
| --- | ---: |
| Documents processed | 1 |
| Physical source pages | 4 |
| Chunks generated | 13 |
| Training examples generated | 13 |
| Accepted examples | 12 |
| Rejected examples | 1 |
| Failed chunks | 0 |
| Real model requests in completed invocation | 26: 13 generation + 13 review |
| Final status | **COMPLETED_WITH_WARNINGS** |
| CLI exit code | 0 |

The real rejection was the Monitoring/Assurance/Records question asking for compliance demonstration and record-retention requirements. Its quoted answer was supported, but the separate review flagged `failed_answers_question` and `failed_preserves_conditions`: it omitted the minimum expected records and the requirement to provide accurate evidence without suppressing/altering/fabricating records. This is one observed validation decision, not a measured general validator accuracy.

The live subset retained 52 quality findings (7 marked corrected, 45 unresolved), including 34 suspicious bullets and two table-spacing findings. These explain the warning status and remain in the data's metadata. The full ten-document live generation was not executed, and these subset counts must not be presented as full-corpus generated data.

A second execution of the same corrected command verified real resumability: **13 generation checkpoints and 13 validation checkpoints reused, zero additional model calls, 12 accepted/one rejected retained without duplicate committed output**. `report.json` now describes this latest invocation and therefore shows zero calls. The first completed invocation's 26-call evidence was saved at:

`artifacts/pipeline1_integration/_runs/6d5dfa4e45a7acdc3c5cd4a685fb11eefbe97b6d3cbbec3625def4d9cfa08474/verification/first-completed-report.json`

Inspection validated every accepted line with the canonical Pydantic record schema, checked assistant-answer equality to its stored evidence after whitespace normalization, all positive stored review flags, source/context references, unique example IDs, and published output hashes. SHA-256 checks verified `.env` and all ten original policy PDFs unchanged. This is actual integration and structural/evidence verification, not mocked execution or an assertion of corpus-wide semantic correctness.

### 10.9 Current limitations and research contribution

The workflow now produces automatically validated, source-traceable conversational records through one command. The conservative extractive design makes evidence review and reproducibility easier, while limiting paraphrase diversity. Some accepted records retain formatting artifacts, including merged table words and `G` bullets. A verbatim response can still be semantically incomplete for its question; the separate reviewer is fallible and shares the generator's model. Scope/exception discovery depends on recognizable heading titles and does not resolve every cross-section dependency. Global exact deduplication is not semantic near-duplicate detection.

The response/context budget is endpoint dependent; combined system, source, and schema prompts may exceed a smaller server context even when chunks fit their byte budget. Execution is synchronous and API calls can be costly/slow for the full corpus. Per-file checkpoint/publication semantics, version management, source-location-sensitive IDs, inherited PDF extraction limitations, and untested cross-platform behavior remain relevant. Five upstream SWIG warnings remain nonblocking. No independent human review, semantic benchmark, training/evaluation split, or fine-tuning performance result is claimed.

For the dissertation, this integrates Corporate Data Transformation into a reproducible, auditable workflow: original source → normalized clauses → generated conversations → explicit validation/rejection evidence. It supports the next fine-tuning pipeline's input preparation without implementing that pipeline. The next authorized step should define dataset acceptance/sampling criteria and the later model-specific conversion/evaluation plan before fine-tuning begins.

## 11. Test-Run Document Limit and the Dirty-to-Clean Data Transformation

### 11.1 Optional document selection

Added the optional `--limit-documents` parameter to the complete Pipeline 1 CLI. Example:

```sh
uv run llmops pip1_data \
  --input ./dumy_policies \
  --output ./artifacts/pipeline1_test \
  --max-tokens 2048 \
  --limit-documents 1
```

Omitting the option preserves full-corpus processing. The positive limit counts supported source-file attempts before parsing and generation, rather than counting successful outputs. Unsupported files/symlinks do not consume a slot; an empty, corrupted, or unreadable supported file does consume a slot and is reported as failed. The implementation does not silently replace a failed first document with another valid one. All chunks of each selected document pass through the normal workflow, so one selected document is not one LLM request.

Selection is deterministic: sorted filenames in each visited directory, followed by sorted subdirectories. A limit larger than the available supported input is allowed. Discovery and run fingerprinting still inspect the input tree. The option does not modify or relocate source documents.

The limit is part of checkpoint identity when supplied. A limited run and an unlimited run have different fingerprints; identical limited runs can resume. Existing unlimited fingerprints remain compatible. The JSON execution report and automatic research-report entry record the selected scope, with `limit_documents`, supported-discovered and selected counts, excluded-by-limit counts, and exact selected/excluded source paths. Intentional exclusions are not failed documents. A limited run is evidence only for its selected subset.

A separate test output directory is recommended because public exports in the chosen destination are replaced with the current run's results. Earlier checkpoint directories remain available. This prevents confusing a one-document test export with full-corpus training data.

### 11.2 Meaning of “dirty data” and “clean data” in this dissertation

The original synthetic policies are authoritative inputs for this research workflow. “Dirty” here describes their extracted representation and generated candidates, not an assertion that the original policy requirements are wrong. PDF conversion introduces glyph, spacing, pagination, and layout artifacts; LLM-generated questions/answers can introduce unsupported claims or omit relevant conditions.

“Clean data” is an operational category: a record has passed the implemented, documented structural, evidence, provenance, grounding-review, and duplicate checks. It is not a claim of perfect extraction, legal validity, complete semantic correctness, or demonstrated fine-tuning benefit. Accepted records can retain formatting warnings and still require human sampling before training.

### 11.3 Transformation from raw source to accepted data

| Stage | Dirty-data concern | Implemented transformation or gate | Result and retained evidence |
| --- | --- | --- | --- |
| Discovery and parsing | Mixed/unsupported inputs, empty/corrupted documents, layout encoded as PDF drawing/text operations | Recursive supported-file discovery; explicit failures; PDF Markdown plus raw per-page text; deterministic IDs and byte hashes | Immutable source references and parsed JSON; failures are visible rather than silently dropped |
| Quality assessment | `G` bullet artifacts, merged table words, repeated page text, irregular whitespace, uncertain boundaries | Structured quality findings with original character/page locations | Explicit uncertainty, not speculative rewriting of policy words |
| Safe normalization | Repeated headers/footers and harmless whitespace interrupt policy clauses | Audited whitespace changes; document-aware allowlisted boilerplate exclusion; explicit Unicode-marker normalization; conservative continuation joins | Separate original/normalized text and line-level actions; excluded content remains recoverable |
| Structure-aware chunking | Arbitrary size/page cuts can detach requirements from exceptions or approvals | Section hierarchy and complete-clause grouping; preserved parent context; source/page spans; no overlapping body text; oversized units flagged | Traceable source chunks; mandatory words, conditions, thresholds, and approvals are retained mechanically |
| LLM generation | Unsupported paraphrases, invented values, misleadingly narrow answers | Generate a specific instruction with a complete extractive source-passage answer; include recognized policy-wide scope/exceptions | Candidate conversation linked to a known document, chunk, and passage; generation alone is not acceptance |
| Deterministic validation | Wrong schema, invented references, altered/partial quotations, credential-like output | Pydantic checks; identity/membership checks; full-passage equality after whitespace normalization; secret suppression | Invalid candidates receive explicit rejection reasons |
| Grounding review | An exact quote can still omit other requirements relevant to the question | Separate model-assisted check of grounding, relevance, conditions, and scope against the full chunk and policy-wide context | All strict review flags must pass; uncertainty/rejection remains recorded |
| Deduplication and export | Repeated examples and duplicated committed output during resume | Global normalized question/answer deduplication; checksummed checkpoints; deterministic replacement exports | Accepted `training.jsonl`, separate `rejected.jsonl`, and an actual-count/hash execution report |

This transformation never changes MUST to SHOULD, removes NOT, invents an approval, or silently changes a numeric threshold. Whitespace normalization can alter line breaks without altering those words. Unknown table spaces and ambiguous `G` glyphs are retained with warnings rather than guessed. Original policy files, original extracted content, and provenance remain available to audit any accepted answer.

### 11.4 Verified example and remaining uncertainty

The previously completed real privacy-policy integration is an observed example of the cleaning gates, not a new live run in this change: 13 candidates were generated, 12 passed automated acceptance, and one was rejected. The rejected response quoted supported text but did not fully answer the question about evidence/record-retention obligations; it omitted the minimum records and accurate-evidence requirements. This demonstrates why “copied from source” is necessary but insufficient for acceptance.

Similarly, four cross-page joins and 80 repeated-boilerplate line exclusions were recorded in the earlier full-corpus chunking run. These are verified earlier normalization actions, not newly executed fixes in this task. The 340 suspicious bullets and suspected table compaction reported earlier were not automatically declared resolved.

The model-assisted reviewer uses the same configured model as the generator, so correlated mistakes remain possible. Exact deduplication does not remove every semantic near-duplicate, and recognizable heading titles do not resolve all global policy dependencies. Final dataset acceptance should therefore include targeted human review, especially for ambiguous table contents, cross-section exceptions, and complex conditional requirements. No semantic-accuracy percentage or fine-tuning result is inferred from the automated pass count.

### 11.5 Changes and actual verification

Modified ingestion and its Pydantic summary schema, pipeline selection/reporting/fingerprint handling, CLI options, end-to-end tests, README, `docs/pipeline1.md`, and this report. No new dependencies or automatic text-repair rules were added.

Executed `uv run llmops pip1_data --help`: the positive optional document limit is displayed. Executed the full automated suite: **89 passed in 2.72s**, with five existing upstream SWIG warnings and no failed/skipped tests. Seven new cases cover limiting before parsing/LLM work, unsupported inputs not consuming slots, deterministic nested selection, resumed limited runs versus unlimited identity, failure consuming the first slot, limits larger than input, CLI propagation/reporting, and invalid zero/negative values (including parametrized cases).

Also executed the existing ingestion component directly against `dumy_policies/` with a limit of one, writing only to `artifacts/pipeline1_limit_check/parsed/`. Actual results: **10 supported documents discovered, one selected and successfully parsed, nine intentionally excluded**. The selected document was `POL-AI-005_Artificial_Intelligence_and_Large_Language_Model_Acceptable_Use_and_Governance_Policy.pdf`. This check made **no LLM requests**. End-to-end option behavior was verified with mocked responses; no additional live generation result is claimed for this change.

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
