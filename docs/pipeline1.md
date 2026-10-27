# Pipeline 1 data contract

## Architecture

A compiled LangGraph `StateGraph` orchestrates five sequential stages: existing ingestion, existing quality/normalization/chunking, generation, validation, and export. Graph state contains stage names and artifact paths, not document bodies or messages. Source text is loaded locally by nodes and persisted as artifacts. File-based checksummed checkpoints provide per-chunk recovery; this implementation does not use a LangGraph SQLite checkpointer. Parsing and chunking rerun deterministically on resume, while expensive completed LLM work is reused.

`pipeline.py` coordinates stages and reporting. `generation.py` implements LangChain's `ChatOpenAI`, JSON prompts, bounded attempts, secret suppression, and whole-source-passage selection. `training_validation.py` implements evidence gates and model-assisted review. `data_schemas.py` defines Pydantic contracts; `data_io.py` provides atomic files and durable checksummed caches. Existing ingestion/chunking implementations remain reusable independently.

Input/output trees must not overlap. Every discovered supported document is attempted. There is no implicit document or chunk cap. Source/configuration/model-endpoint/prompt/dependency identity contributes to a run fingerprint. The API credential is excluded from the fingerprint and checkpoint content. The schema/algorithm version must be advanced if future validation or passage-selection semantics change; automatic source-code hashing is not used.

## Canonical conversational record

Each UTF-8 JSONL line has:

- `schema_version`: `pipeline1-extractive-v1`.
- `example_id`: deterministic SHA-256 over source chunk/passage identity and messages.
- `messages`: exactly three objects, ordered `system`, `user`, `assistant`, each with string `content`.
- `metadata`: trusted application-built source/evidence/validation fields. LLM-provided arbitrary metadata is not accepted.

The system message contains policy title plus recognized policy-wide Scope/Exceptions text. The user message contains the generated instruction. The assistant message is a complete source passage, with whitespace normalized. This deliberately narrow extractive strategy limits unsupported paraphrasing and preserves mandatory language, conditions, and approvals inside the passage. Leading ambiguous `G` glyphs are retained if present; table word boundaries are not guessed.

Metadata fields include `document_id`, `source_content_hash`, `source_path`, `chunk_id`, `section_id`, `section_title`, `parent_sections`, `page_numbers`, `source_spans`, `passage_id`, `evidence_quote`, `evidence_start`, `evidence_end`, `context_references`, `quality_warnings`, and `validation`. Evidence offsets reference the chunk content; the exact source quote is retained even when whitespace differs from the answer. Source spans connect back to the original parsed page text. Policy-wide context references identify their document/chunk/section/pages.

This `messages` representation is compatible with a later Qwen3-14B chat-template conversion: select the messages and apply the model's tokenizer/template in the next pipeline. Metadata must remain a sidecar or be excluded from model loss. This stage does not choose chat-template special tokens, thinking mode, label masks, truncation, train/evaluation splits, or a trainer. It does not claim that these exports have been tokenized or used successfully for Qwen training. The [official Qwen3-14B example](https://huggingface.co/Qwen/Qwen3-14B) uses role/content messages and `apply_chat_template`.

## Generation and validation

The generator is prompted to emit an `examples` object containing one candidate; the bounded contract permits one to three candidates. Each has exactly `instruction`, `response`, `document_id`, `chunk_id`, and `passage_id`. Endpoint-native JSON-schema/tool support is not required: JSON-only prompting is followed by Pydantic parsing. A complete leading reasoning block or outer Markdown JSON fence can be removed before parsing; incomplete JSON never becomes accepted output.

The model receives complete source passage choices, section context, and policy-wide context. Passage choices retain list clauses with continuation paragraphs; marked conditional continuations remain grouped. Plain prose runs remain intact. This is structural evidence selection, not a general proof of clause independence.

Required gates:

1. Pydantic candidate schema and required field bounds.
2. Document/chunk identity and source passage membership.
3. Answer equality to the entire selected passage after whitespace normalization. No unsupported wording, selective sentence truncation, changed negation, or altered number is allowed by this gate.
4. Credential-like output suppression.
5. Separate model-assisted review against the full chunk and recognized policy-wide context. Strict boolean verdicts must affirm grounding, relevance, condition preservation, and scope. Uncertainty is a rejection.
6. Global exact question/answer deduplication after case folding and whitespace normalization. This is not a near-duplicate/semantic similarity detector; different questions about the same rule may remain.

Failed individual candidates are rejected with explicit reasons. Schema/provenance/extractive failures do not need another paid review call. Verification transport/malformed-output exhaustion is a mandatory processing failure, not implicit acceptance. A valid sibling may survive another candidate's rejection. All candidates rejected causes a failed run; mixed valid/rejected data can complete with warnings. Formatting warnings are propagated as evidence and do not cause blanket rejection.

The generation and review calls use the same configured model. The review is a separate call, not an independent model or human evaluator. Correlated errors, prompt injection, selection bias, source extraction defects, and unresolved cross-section dependencies remain risks. Exact copying proves support for the quoted text, not that the question is semantically sound; model-assisted review reduces but does not eliminate that risk. Human sampling and task-specific evaluation remain necessary before fine-tuning.

## Recovery and committed output

Checkpoints are stored under `_runs/<run_id>/generation/` and `validation/`. Each is keyed by run/chunk content and carries a payload checksum. Successful generation and completed validation outcomes are reused. Corrupt or incomplete checkpoints are recomputed. Process interruption can repeat an uncommitted API call but cannot append duplicate training output. Retrying failed calls can incur additional endpoint cost. No provider idempotency key is assumed.

`training.jsonl` contains only accepted records. `rejected.jsonl` distinguishes `kind=example` from `kind=chunk_failure`; operational failures are not fabricated generated examples. `report.json` separates generated/accepted/rejected examples, failed chunks, rejection-record counts, quality warnings, resumed work, and calls made this invocation. Generation attempts made before an uncommitted crash cannot be reconstructed exactly from surviving files.

Output files are replaced rather than appended; a single output-directory lock prevents concurrent pipeline writers. A report with final status and JSONL checksums is published last. Readers must verify the report status and hashes, especially after termination during publication. File checksums detect corruption, not malicious tampering. Concurrent runs in different output directories are not coordinated when updating the shared research report. Artifacts from older fingerprints remain for recovery and audit.

## Status and limitations

`COMPLETED` means all mandatory stages completed, accepted records exist, and no quality/rejection/unsupported-input warnings remain. `COMPLETED_WITH_WARNINGS` permits accepted records plus retained quality issues or candidate rejections. Both exit 0. `FAILED` exits 1 for parsing/chunk processing failures, LLM/verification exhaustion, all-rejected output, or mandatory report-update failure. Typer/Pydantic configuration errors exit 2. HTTP status, timeout, and connection categories are recorded without raw exception bodies.

The chunk budget uses the existing conservative UTF-8 byte-token upper bound, not a selected model tokenizer. Response limits/timeouts/attempts are configured independently through `PIPELINE1__*`. Unsupported documents, OCR limits, glyph errors, table issues, heuristic section detection, and long indivisible clauses inherit the earlier stage limitations. The workflow is synchronous and sequential; a large corpus can take many API calls. The configured endpoint must have sufficient combined prompt/context capacity. No legal correctness, semantic completeness, fine-tuning benefit, or model performance claim follows from an accepted record.

Technology references: [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api), [LangChain ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai). Only the compatible chat-completions interface is used; provider-specific extensions are not assumed.

## Optional document limit

`--limit-documents N` / `run_pipeline(..., limit_documents=N)` restricts parsing and all downstream stages to the first N supported source files in deterministic sorted directory traversal. Positive integers are required. Omission processes all supported sources. Unsupported files remain reported and do not consume slots; a supported file that fails parsing does consume a slot and remains a mandatory failure. A limit greater than available input simply processes all available supported documents.

The ingestion report now distinguishes discovered files, supported discovered files, selected documents, excluded-by-limit files, and exact selected/excluded paths. Intentional exclusions do not count as failed documents. Ingestion accounting with a limit is `discovered = parsed + failed + unsupported + excluded_by_limit`. All chunks from selected sources remain eligible for normal processing; this option is not a chunk or API-call cap.

The limit participates in the run fingerprint when supplied, preventing a limited run from reusing an unlimited run's state as though their selection were equal. Unlimited fingerprints remain compatible with earlier unlimited runs. Discovery/fingerprinting still scan the source tree. No source files are copied, renamed, or edited to implement selection. Use a distinct output directory for test data if existing full-run public exports must be retained.
