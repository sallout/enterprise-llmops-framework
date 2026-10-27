# Pipeline 4 — Unified Dataset Validation & Training Preparation

## Run and readiness gates

```sh
uv sync --locked
uv run llmops pip4_prepare \
  --input ./artifacts \
  --output ./artifacts/pipeline4 \
  --max-tokens 4096
```

The command reads exactly `pipeline1/training.jsonl`, `pipeline2/training.jsonl`, and `pipeline3/training.jsonl` under the input root, plus each pipeline's `report.json`. It does not recursively glob old runs, regenerate upstream datasets, invoke LLMs, execute repository code/tests, download model weights, or train a model. Docker, PyTorch, and GPUs are unnecessary.

Normal preparation requires all three upstream runs to report `COMPLETED` or `COMPLETED_WITH_WARNINGS` without errors, with matching export SHA-256 hashes and record counts. Each accepted input is independently revalidated. Missing/failed inputs, inconsistent hashes, unavailable actual tokenizer assets, inadequate independent split groups, or detected leakage produce `FAILED` and CLI exit 1. Source files and reports are hashed again before publication to detect concurrent modification. Output paths, including resolved symlinks, cannot overlap the source pipeline directories/files.

**Current actual-input limitation:** `artifacts/pipeline1/` is absent and `artifacts/pipeline2/report.json` reports `FAILED`. Therefore the default command deliberately fails and publishes empty split files with `training_ready=false`. Restore/generate healthy Pipeline 1 exports and complete Pipeline 2 before claiming a complete three-source preparation run. Pipeline 3's 1,000-record export is present and healthy.

For explicit diagnosis only:

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

This permits readable, hash-matching exports from a failed upstream run to be inspected record by record; missing/hash-mismatched sources remain excluded. The flag always sets `training_ready=false`, even if every selected record passes. This output is a diagnostic artifact, not the final dissertation training cohort. A checksum mismatch is never bypassed by the flag.

Optional flags: `--seed` (42), `--max-examples`, repeated `--mix-ratio NAME=value`, repeated `--split-ratio NAME=value`, `--tokenizer-revision`, `--local-files-only`, and `--config path.json`. JSON overrides strictly validate `Pipeline4Settings`; environment uses `PIPELINE4__*`. Ratio assignments merge with defaults and normalize to sum one. Keys must be exactly pipeline1/pipeline2/pipeline3 or train/validation/test. Weights must be finite and nonnegative; training has positive weight.

## Unified schema and adapters

The NVIDIA-inspired envelope adds `id`, `domain`, `source`, and `verification` around canonical `messages` and `metadata`. [The saved JSON Schema](schemas/pipeline4-training.schema.json) defines fields and role/tool structures. Code-verification evidence is separate, following the useful distinction in OpenCodeInstruct between a response, tests, and reported execution.

```json
{
  "schema_version": "unified-qwen-sft-v1",
  "id": "deterministic SHA-256",
  "messages": [
    {"role": "user", "content": "Original question"},
    {"role": "assistant", "content": "Original answer"}
  ],
  "domain": "coding",
  "source": "nvidia/OpenCodeInstruct",
  "verification": {
    "status": "externally_reported",
    "executed_in_pipeline4": false,
    "unit_tests": "original test text when supplied",
    "evidence": {"original_external_test_metadata": {}, "locally_verified": false}
  },
  "metadata": {
    "pipeline": "pipeline3",
    "upstream_example_id": "original ID",
    "provenance": {},
    "input_reference": {},
    "component_group": "stable source group",
    "official_holdout": null,
    "preparation": {},
    "split": "train",
    "split_group": "stable connected-component identifier"
  }
}
```

`tools` is an optional top-level field for original function definitions. Only `messages` and necessary `tools` feed the model template. The original metadata is retained in `metadata.provenance`, with file hash, line, raw-record hash, upstream ID, grouping, and preparation information alongside it. No fabricated system prompt or test outcome is added, and original message order/content remains unchanged.

- Pipeline 1 uses its existing strict extractive-record schema. Document/chunk/source hashes, page references where provided, and equality of the extractive response to compacted source evidence are checked. All original policy conditions/context remain in messages. `verification.status=not_applicable` for runtime execution; upstream policy judgment is retained as evidence, not independently certified.
- Pipeline 2 uses its code-record schema. Repository snapshot/file hashes, relative path, and line ranges are checked. Executable task types require consistent upstream `verified` execution, Docker executor, image SHA, positive tests, exit zero, and zero failures/errors/skips/expected failures/unexpected successes. Mock/unverified execution is rejected. This is inherited upstream evidence, not cryptographic authentication or Pipeline 4 reexecution. Knowledge-only explanations have runtime `not_applicable`. An unambiguous original `Tests:` fenced block is copied into verification when available; no missing tests are generated. Original messages already containing tests/execution text are preserved rather than rewritten.
- Pipeline 3 uses its typed external-record schema and original multi-turn/tool protocol validation. Dataset/config/split/revision/record ID, source URL, content hash, and license provenance are required. OpenCodeInstruct maps to coding with original test/status/judgment metadata labeled externally reported. Nemotron domain comes from its configuration (chat/general dialogue, math/reasoning, science, safety, agentic, etc.). Supplied test results are never promoted to local passes. Tool definitions/call names/results must match and are retained. Unsupported roles, incomplete conversations, and missing tool definitions are rejected.

Verification labels: `upstream_local_pass`, `externally_reported`, `not_applicable`, `unverified`. `executed_in_pipeline4` is always false. Preserve these distinctions in later research comparisons; a passing upstream test is evidence for that test suite, not proof of general correctness.

## Deterministic quality checks

Streaming JSONL loading rejects blank/malformed/non-finite JSON, excessive line sizes, schema/role errors, bad references/hashes/URLs, incomplete tool interactions, embedded model control tokens, and unwanted explicitly marked reasoning. Whole oversized conversations are rejected; nothing is truncated. The existing Pipeline 3 role/order validator is reused without changing that pipeline.

Conservative secret detection scans original messages and auxiliary metadata before persistence: configured credentials, private-key markers, high-confidence API/HF/GitHub/AWS key patterns, and non-placeholder high-variation credential assignments. A configured sensitive-identifier rule flags SSN-shaped values. Matched values are never written to rejection logs. Rejections retain source hash/line/reason only. These checks can flag synthetic credentials and miss other sensitive information; they are neither semantic privacy certification nor a security-auditing pipeline. Email/name/ordinary number strings are not indiscriminately deleted. Configurable settings control the secret/identifier checks and marked-reasoning policy.

Formatting warnings inherited from policy parsing are preserved, not automatically treated as fatal. No source text is repaired or rewritten by Pipeline 4.

## Exact Qwen tokenization

Only allowlisted tokenizer/config files are retrieved from [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B): tokenizer JSON/config, vocab, merges, optional special-token/chat-template files, and small model config. No model weight names are allowed. Asset SHAs and resolved revision are pinned and verified; remote Python execution is disabled. The observed revision is `40c069824f4251a91eefaf281ebe4c544efd3e18`.

Transformers' actual `apply_chat_template` measures the complete conversation with `tokenize=True`, `add_generation_prompt=False`, `enable_thinking=False`, `truncation=False`, and original tool definitions. The pinned Qwen template can add its own empty thinking wrapper to the final answer; Pipeline 4 neither invents a reasoning trace nor manually removes that template behavior. Token counts include this overhead. The snapshot's context guard is also checked (observed config/tokenizer minimum: 40,960); the requested 4,096 limit remains the stricter bound. This does not establish long-context runtime performance.

`metadata.preparation` records exact count, tokenizer revision, template hash, rendered-token hash, maximum, flags, and `truncated=false`. The model-facing token hash also detects identical training sequences that differ only in template-ignored fields such as opaque tool-call IDs. Metadata/verification are excluded from tokenization, except necessary supplied tool definitions.

The [Transformers chat-template convention](https://huggingface.co/docs/transformers/main/chat_templating) must also be used by the future trainer. Do not add a generation prompt to completed SFT conversations, tokenize rendered text twice with extra special tokens, or truncate after preparation. The original template is unmodified; assistant-only loss masking belongs to the downstream training implementation and is not claimed to be generated here.

## Deduplication and provenance

Exact duplicate detection uses the actual rendered-token sequence hash. A deterministic prefix-filtered index finds candidate word-five-gram sets for Jaccard comparison (default threshold 0.90), with full intersection verification. SHA-256 shingle ordering and length bounds make this independent of random LSH recall. A regression compares prefix-filter candidates against brute-force overlaps.

Highly similar records are removed only when they are safe formatting equivalents (default removal threshold 0.98). Code indentation is never whitespace-collapsed for deletion; numbers, punctuation, case, mandatory language, negations, conditions, and answers are not rewritten. Other similar examples remain and their groups are connected to prevent cross-split leakage. Thus similarity does not imply semantic redundancy. Paraphrases outside the configured shingle metric remain a limitation.

Representatives prioritize inherited verified coding evidence, then policies, repository knowledge, and external records. Duplicate provenance remains in `rejected.jsonl` with the representative ID and source references. Up to 100 duplicate-reference previews are embedded in the accepted metadata; a total count and audit pointer preserve bounded memory. Official holdout constraints propagate through removed duplicates to the retained representative/group.

## Mixing and splitting

Default source weights are policy 40%, repository 30%, external 30%. Sampling is without replacement and never repeats scarce examples to manufacture the ratio. The continuous bottleneck `floor(min(available_i / weight_i))`, limited by optional `--max-examples`, sets the cohort cap; Hamilton rounding assigns integer quotas. This conservative cap can leave records unused even when a slightly larger integer-rounded cohort would fit. Requested/actual weights and exclusions are reported explicitly. With only one repository record, the default balanced cohort can be very small; obtain more valid repository data or explicitly choose another research ratio.

Within a source, proportional domain allocation retains at least one example per available domain when quota permits. Verified coding examples are prioritized within their domain; seeded hashes choose among equal-priority records. Partial diagnostic mode renormalizes over sources with eligible data, while continuing to report missing inputs and false training readiness.

Source groups are full policy documents, repository source files as components, and external original-example identities (using dataset/config/revision/ordinal where no upstream ID exists). Exact/near links connect these groups before mixing, including unselected bridge examples. Group statistics and assignments reside in SQLite; messages are not all loaded into RAM. Groups remain intact.

Default split targets are 90/5/5. A seeded greedy deficit objective balances total counts, source, domain, and verification strata subject to groups. Exact ratios are not guaranteed when groups are large. Explicit original test/validation/dev/evaluation labels and benchmark flags constrain held-out placement; test wins over conflicting validation, with conflicts counted. Existing source `train` rows may be repartitioned into new holdouts. Unknown/unmarked benchmark overlap is not magically detected.

Final audits require no component, exact-token, or defined near-pair leakage. Too few independent groups or empty required splits fail safely. The held-out test file is reserved for final evaluation, never for training or hyperparameter selection; Pipeline 4 does neither.

## Outputs, checkpoints, and consumers

```text
artifacts/pipeline4/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
├── rejected.jsonl
├── manifest.json
├── report.json
├── _tokenizer/                 # Small allowlisted tokenizer files and checksums
└── _runs/<run-id>/
    ├── state.sqlite3           # Records, groups, prefix index, decisions
    └── last_execution_report.json
```

`rejected.jsonl` distinguishes invalid records, removed duplicates, and valid examples excluded by mixing. Mixing exclusions are not validation failures. Counts satisfy the successful-run accounting relation: input rows = exported + invalid + duplicate-removed + not-selected.

Input/model/source-code/dependency/settings hashes identify a run. Transactional per-row loading and per-record deduplication resume interrupted work without duplicate output. Completed reruns reuse parsing/tokenization/dedup checkpoints, while rechecking file hashes and recomputing deterministic selection/splits. Each JSONL is atomically replaced; the directory is not a whole-filesystem transaction.

Consumers must check both `report.json` and `manifest.json`, require successful status plus `training_ready=true`, verify output hashes, and apply only messages/tools to the pinned template. A failed export invalidates the manifest. A failed research-report write also clears manifest readiness. Partial-source outputs are never labeled fully ready. `training_ready` describes these preparation gates, not factual/code correctness, downstream trainer implementation, or measured learning benefit.

## Implementation and verification

Nine modules under `pipelines/dataset_preparation/` provide schemas, adapters, deterministic quality rules, pinned tokenizer loading, SQLite storage, deduplication, mixing/splitting, and LangGraph orchestration. Existing pipelines are not rewritten. CLI/Pydantic configuration, atomic file helpers, output locking, and source schemas are reused.

Installed Transformers 5.20.2 and Jinja2 3.1.6; tokenizers 0.23.3 is a dependency. Transformers 4.x conflicted with the project's existing Hugging Face Hub 1.x requirement, so compatible 5.x was selected. Jinja2 is explicitly required for template rendering. The informational “PyTorch was not found” notice does not prevent tokenizer-only preparation. No torch/GPU package or model weights were installed.

See [report.md](../report.md) for actual strict/partial integration results, tests, split sizes, token statistics, source blockers, and reproducibility checks. Synthetic tests do not stand in for unavailable real policy inputs; the real cached tokenizer and available-source integration are identified separately. No fine-tuning, quantization, benchmarking, or security auditing was begun.
