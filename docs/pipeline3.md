# Pipeline 3 — External Dataset Ingestion & Preparation

> **Current workflow:** enterprise preparation is automatic; see [enterprise guide](enterprise.md).
> Native NVIDIA streaming adapters do not require an exported/merged external dataset.
> The Pipeline 4 CLI is retained only for compatibility; earlier descriptions below
> also document the internal engine and historical workflows.


> **Current export contract:** see [shared schema and portability](data-portability.md).
> New exports use `unified-qwen-sft-v1`; previous native schema descriptions below
> also document retained internal/legacy records. Existing exports are not migrated.
> Rerunning into an existing exported output now requires explicit `--overwrite`.


## Run

```sh
uv sync --locked
uv run llmops pip3_datasets \
  --output ./artifacts/pipeline3 \
  --max-samples 10000
```

This command processes external data independently of Pipeline 2's code-execution readiness. It requires neither an LLM endpoint, Docker, GPU, nor a Qwen/model/tokenizer download. Hugging Face Datasets streams a bounded sample, never an eager full-dataset download. Hub metadata/card requests and library read-ahead still transfer data. Private/gated access uses `HF_TOKEN` through environment or Pydantic Settings; credentials are not persisted in artifacts.

Default accepted caps are 5,000 per dataset. The cap applies to accepted records, not raw rows. Rejections do not consume accepted quota. No dataset exceeds its allocation; unused quota is reported rather than silently reassigned. Exhaustion or the bounded candidate limit can produce fewer records. This is ingestion sampling, not Pipeline 4's final mixing ratios or training splits.

Examples:

```sh
# Optional custom allocation; unspecified datasets share the remaining total.
uv run llmops pip3_datasets --max-samples 10000 \
  --allocation nemotron=6000 --allocation opencode=4000

# Single dataset; full Hub IDs are also accepted.
uv run llmops pip3_datasets --dataset opencode --max-samples 100

# Select multiple Nemotron configurations (default is chat).
uv run llmops pip3_datasets --max-samples 100 \
  --config nemotron=chat --config nemotron=instruction_following --seed 42

# Inspect and select all declared Nemotron configurations.
uv run llmops pip3_datasets --max-samples 100 --config nemotron=all

# Re-resolve current revisions; ordinarily existing pinned metadata is reused.
uv run llmops pip3_datasets --max-samples 20 --refresh-metadata
```

Repeat `--revision ALIAS=commit_or_tag` or `--split ALIAS=split` to override selection. Requested references resolve to exact Hub SHAs before data ingestion. Configurations and splits must match inspected card declarations. Allocation is divided evenly among selected configurations; if the allocation is smaller than their count, some configurations have zero quota and are not streamed. The default does not claim coverage of all eight Nemotron domains.

## Inspected sources and licensing

Sources inspected on 2026-10-26:

| Source | Inspected revision | Configurations / split | Published license |
| --- | --- | --- | --- |
| [Nemotron-Cascade-2-SFT-Data](https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data) | `9f36020daf067f1a8b39336bf619fe30af30bb02` | math, science, chat, instruction_following, safety, conversational_agent, swe, terminal_agent; train | Card says `license: other`, `license_name: nvidia-open-model-license` |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` | train / train | CC BY 4.0 |

The Nemotron card provides conversation fields `domain`, `source`, `messages`, and `generator`. Bounded first-record inspection of every published JSONL file confirmed those keys, variable turn counts, blank system messages, and marked reasoning content. These previews do not characterize the full corpus. OpenCodeInstruct provides `id`, `input`, `output`, `domain`, `generation_algorithm`, `llm_judgement`, `unit_tests`, `tests_execution_status`, and `average_test_score`. Its card declares the last field as a string, while a real streamed row supplied a float; source metadata preserves the received value without pretending the card and stream agree on every type.

OpenCodeInstruct's [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) require appropriate credit, a license link, and indication of changes. Preserve NVIDIA authorship, dataset title/source, attribution/citation, license, and normalization history in downstream artifacts and redistribution notices. The full card snapshot and per-record provenance are retained for that purpose; metadata alone does not automatically satisfy every downstream publication obligation.

Nemotron points to the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). The linked agreement includes conditions of use, Trustworthy AI terms, redistribution notices, and separate-component terms. Its model-oriented wording is retained as the publisher's dataset license declaration, not reclassified as CC BY or a blanket certification of rights. Review applicable source/component notices before redistribution. The card is pinned; externally linked terms may change independently. Pipeline 3 records declared licensing and rejects unrecognized/missing license metadata; it does not determine legal suitability or automatically accept gated terms.

## Architecture

LangGraph executes metadata discovery → streaming ingestion → dataset-specific normalization → structural/quality filtering and exact deduplication → JSONL export → research/statistics reporting. Ingestion/normalization/filtering loop per record, committing each decision transactionally before proceeding. Graph state contains stage names and artifact paths; record bodies and buffers are not placed in graph state.

Modules under `src/policy_dissertation/pipelines/external_datasets/`:

- `sources.py`: Hub metadata, license/config/split inspection, immutable file URLs, bounded metadata retries, and Hugging Face Datasets packaged JSON/Parquet stream readers.
- `sampling.py`: seeded bounded buffer, original stream ordinals, read counters, per-record size guards, resource cleanup.
- `schemas.py`: typed conversational, source metadata, and tool-call contracts.
- `normalization.py`: dataset-specific field mapping, explicit reasoning/token rules, exact provenance and transformation hashes, no generated answers.
- `storage.py`: SQLite transactions, accepted-content uniqueness, rejection audit references, and progress counters.
- `pipeline.py`: graph, bounded stream retries, deterministic replay/resume, allocation, atomic export, actual statistics, root report updates.

The common `ConversationArtifact` base supplies the same `schema_version`, `example_id`, `messages`, and `metadata` layout. Existing Pipeline 1/2 records retain their exact three-message contract. Pipeline 3 specializes the base for optional original system messages, multiple turns, and native tool messages. It adds no fabricated system prompt, joins no turns, and executes no tools. Pipeline 4 must dispatch the schema version and preserve the message sequence when applying the Qwen chat template.

## Canonical schema

See [the saved JSON Schema](schemas/pipeline3-training.schema.json). A minimal shape is:

```json
{
  "schema_version": "external-datasets-v1",
  "example_id": "deterministic SHA-256",
  "messages": [
    {"role": "user", "content": "Original source question"},
    {"role": "assistant", "content": "Original final answer"}
  ],
  "metadata": {
    "dataset": "nvidia/OpenCodeInstruct",
    "config": "train",
    "split": "train",
    "revision": "resolved Hub SHA",
    "record_id": "upstream ID or row:<ordinal>",
    "record_id_kind": "upstream",
    "stream_ordinal": 0,
    "content_hash": "hash of original complete record",
    "source_url": "pinned dataset tree URL",
    "license": "cc-by-4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "source_fields": {},
    "source_message_metadata": {},
    "transformations": [],
    "token_count": 123,
    "token_method": "utf8_serialized_messages_bytes_v1",
    "locally_executed": false,
    "external_test_results_status": "not_locally_verified",
    "semantic_status": "structural_checks_only"
  }
}
```

Source fields retain original unit-test text, reported test results/scores, judgments, domain, generator, and tool definitions where present. They remain metadata rather than unsolicited model-facing text. Message extras are kept under indexed source-message metadata. Tool call IDs, function names, original arguments, result references, and contents are preserved. Tool arguments must be JSON objects, calls/results must match, and the conversation must end with a completed assistant answer. Unsupported modalities/roles, invalid tool structures, and unfinished answers are rejected rather than guessed into a different format. Blank original system content is allowed; empty user/assistant answers are not.

## Sampling, limits, and filtering

A seeded buffer samples/reorders a bounded prefix of each selected stream. This provides reproducibility for the same source revisions, seed, allocation, settings, and dependency/code versions. It is **not a uniform random sample of the whole dataset**. Original stream ordinals, not sampled positions, identify rows without upstream IDs. No skip to a random end-of-corpus offset or full scan occurs by default.

Pydantic `PIPELINE3__*` settings can be overridden through environment or `--filter-config path.json`. JSON overrides merge with settings and reject unknown keys. Defaults:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `reasoning_policy` | `strip_marked` | Alternative `reject` rejects marked reasoning, including nonempty reasoning fields |
| `max_tokens` | 32768 | Conservative byte count of serialized normalized messages, not exact Qwen subword tokens |
| `min_assistant_chars` | 1 | Minimum nonwhitespace final-answer length |
| `max_record_bytes` | 1048576 | Oversized raw records rejected without retaining large text |
| `shuffle_buffer` | 16 | Maximum buffered decoded candidates (range 1–128) |
| `scan_multiplier` | 10 | Maximum processed candidates per source = accepted quota × multiplier |
| `max_attempts` | 3 | Bounded metadata/read retries |
| `request_timeout` | 30 | Hub metadata / HTTP client timeout in seconds |
| `forbidden_tokens` | listed in config.py | Explicit incompatible template/control token strings |
| `unsupported_reasoning_markers` | `<analysis>`, `</analysis>` | Reject these in assistant content; configurable |

For strict exclusion instead of extraction:

```json
{
  "reasoning_policy": "reject",
  "shuffle_buffer": 4,
  "max_tokens": 32768,
  "min_assistant_chars": 1
}
```

The default removes only complete leading `<think>…</think>` blocks from assistant content, and explicit `reasoning`/`reasoning_content`/`analysis` fields. Each removal records original/result hashes and location. Nested, unclosed, nonleading, and empty-final-answer cases are rejected. Ordinary prose/code that mentions reasoning is not heuristically rewritten. The library's own implicit agent-trace parsing is disabled, so these transformations remain explicit. Source originals remain retrievable at pinned revisions and are identified by full-record hashes; removed reasoning is not copied into accepted or rejected artifacts.

No truncation, guessed answers, or invented execution evidence is allowed. Byte estimates count tool-call payloads along with conversation text. They exclude metadata, future tool-definition injection, and the final Qwen chat template; recheck exact tokenizer/context limits during later training preparation. Exact complete-conversation duplicates are removed across the selected external datasets using a disk-backed index. Whitespace inside code is preserved; near-duplicates, paraphrases, benchmark leakage, and cross-Pipeline 1/2 duplicates are not detected here.

## Outputs and recovery

```text
artifacts/pipeline3/
├── training.jsonl            # Accepted canonical records
├── rejected.jsonl            # Source identity/hash, ordinal, size, and rejection reasons
├── report.json              # Actual run counts, source revisions/cards/licenses, filters, hashes
├── _metadata/               # Checksummed public inspection snapshots
└── _runs/<run-id>/
    ├── manifest.json
    ├── decisions.sqlite3    # Transactional decisions, exact-dedup index, counters
    ├── metadata/<alias>/README.md
    └── last_execution_report.json
```

Rejected text is not copied in full: pinned provenance/hash allows retrieval while keeping artifacts bounded and avoiding persistence of unwanted traces. The decoder may read a large individual row before its size guard applies. Datasets/Arrow/fsspec may fetch chunks/row groups ahead; decoded rows read are counted, including sampler buffer/replay, while actual network bytes and all library-prefetched rows are not measured. This is not a claim of a strict network-byte or peak-memory limit.

SQLite transactions commit accepted/rejected decisions and counts together. Output locking prevents concurrent writers. Public JSONL files are streamed to temporary files and atomically replaced; report hashes/status describe committed exports. File publication is not a whole-directory transaction, so consumers must check final status and hashes. Repeating a completed command re-exports checkpoints with zero data-stream reads and identical training/rejection bytes. Requested moving branches reuse the pinned snapshot until `--refresh-metadata`; explicit SHAs are recommended for experiments.

On an interrupted/network-failed source, the same seeded sampler replays from the beginning, verifies every committed decision's raw hash, and resumes after committed work. This avoids duplicates but can reread data; no claim of zero-cost random seek is made. A drifted pinned stream fails rather than mixing content. Missing metadata/access failures are retried on later runs; successfully pinned metadata is reused. If one dataset is inaccessible, accessible sources can still produce auditable partial artifacts, but mandatory access failure makes status `FAILED` with nonzero CLI exit.

Statuses: `COMPLETED` for completed targets without rejected/modified records; `COMPLETED_WITH_WARNINGS` for valid output with rejections, tracked normalization, or bounded shortfall; `FAILED` for operational failures or no accepted records. Accepted means the documented automated structural/filtering gates passed. It does not certify answer correctness, code execution, license suitability, dataset representativeness, or fine-tuning benefit.

## Verification

```sh
uv run pytest -q
PIPELINE3__SHUFFLE_BUFFER=4 uv run llmops pip3_datasets \
  --output ./artifacts/pipeline3_integration --max-samples 6
# Repeat the exact command to test completed-run recovery.
```

See the root [research report](../report.md) for executed results. Synthetic stream tests cover both adapters, multi-turn/tool preservation, configured reasoning/token filters, malformed/empty/oversized records, global duplicates, allocation limits, reproducibility, exhaustion, transient/midstream errors, access failure, interruption recovery, stream drift, export failure, and CLI behavior. They do not substitute for the recorded real ingestion trial. Pipeline 4 mixing/splitting and all model fine-tuning remain outside scope.
