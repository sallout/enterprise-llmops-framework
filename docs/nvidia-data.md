# External Data — NVIDIA native preparation

`nvidia_data` completes the external data stage without an LLM, GPU, Docker or full
normalized dataset download/export. It prepares a reproducible selection from the
original NVIDIA datasets, checks it against every prepared enterprise split, and
writes a source plan that can replay the accepted records later. Model training,
model loading, optimizer state and source-mixing ratios are separate future work.

## Small local verification

```bash
uv run llmops nvidia_data \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/nvidia_local \
  --max-samples 20 \
  --max-records-per-dataset 30 \
  --shuffle-buffer 2 \
  --max-tokens 4096
```

`--max-samples` is the total accepted cap across both sources, default 10000. By default
the allocation is equal (10 + 10 in this example); unspecified allocations share the
remaining total. Examples:

```bash
# Explicit accepted allocation across the two sources.
uv run llmops nvidia_data --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/nvidia_server \
  --max-samples 10000 --allocation nemotron=5000 --allocation opencode=5000

# One dataset / a different inspected configuration.
uv run llmops nvidia_data --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/nvidia_math --dataset nemotron \
  --nemotron-config math --max-samples 100
```

The optional candidate cap limits committed sampled candidates per dataset; it does
not limit reader/Arrow/HTTP read-ahead. Without a candidate cap the scan limit is the
source accepted quota multiplied by `PIPELINE3__SCAN_MULTIPLIER` (default 10). Rejected
rows do not count toward the accepted cap. Exhaustion/candidate limits can produce a
reported shortfall. A positive-quota source producing no accepted records fails the
stage; the command never invents missing answers to fill a quota.

`--dataset` can repeat `nemotron` / `opencode`. Default configuration is Nemotron
`chat`, OpenCode `train`; available configurations are inspected, not assumed equal.
`--revision ALIAS=COMMIT` pins explicitly. Otherwise the first metadata resolution
pins a SHA and is cached. Use a new output for a deliberately refreshed source pin.
`--seed` defaults to 42. Sampling is a deterministic, bounded prefix-buffer shuffle,
not uniform sampling over the whole upstream corpus. `--shuffle-buffer` is bounded
1–128; no complete dataset shuffle/download is performed.

The enterprise input can be the local prepared artifact directory, or downloaded
prepared Hub files containing `manifest.json` and the three split JSONLs. The latter
need not include generation `report.json`. Content hashes, counts, split labels,
readiness and original tokenizer/template identity must match. The native stage derives
its tokenizer revision from the enterprise manifest/report rather than floating `main`,
so the same inputs work on a new server. `.env` may provide `HF_TOKEN` for authenticated
Hub access. LLM endpoint credentials are not needed; credentials are never serialized.

## What the workflow checks

A LangGraph runs discovery/pinning → enterprise indexing → native adaptation/filtering
with transactional checkpoints → source plan and report. Graph state contains stage
and artifact references, not dataset bodies. Inspection uses Hugging Face metadata
and pinned cards/licenses; actual files are streamed using the existing packaged
JSON/Parquet readers, with no remote dataset Python scripts or code execution.

- NVIDIA-specific normalization reuses the existing Nemotron/OpenCode adapters,
  preserving roles, multi-turn dialogue, supported tools, native IDs, revisions,
  original test data/results and source fields. Externally supplied tests are not
  locally executed or upgraded to local verification.
- Explicit configured rules strip only closed leading marked reasoning or reject
  reasoning/incompatible content. Transformation hashes are recorded. Missing text is
  never guessed. A separate byte guard bounds raw records/normalization memory.
- The original pinned Qwen chat template counts exact model-facing tokens without
  truncation. Schema, secret/identifier, tool protocol and source references are checked.
- The shared fingerprint gate preloads ALL enterprise train/validation/test records.
  External training candidates conflicting exactly, approximately or by source group
  with holdouts are excluded. Same-training-set duplicates are excluded across both
  native sources. A declared holdout in a training stream fails closed; holdouts are
  never silently relabeled. Unknown benchmark contamination is not semantically proven
  absent by these checks.

## Files, recovery and portability

```text
artifacts/nvidia_local/
├── source_plan.json         # Native revisions/configs, approved selection, tokenizer/settings
├── report.json              # Actual counts, warnings, rejections, transforms and recovery evidence
├── _metadata/               # Checksummed inspected metadata / pinned dataset card snapshots
├── _tokenizer/              # Tokenizer/config assets only; no model weights
└── _runs/<fingerprint>/
    └── fingerprints.sqlite3 # Leakage index + accepted/rejected source coordinates and hashes
```

No new NVIDIA `training.jsonl` is generated. The SQLite decisions contain positions,
original stream ordinals, raw hashes, normalized IDs, token counts/hashes, transformation
hashes and reasons, not full questions/answers/tests. Enterprise/source hash fingerprints
also avoid embedding full record bodies. Disk usage still grows with shingle/decision
counts; provision storage for large selections. Counts named `decoded_*` in the report
measure records/bytes delivered to the sampler, including buffer/replay, not hidden
reader decoding or network transfer volume. Network bytes are not measured.

Repeat the same command to resume automatically: fingerprints include enterprise
split hashes, source revisions, settings, adapter code and dependency versions.
Matching completed accepted selections do not reread source rows. Interrupted/network
reads restart the pinned seeded stream, verify committed ordinal/raw hashes and skip
committed decisions before continuing. Gate entries and decisions commit together, so
retries never append duplicate approved output. Retries are bounded and provider
exceptions are reduced to safe categories. Shortened/changed replay, metadata/checkpoint
corruption or changed enterprise bytes fail closed. A changed finalized checkpoint is
not silently trusted. Existing enterprise/coding/export directories cannot be used as
native plan outputs. Earlier fingerprints remain isolated if settings change.

To reuse a local test on a server, copy the entire native plan folder (including its
matching `_runs/` decision database) and the prepared enterprise split files/manifest,
then install the same project code and locked dependencies with `uv sync --locked`.
Do not copy the Mac virtual environment. Credentials belong in the server environment,
not in those artifacts. The source plan uses relative decision-database paths and
content hashes; original document/repository provenance paths remain informational.
The same server command with a NEW output can prepare a larger selected sample.

## Consumption API (data loader only, not a trainer)

`iter_prepared_native(plan_path, settings, gate, tokenizer=...)` in
`pipelines/native_datasets/replay.py` replays only approved source coordinates.
Before using it, register the exact enterprise splits and all other selected evaluation
sets in a FRESH shared `TrainingInputGate`, then seal holdouts. The function requires
matching enterprise split hashes, near-duplicate threshold, exact tokenizer/template,
code/dependency versions, decision database hash and selection digest. Each native row
is reread from its pinned source, raw-hash checked, normalized and retokenized; IDs and
token hashes must match the plan before passing the fresh global gate. It refuses
altered, missing or newly incompatible content. Native metadata/test evidence is
available in the yielded canonical record but stays outside model-facing messages/tools.

A failure during training-time replay is propagated; the loader does not silently
re-yield already consumed records or manage optimizer checkpoints. Future trainer
code must own training cursors, reproducible mixing weights, complete selected-source
preflight and all chosen holdout registration before optimization. Adding a new source
or holdout requires a corresponding shared audit; this is not a full-corpus certificate.
No semantic correctness, code-test execution, server-scale performance, GPU training,
security audit or benchmarking is implied by a successful small local test.

## Freeze committed progress after interruption

Do not manually change FAILED/RUNNING to COMPLETED or reduce the old command's cap: the cap is part of the original fingerprint. After stopping the writer, create a separate bounded selection from its committed decisions:

```bash
uv run --locked llmops finalize_native_selection \
  --input ./artifacts/opencode_full \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/opencode_77k \
  --max-tokens 4096 --seed 42
```

The original token limit, seed, settings, code, source pins and runtime must match. The command locks the source, checks SQLite integrity, copies the database through SQLite backup and copies cached metadata/tokenizer assets, verifies contiguous/checksummed decisions through existing discovery, reconstructs the exact original fingerprint, and calls the existing finalizer without source streaming. Only an interruption or a stopped RUNNING report without other recorded errors is eligible. The original report and checkpoints remain unchanged; interrupted evidence is retained in the new output. Zero accepted rows, ambiguous runs, changed settings and other processing failures are refused. Existing output cannot be overwritten.

The new plan declares `partial_selection: true` and preserves the original requested cap; it does not claim that all requested examples were processed. Use its actual `accepted_examples` count. A training-ready partial plan does not certify semantic correctness or replace full replay validation before training. No GPU, model weights, new dataset ingestion or training is used by finalization. Further skill profiling must examine the approved records; source-coordinate metadata alone does not establish their skills.
