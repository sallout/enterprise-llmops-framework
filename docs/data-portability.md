# Shared training records and portable dataset preparation

> **Current workflow:** enterprise preparation is automatic; see [enterprise guide](enterprise.md).
> Native NVIDIA streaming adapters do not require an exported/merged external dataset.
> The Pipeline 4 CLI is retained only for compatibility; earlier descriptions below
> also document the internal engine and historical workflows.


Pipelines 1–3 now export `unified-qwen-sft-v1` directly in `training.jsonl`.
Pipeline 4 consumes this format as well as the three original pipeline formats.
`src/policy_dissertation/data/schemas.py` is the shared Pydantic contract;
`docs/schemas/unified-training.schema.json` is its generated JSON Schema.
The old schema files remain documentation for legacy records and internal checkpoints.

## Record contract

```json
{
  "schema_version": "unified-qwen-sft-v1",
  "id": "<deterministic SHA-256>",
  "messages": [
    {"role": "user", "content": "A source-grounded question"},
    {"role": "assistant", "content": "The supported answer"}
  ],
  "domain": "corporate_compliance",
  "source": "pipeline1",
  "verification": {
    "status": "not_applicable",
    "executed_in_pipeline4": false,
    "unit_tests": null,
    "evidence": {}
  },
  "metadata": {
    "pipeline": "pipeline1",
    "upstream_example_id": "<original ID>",
    "provenance": {"document_id": "<original reference>"}
  }
}
```

This shortened illustration is not a complete valid provenance record. Actual exports
retain every original provenance field. `id` replaces top-level `example_id`;
`metadata.provenance` contains the original metadata, and
`metadata.upstream_example_id` retains the original identifier. Source messages remain
unchanged. Optional `tools`, tool calls/results, multi-turn conversations, tests,
external results, dataset revisions, license declarations and transformation history
are preserved. Metadata is not inserted into the model-facing messages.

Verification statuses distinguish `not_applicable`, `unverified`,
`externally_reported`, and inherited `upstream_local_pass`. The last requires the
existing Pipeline 2 restricted-Docker evidence checks; Pipeline 4 neither reruns nor
authenticates those executions. External NVIDIA test statuses are never promoted to
local passes. The legacy `executed_in_pipeline4=false` field is retained for compatibility.

The shared adapter revalidates unified records against their original provenance;
changing the claimed source, ID or verification without corresponding valid evidence
is rejected. Policy exact-evidence and repository execution gates remain active.
Native NVIDIA adapters reuse Pipeline 3's existing normalization and license rules.
No additional LLM transformation is used for dataset transport.

## Compatibility and overwrite protection

Existing generated files are not migrated in place. Pipeline 4 accepts legacy
`pipeline1-extractive-v1`, `repository-data-v1`, and `external-datasets-v1` records.
Pipelines 1–3 keep their existing internal generation/validation checkpoint formats;
new accepted exports pass through the shared contract. Python callers can use
`data.adapters.legacy_record()` to recover the original validated record.

Existing CLI flags and default source/output locations remain available. All four
pipeline commands now refuse an output directory containing previous exported files
unless **`--overwrite`** is explicit. This also applies to interrupted-run resume:
rerun the original command with `--overwrite`. Checkpoints remain available; normal
fingerprint and checksum rules determine whether they can be reused. Python runners
have the equivalent `overwrite=True` argument. This option permits replacing exports;
it does not bypass validation or erase source documents.

## Mac: optional private publishing

```bash
uv run llmops pip1_data \
  --input ./dumy_policies --output ./artifacts/pipeline1 --max-tokens 2048 \
  --push-to-hub OWNER/enterprise-policies-sft

uv run llmops pip2_code \
  --input ./code_repos --output ./artifacts/pipeline2 --max-tokens 2048 \
  --push-to-hub OWNER/enterprise-code-sft
```

Provide `HF_TOKEN` through the environment or the existing root `.env`; do not put it
in YAML, source control, URLs or reports. LLM configuration is still needed only for
Pipelines 1 and 2. Publishing is disabled unless `--push-to-hub` is supplied.

Only successful, nonempty, checksum-matched accepted exports can be published.
The publisher revalidates each record and scans configured secrets/high-confidence
secret patterns before uploading. It stages exactly `training.jsonl`, `manifest.json`
and `README.md`; it never uploads the output directory recursively. Checkpoints,
rejections and raw source files are excluded. Accepted examples necessarily contain
source-derived policy/code text and provenance, so private publishing still requires
appropriate redistribution rights. Source license references remain in provenance;
no new license is inferred when the source license is unspecified.

Repositories are created private. Existing public targets are refused. The local
manifest records the resulting immutable commit SHA, repository, visibility and
training hash. Uploads use bounded retries; failures preserve the local dataset and
record a safe failure marker in an existing manifest. Publication failure returns a
CLI error even if local generation succeeded. No live publication was performed in
the current verification because no HF token or destination repository was configured.
The implementation uses the official [Hub upload API](https://huggingface.co/docs/huggingface_hub/guides/upload).

## CPU server: configurable Pipeline 4 sources

Edit `configs/datasets.yaml` with your repositories and run:

```bash
uv run llmops pip4_prepare --config configs/datasets.yaml
```

YAML relative paths resolve against the YAML file's directory, not the shell working
directory. The example uses `../artifacts/...` because the file is inside `configs/`.
Move the config and its relative input tree together or adjust those relative paths.
Paths in inherited provenance describe the original sources and need not exist on the
CPU server. Explicit CLI `--output`, `--max-tokens` and `--seed` override YAML values.
The existing JSON `Pipeline4Settings` configuration and old CLI remain supported:

```bash
uv run llmops pip4_prepare \
  --input ./artifacts --output ./artifacts/pipeline4 --max-tokens 4096
```

The old interface still requires healthy, checksum-matched exports from all three
upstream pipelines. YAML explicitly defines the required source set and its weights;
readiness then applies to that configured source set, not implicitly to all three.
Illustrative weights are not scientifically optimized. Mixing still uses the existing
without-replacement bottleneck allocation, which may exclude valid excess records.

Supported sources:

- `type: local`: a JSONL file, or a directory containing `training.jsonl`. An adjacent
  upstream report, when present, must show successful completion and a matching hash.
  Standalone files receive hashes and per-record validation, but do not prove that
  the original producer processed every intended input. Optional `sha256` pins bytes.
- `type: huggingface`: `repo_id`, optional `config`, `split` (default `train`), and
  `revision` (default `main`). Authentication uses the configured HF token. Resolution
  pins an immutable SHA; `_sources/` caches complete, checksum-verified snapshots.
  Explicit commit SHAs are recommended when transferring a run to another machine.
- Sources can be mixed and named independently, with unique names and finite positive
  total weight. Duplicate and related-source grouping runs across all named sources.

Published datasets from this project are downloaded with their manifest and verified.
Other prepared Hub datasets use [Datasets streaming](https://huggingface.co/docs/datasets/stream)
into a local snapshot before validation/mixing. `adapter: auto` requires unified or
legacy pipeline records. Explicit `adapter: nemotron` or `adapter: opencode` accepts
native rows from the matching NVIDIA dataset and reuses Pipeline 3 normalization:

```yaml
sources:
  - name: native_math
    type: huggingface
    repo_id: nvidia/Nemotron-Cascade-2-SFT-Data
    config: math
    adapter: nemotron
    revision: main
    max_records: 100
    weight: 1.0
output:
  directory: ../artifacts/pipeline4_math
  max_tokens: 4096
```

`max_records` is an optional cap on decoded remote records, not accepted records;
filters can reduce the exported count. Without it the selected remote split is fully
materialized. Streaming bounds processing memory, not total disk use or library
read-ahead. General remote inputs must be data files supported by Datasets; no custom
remote Python execution is introduced. JSON agent-trace conversion is disabled for
JSON-only repositories to preserve conversation structure.

Incomplete streams are never committed as cache hits; bounded retries restart their
snapshot at the pinned revision. Completed snapshots and Pipeline 4's SQLite stages
are reused. There is no byte-offset resume inside an incomplete remote snapshot.
Source hashes are checked again before final export. Source revisions, configurations,
splits, license metadata, row/file hashes, and acquisition provenance are recorded.
Native normalization failures become rejected-record audit reasons.

## Quality and readiness

Qwen3-14B's pinned original tokenizer/template, exact token limits, secret/identifier
checks, mandatory policy evidence checks, deduplication, source-component grouping,
official holdouts, seeded splitting and leakage audits remain in force. No silent
truncation or test-result invention is introduced. Metadata and verification remain
outside model-facing content. The final outputs remain `train.jsonl`,
`validation.jsonl`, `test.jsonl`, `rejected.jsonl`, `report.json` and `manifest.json`.
A `training_ready=true` result is automated preparation evidence for the selected
sources; it is not semantic certification, proof of test authenticity, or a training run.
No mandatory Docker dependency was added. Existing optional code-execution checks
still need their supported runtime; unsupported coding tasks remain quarantined.
The GPU-server trainer (Pipeline 5) is not implemented by this refactoring.

## Publish an already-generated accepted subset

Generation is not needed merely to upload existing records:

```bash
uv run llmops publish_dataset \
  --input ./artifacts/pipeline2 \
  --repo-id OWNER/enterprise-code-sft \
  --allow-incomplete-run
```

Omit `--allow-incomplete-run` for a healthy completed producer. The explicit flag
allows only the existing accepted export from a failed/incomplete run; it does not
promote rejected or quarantined examples. Every accepted record, evidence, secret
scan, export hash and count still passes the same publication checks. No generation,
LLM call or source retry occurs. The original report remains unchanged. The Hub
manifest/card and local publication metadata disclose the incomplete producer and
accepted-subset status. Configured Pipeline 4 Hub loading retains that disclosure and
reports a warning; this never establishes completion of the original Pipeline 2 run.
