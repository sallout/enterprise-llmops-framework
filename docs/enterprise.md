# Enterprise data workflow

`enterprise_data` creates one accepted conversational dataset from corporate policy
files, source repositories, or both. It composes the existing document and repository
components through LangGraph, rather than replacing their tested parsers, LLM clients,
evidence checks, sandbox restrictions or checkpoints. `pip1_data` and `pip2_code`
remain compatibility commands. The external-data command is still `pip3_datasets`,
and final preparation is still `pip4_prepare`; no training is implemented.

## Inputs

```bash
# Both source folders: generates fresh document and code examples.
uv run llmops enterprise_data \
  --documents ./dumy_policies \
  --code ./code_repos \
  --output ./artifacts/enterprise \
  --max-tokens 2048

# Documents only (omit --code), or code only (omit --documents).
uv run llmops enterprise_data --documents ./dumy_policies --output ./artifacts/enterprise_policies
uv run llmops enterprise_data --code ./code_repos --output ./artifacts/enterprise_code
```

At least one input is required; an omitted branch is not executed. `--max-tokens`
limits source chunks/code units, not model response length. Optional `--limit-documents`
and `--limit-units` constrain raw-source generation for testing; omit them for full
selected-source processing. They do not limit imported accepted datasets.

## Preserve and reuse existing coding records

**Existing `artifacts/pipeline2/training.jsonl` (and any `training.json`) are not deleted,
moved or rewritten by this implementation.** Do not run code generation again merely
to merge an existing coding export. Use the read-only import path:

```bash
uv run llmops enterprise_data \
  --documents ./dumy_policies \
  --code-dataset ./artifacts/pipeline2 \
  --allow-incomplete-import \
  --output ./artifacts/enterprise \
  --max-tokens 2048
```

`--allow-incomplete-import` is only needed when the existing producer report is failed
or incomplete. The currently observed local coding export contains 112 accepted
knowledge examples, zero runtime-verified examples, and a failed producer report with
four failed units. This explicit flag reuses only accepted records, retains that
producer status and reports `source_run_complete=false`; it does not promote rejected
or quarantined examples or claim that those failures were resolved.

To test document generation while reusing coding data, add `--limit-documents 1` and
use a separate output folder such as `artifacts/enterprise_test`. This still imports
the entire selected accepted coding export without any code-generation calls.

`--documents-dataset` provides the equivalent document reuse path. Two previous
accepted datasets can be combined without any LLM calls:

```bash
uv run llmops enterprise_data \
  --documents-dataset ./artifacts/pipeline1 \
  --code-dataset ./artifacts/pipeline2 \
  --allow-incomplete-import \
  --output ./artifacts/enterprise_merged
```

An import can be a JSONL file or output directory, with an adjacent producer
`report.json`. Its accepted count and checksum must match, and every row must pass
the shared schema, original policy/code provenance and secret checks. Imported rows
must belong to the specified branch. A source folder and a dataset import are mutually
exclusive for the same branch, avoiding accidental duplicated generation.

## Outputs and recovery

```text
artifacts/enterprise/
├── training.jsonl          # one combined, validated accepted export
├── rejected.jsonl          # merge audit + fresh-branch rejection audits
├── report.json             # branch statuses, counts, warnings, input hashes
├── manifest.json           # schema, source completeness, outputs, publication
├── _branches/
│   ├── documents/          # only when generating documents
│   └── code/               # only when generating code
└── _merge.sqlite3          # disk-backed merge staging; not a training input
```

Original IDs, domains, messages and verification remain unchanged. Source-specific
metadata and document/page/section or repository/symbol references remain in the shared
schema. Extra enterprise import metadata records branch mode, file hash and producer
completion. Exact message/tool duplicates are collapsed deterministically; duplicate
provenance is retained. Conflicting IDs with different content are rejected. Enterprise near-format deduplication and leakage-safe splits now run automatically
after the merge. Cross-source leakage checks remain required when external data is
loaded; final source sampling weights belong to the future trainer.

The report separates merged accepted records, invalid imported records, duplicate
occurrences, branch rejection/quarantine counts and producer completion. For imports,
historical rejection details remain in the original artifacts; they are not regenerated
or copied into the accepted dataset. No missing folder or failed requested branch is
silently treated as successful. An independent branch can still finish and retain its
accepted records while the overall run returns a failed status.

Each generated branch retains independent existing checkpoints. Resume with the same
inputs/options and explicit `--overwrite`; compatible completed LLM work is reused.
That option permits replacing only enterprise exports; existing non-enterprise output
directories and overlapping source paths remain protected. A merge is rebuilt from
validated exports into fresh SQLite staging, preventing duplicate appended rows.
New code revisions can invalidate old branch fingerprints; importing accepted data is
the reliable route when avoiding regeneration is the priority.

## Optional one-dataset publishing

Add `--push-to-hub OWNER/enterprise-sft` to publish one combined private dataset using
`HF_TOKEN`. If source runs are incomplete, publishing additionally requires explicit
`--allow-incomplete-run`; this preserves the existing accepted-subset disclosure.
Alternatively publish the existing combined export without regeneration:

```bash
uv run llmops publish_dataset \
  --input ./artifacts/enterprise \
  --repo-id OWNER/enterprise-sft \
  --allow-incomplete-run
```

Publishing remains disabled by default and uses the existing allowlisted private
publisher. No API token is placed in records or reports, and no raw source tree,
checkpoint or rejection artifact is uploaded. Accepted source-derived text still
requires appropriate redistribution rights. Inherited runtime evidence is not rerun
or authenticated by this wrapper. Imported incomplete producer status is disclosed,
not reclassified as a successful complete source run.

## Embedded enterprise preparation and direct external loading

The command now prepares enterprise data automatically; no separate `pip4_prepare`
command is needed. `--training-max-tokens` sets the exact original-Qwen-template bound
(default 4096), independently of source chunk/code-unit `--max-tokens`. The finalizer
reuses the tested internal preparation engine with a single enterprise source, so
there is no enterprise-source mixing quota. It checks sensitive content, exact token
counts, duplicate conversations, conservative near-format duplicates, source grouping,
seeded splits and leakage. Related policy clauses/repository files stay in one split.
Oversized examples are rejected without truncation. Fewer than three independent
source groups can legitimately fail the default three-way split; `--skip-preparation`
is an explicit generation-only option for tiny fixtures, not training-ready output.

Outputs now include `train.jsonl`, `validation.jsonl`, `test.jsonl`,
`preparation_rejected.jsonl`, and detailed preparation diagnostics. `training.jsonl`
is a union of accepted records from all splits for audit/backward compatibility;
**it must not be used wholesale as training input**. Immutable original accepted
snapshots live in `_accepted/`, and the preparation engine/checkpoints in
`_preparation/`. `report.json` distinguishes original generation acceptance from
final prepared acceptance and retains incomplete producer warnings. `training_ready`
means enterprise-only automated preparation, not full-source/semantic certification
or completion of a cross-dataset audit.

For existing accepted enterprise data, use a new output directory and no LLM calls:

```bash
uv run llmops enterprise_prepare \
  --input ./artifacts/enterprise_combined \
  --output ./artifacts/enterprise_ready --max-tokens 4096
```

Hub publication includes declared train/validation/test split files and a dataset card
which loads those splits rather than treating the all-split audit union as train.
Original accepted imports and their incomplete-run disclosures remain unchanged.

NVIDIA data can remain native on the Hub. `data/training_inputs.py` reuses inspected
pinned metadata, license checks, native schema adapters, configurable reasoning rules,
secret checks and exact Qwen token validation. It yields common conversational records
without writing a normalized NVIDIA JSONL dataset. Original code tests and supplied
execution statuses remain metadata, never newly locally verified evidence. The
optional earlier `pip3_datasets` export remains supported for caching/old workflows.

A mandatory shared `TrainingInputGate` indexes hashes/shingles and source components
in SQLite. All selected enterprise and external holdouts must be registered before
`seal_holdouts()` permits training records. Cross-split exact/near/component conflicts
are blocked, and same-split exact/format duplicates are excluded. Registering a late
holdout is refused. A native record explicitly marked as a holdout aborts a training
stream rather than being silently treated as train. Current NVIDIA adapters select
published train splits; unidentified benchmark overlap still needs research review.
Native rows are not truncated. Data or network errors fail the stream; there is no
silent skip of unavailable source blocks or training-cursor resume.

```bash
uv run llmops check_training_sources \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/training_input_check \
  --max-external-records 10 --max-tokens 4096
```

This bounded preflight reads a deterministic prefix of each pinned NVIDIA source;
it writes an audit report and fingerprint index, not training data. Counts refer to
decoded rows and may not equal transferred bytes because of library read-ahead.
Use a new audit output for each run. Prefix checks are not random sampling or a full
corpus leakage certificate. Full selected-source preflight and all selected holdout
registration are required before a future optimizer starts. The data loader is not
a trainer; model loading, source mixing weights and optimization are not implemented.
No Pipeline 5 or fine-tuning run was introduced. Index disk use scales with the number
of compared tokens; production sources need appropriate storage planning.

The former Pipeline 4 implementation is retained internally and as a hidden legacy
CLI, preserving existing scripts/tests. Existing user-customized YAML files remain
untouched and are optional compatibility workflows, not required steps.
