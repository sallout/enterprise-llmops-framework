# Interactive data-preparation workspace

Start from the project directory:

```bash
uv sync --locked
uv run llmops
```

`uv run llmops menu` opens the same workspace. Existing commands and `--help` remain
available for automation; no existing pipeline interface was removed.

The header uses the requested **Bridging the Cusem** title, **Mohammed Sallout** author,
and clickable **https://www.linkedin.com/in/msallout/** profile. Branding is configurable
through Env setup (`UI__TITLE`, `UI__AUTHOR`, `UI__LINKEDIN_URL`). The title spelling is
retained as supplied; it can be edited without code changes.

On a real terminal use arrow keys or the displayed numbers, then Enter. SSH terminals
are supported. Piped/non-TTY sessions use a numbered Rich prompt fallback. Ctrl+C
cancels a form/action; `:back` cancels form entry. At the main menu Ctrl+C, EOF or Exit
ends the session. Paths support quoted names, spaces, tilde expansion and common Mac
drag-and-drop escaping. Operations show a configuration summary and require an
explicit Run confirmation before processing starts.

## Menu and actual capabilities

| Option | Behavior |
|---|---|
| Enterprise data extraction | Existing enterprise generator/merge, with optional document and code branches |
| Enterprise data validation | Prepare accepted data/splits without regeneration; inspect a report; optionally publish accepted data privately |
| External dataset installation | Existing native NVIDIA preparation plan, not a full corpus/model-weight installation |
| External dataset validation | Read-only replay validation of an existing approved native selection |
| Fine-tuning | Disabled, not implemented |
| Security check | Disabled, not implemented |
| Prompt injection check | Disabled, not implemented |
| Model customization | Disabled, not implemented |
| Env setup | Read-only configuration presence, secure `.env` editing and pipeline defaults |

Unavailable options are visibly disabled in the arrow menu. The numeric fallback
also refuses to start them. Opening the menu never starts training, installs a GPU
runtime, calls an LLM endpoint, publishes data or modifies `.env` automatically.

## Enterprise extraction and validation

Choose a small local test, full selected sources or custom generation limits. Each
branch can read a raw folder, reuse accepted JSONL plus its adjacent producer report,
or be skipped. At least one branch is required. Document ingestion supports the
existing PDF, Markdown and TXT formats. Code import preserves the existing coding
training file; choose **Reuse accepted code dataset** rather than the raw code folder
when avoiding regeneration is the priority.

The wizard detects an accepted dataset from the wrong/mixed branch before running
it, preventing the previous mistake of passing coding data as document examples.
Incomplete producer data requires explicit accepted-subset consent. Existing output
folders require choosing another location or explicitly allowing resume/rewrite.
Backend overlap checks, fingerprints, schema checks and dataset protection remain
active; the UI cannot bypass them.

The suggested source chunk/code-unit budget is **4048 UTF-8 byte-token units**. It is
not the LLM response budget and not the final Qwen conversation length. Optional
local limits are 1 document / 2 repository units. A document can contain multiple
chunks, so this is not a hard cap on API requests. Source-folder generation requires
configured LLM endpoint/key/model; import-only merges do not.

Extraction defaults to a separate validation step (`--skip-preparation` internally),
matching the menu's separate extraction/validation actions. It can also include final
preparation when explicitly selected. The direct `enterprise_data` command retains
its existing automatic-preparation default. Exact Qwen limit (4096 default), seed,
source paths, output, task types and generation limits are configurable.

Validation prepares an existing accepted enterprise artifact into a new directory,
without rerunning document/code generation. Very small fixtures may lack three
independent groups for train/validation/test and legitimately fail; the UI does not
silently split related source groups or fabricate evaluation examples. Publishing is
an explicit sub-action with a real owner/dataset namespace and private Hub target;
failed/incomplete source publication requires separate consent. Existing source
status and accepted-subset disclosure remain visible.

## External preparation and validation

Installation offers local/server/custom accepted caps, dataset selection, candidate
limits, source configuration/revision, allocation, sampling seed, buffer and exact
Qwen token limit. It routes to `nvidia_data` and inherits metadata pinning, license
inspection, original external test metadata, native schema adapters, leakage checking
and transactional recovery. It produces source coordinates/fingerprints, not a full
normalized NVIDIA dataset. A matching existing native output resumes automatically;
changed settings create an isolated fingerprint.

Validation routes to the new `nvidia_validate` action. It uses a fresh leakage gate,
indexes all enterprise splits, and replays approved pinned source rows through the
existing integrity/schema/token checks. It preserves original source-plan and
enterprise bytes. Full validation checks every approved example, not the entire
upstream NVIDIA corpora. An optional smaller limit is explicitly reported as sample
validation: `full_plan_validated=false`, status with warnings. Bad/stale plans and
corrupt hashes return failure rather than a green readiness badge.

For automation the equivalent action remains available:

```bash
uv run llmops nvidia_validate \
  --plan ./artifacts/nvidia_local_verification_20261030 \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/nvidia_validation
```

No LLM calls, code-test execution or model training are performed by native validation.
Only a validation report and fresh fingerprint audit are written. Old audit outputs
require an explicit rewrite choice; new fingerprint runs remain isolated.

## Environment safety and tuning

Presence checks show configured/missing booleans only. They do not verify endpoint
reachability or token permissions. In an interactive terminal, keys/tokens and the
LLM endpoint are entered through hidden prompts. Secret entry is refused for piped
input to avoid accidental echo. Blank answers keep existing values.

Saving `.env` is an explicit operator action: staged values are validated, unrelated
settings/comments are preserved, and the original file is backed up to a unique
`.env.backup-*` file. New file and backup use owner-only mode 0600. Replacement is
atomic, symlinked env files are refused, and unsupported/multiline/interpolated edits
are rejected. Neither menu history nor reports store credential values. Failed file
validation does not replace the original. Process environment overrides still take
precedence and are reported without printing their values.

The tuning action controls document/code LLM response budgets (256–32768), endpoint
timeouts/retries, external timeout/retries/candidate-scan multiplier, and logging
level. Changing generation defaults can invalidate checkpoints; import old accepted
code data to avoid costly regeneration. No settings/network test or API call is
performed just by saving these values. No actual project `.env` was edited during
implementation verification; write/backup tests use temporary fixtures.

## Results, history and limitations

Backend progress is streamed live. Exit codes, warning/failure states, prepared flags,
full-versus-sample validation and numeric report summaries are preserved; private
provider responses and record contents are not displayed. A published accepted subset
is not incorrectly described as a successful full producer run. The main session can
return to the menu after a failed job; its recorded job exit code remains available.
Use the original direct commands when a script needs their nonzero process exit codes.

Local `.llmops/last_run.json` records operation/output/exit code/duration, and
`.llmops/outputs.json` remembers successful output locations as future defaults. Neither
stores secrets or full input answers. Both are Git-ignored. History write failure
cannot turn an already completed processing operation into a claimed processing
failure. Existing reports/datasets remain the authority for actual counts/readiness.

The UI is synchronous and terminal-based. It does not implement a background job
scheduler, training/optimizer, remote GPU rental, new security or prompt-injection
algorithms, model customization, or a complete semantic correctness certificate.

## Restore an already published enterprise dataset

On a new server, configure `HF_TOKEN` through **Env setup** (a read token with
access to the private dataset is sufficient). Start `uv run llmops`, choose
**Enterprise data validation → Download an existing enterprise dataset from
Hugging Face**, enter `sallout/enterprise-sft`, select a revision (default `main`),
and choose a local directory such as `artifacts/enterprise_ready`. Confirm the
preview. This does not run document extraction or code generation.

Equivalent command:

```bash
uv run llmops download_dataset \
  --repo-id sallout/enterprise-sft \
  --output ./artifacts/enterprise_ready
```

For the exact locally verified publication, add
`--revision 28c64f71a8917f1c09af94fbd94c738643062ca5`.
The branch/tag is resolved to an immutable commit before downloading. Only the
published accepted JSONL, split files, manifest and dataset card are fetched;
Hub cache supports retry/reuse. Existing exports require explicit `--overwrite`.
A failed integrity check does not commit new export files. The generated local
`report.json` records restoration evidence and preserves inherited incomplete-source
warnings. It does not claim new runtime or semantic verification.

Prepared exports can be selected as the enterprise input for NVIDIA preparation;
unprepared accepted-only exports still need the existing enterprise preparation
step. Original parsing artifacts, repository sources and interrupted-generation
checkpoints are not included in the published dataset and are not restored. The
native NVIDIA source plan is a separate artifact, not part of this enterprise download.
