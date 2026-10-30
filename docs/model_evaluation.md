# Model evaluation

The `Model evaluation` menu prepares a frozen test selection and compares Qwen3
checkpoints sequentially on Linux/NVIDIA CUDA. It never starts training or changes
model weights. The default compression comparison uses original Qwen3-14B BF16
at the compression run's original immutable revision and the saved final NF4 model.
Optionally include the pruned BF16 and recovered/merged BF16 stages to distinguish
pruning, recovery training and final quantization effects. Explicit `--model`
options support old/new training runs; the first label is the reference model.

## Test groups and limits

- HumanEval: official `openai/openai_humaneval`, configuration `openai_humaneval`,
  test split (164 tasks), MIT license.
- MBPP: official `google-research-datasets/mbpp`, configuration `sanitized`, test
  split (257 tasks), CC BY 4.0. Retain attribution when sharing exported tasks.
- Enterprise: only hash-verified `test.jsonl` from a prepared enterprise directory.
  Train/validation/test hashes and disjoint test IDs are checked before selection.
  The final textual assistant turn is withheld as a reference; preceding turns
  retain the original prompt. No answer or policy context is invented.

These are **base HumanEval/MBPP tests**, not the expanded EvalPlus + suites. The
initial version implements the selected base suites directly. HumanEval requests a
complete implementation, not a suffix completion. MBPP includes its provided
assertions as visible examples so function names and calling conventions are
specified; those same base tests are the executable criterion, not independently
generated hidden tests. Reference implementations must pass the selected tests in
the execution image before any model receives a score.

`--max-tasks N` limits questions **per coding benchmark**, not their combined sum.
`--enterprise-limit N` limits held-out enterprise records. Zero means all available
records in that group. With both coding groups, N=20 and enterprise-limit=10 select
up to 50 tasks per model. A seeded shuffle chooses tasks from the small official
test splits; dataset Hub revisions are resolved and recorded. Hugging Face may
cache additional tiny benchmark splits while loading their configuration; no
NVIDIA corpus or model weights are downloaded during suite preparation.

A supplied `--suite` reuses its exact questions; selection-size/selection-seed
options do not alter it. To expand a pilot, prepare another suite and use a new
output directory. Never combine different selections as if they were one matched
comparison. If benchmark results guide further training, that selection is a
**development set**. Reserve a separate final test set and avoid repeatedly tuning
against it. These benchmark selections are made after training; describe their
chronology accurately. Python results do not establish Kotlin/TypeScript gains.

## Commands

No GPU is needed to prepare the frozen suite:

```bash
uv run llmops evaluate_model \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/evaluation_pilot \
  --max-tasks 5 \
  --enterprise-limit 5 \
  --prepare-only
```

On a Linux GPU server with a Docker daemon and a local Python image:

```bash
uv sync --locked --extra training
# On the Docker evaluation host, prepare this small CPU execution image once:
docker pull python:3.12-slim
uv run llmops evaluate_model \
  --compression-run /workspace/artifacts/compression_opencode_10k \
  --suite ./artifacts/evaluation_pilot/suite.json \
  --output /workspace/artifacts/evaluation_10k \
  --include-intermediates \
  --max-new-tokens 1024
```

Without `--include-intermediates`, only original BF16 and final NF4 are compared.
Original vs final measures the combined workflow; the intermediate comparisons
help separate its components. No result is assumed to be an improvement.

RunPod containers often have no Docker daemon. The menu offers **Generate answers
only**, and the following two-host flow avoids untrusted host-code execution:

```bash
# GPU host: no Docker needed, but no executable correctness score yet.
uv run llmops evaluate_model \
  --compression-run /workspace/artifacts/compression_opencode_10k \
  --enterprise ./artifacts/enterprise_ready \
  --max-tasks 5 --enterprise-limit 5 \
  --output /workspace/artifacts/evaluation_answers \
  --executor disabled

# Copy the COMPLETE evaluation_answers directory to a Docker-capable host.
# That host needs the project dependencies, not the model weights or GPU.
uv run llmops score_evaluation \
  --input ./artifacts/evaluation_answers \
  --output ./artifacts/evaluation_scored
```

`--executor disabled` explicitly leaves code scores unverified, with
`evaluation_complete=false` and `pass_at_1=null`. Offline scoring verifies exported
suite/answer checksums, checks reference tests, executes saved answers, and publishes
new scored results. It retains the original GPU timing evidence. There is no unsafe
host-execution fallback. Docker Desktop can supply the CPU execution host on macOS;
model inference remains Linux/CUDA only. The image must provide imports required by
the reference solutions; missing dependencies fail reference preflight before GPU
work. Set `--python-image` to a prepared image if additional packages are needed.

Compare a newly trained model using the same development suite:

```bash
uv run llmops evaluate_model \
  --suite ./artifacts/evaluation_pilot/suite.json \
  --model original=Qwen/Qwen3-14B \
  --model final_10k=/workspace/artifacts/compression_opencode_10k/model_4bit \
  --model final_larger=/workspace/artifacts/compression_larger/model_4bit \
  --training-tokens /workspace/artifacts/compression_larger/_runs/FINGERPRINT/inputs/tokens.jsonl \
  --output /workspace/artifacts/evaluation_larger
```

The explicit training cache screens only the supplied run, not all compared runs.
For a causal source-data claim, use additional matched training ablations; simple
before/after comparisons cannot attribute a gain to NVIDIA vs enterprise data.
Resume an interrupted evaluation with its identical arguments and `--resume`.
A change to models, suite, code, runtime, image or settings requires a new output.
This reuses checksummed task checkpoints; it does not append duplicate results.
Reference failures or infrastructure failures return a nonzero CLI status. A model
answer failing a test is a valid evaluation outcome, not an infrastructure failure.

## Metrics, provenance and limitations

One greedy answer is generated per task with thinking disabled, fixed response and
input budgets, and no silent truncation. Incomplete output/context-limit failures
remain visible and count as unsuccessful tasks. Base-test pass@1 is the fraction
passing all selected tests; marginal Wilson 95% intervals show sample uncertainty.
Matched differences are descriptive, not significance or causal claims. All models
use the same selected tasks after overlap exclusion. Batch size is one; inference
uses BF16 or the checkpoint's saved NF4 configuration. No additional quantization is
silently applied to the original baseline. A fixed seed does not guarantee bitwise
reproducibility across all CUDA environments.

Enterprise normalized exact match and word-token F1 are **lexical proxies only**.
A high score can miss a changed MUST/NOT, exception, approval condition or threshold.
`enterprise_review.jsonl` leaves human rubric fields blank for correctness,
conditions/exceptions and unsupported claims. No automated semantic pass is issued.
The existing held-out prompt may ask about facts absent from training; evaluate
those with appropriate supplied policy context before claiming learned facts.
The current enterprise validation split contains no policy records, and its small
policy test set limits generalisation. Further enterprise evaluation design remains
required; software workflow completion is separate from semantic approval.

With `--compression-run`, the evaluator finds the exact committed training-token
cache. A streaming overlap audit decodes each row, screens prompt/reference
13-word ngram containment (80% threshold), and excludes suspected matches for every
model. In explicit model mode, use `--training-tokens` or retain the
`training_overlap_not_checked` warning. This limited heuristic misses short text,
paraphrases and unknown pretraining overlap; no-match is not proof of clean public
benchmark independence. When training grows, check exclusions again and compare
only identical eligible selections.

GPU metrics include synchronized generation wall time, output tokens/second
**including prefill**, and peak allocated/reserved GPU memory. One short unscored
warmup runs per model; loading and warmup are excluded. TTFT, energy, cost, security
and adversarial robustness are not measured. Reused checkpoints retain their
original measured timing, so a resumed evaluation is not one continuous timing run.
Code executes without network or GPU, as a non-root user in a read-only container
with CPU/memory/process/time limits and only the individual task mounted. This is
an isolation measure, not a proof against arbitrary adversarial programs or cheating
against a visible evaluator. Keep the Docker daemon and host appropriately isolated.

Output files:

```text
evaluation/
  suite.json                    # frozen questions, references, source pins and tests
  manifest.json                 # model checksums/revisions, config, runtime, image
  report.json                   # metrics, differences, errors, warnings, output hashes
  results.jsonl                 # every model answer and test outcome
  enterprise_review.jsonl       # reference/answer pairs and unfilled human rubric
  overlap.json                  # checksummed limited contamination audit, if requested
  _checks/                      # reference-test checkpoints
  _checkpoints/                 # per-model/task resumable decisions
```

Do not use these evaluation questions or reference answers as future training data.
Freeze suite/manifests with the research record; they are separate from training
exports. Offline scoring exports suite/results/report and points to the original
enterprise review sheet; it does not copy model weights.

## Verification

Ordinary local tests use synthetic cases and mocked generation/execution. Real
benchmark preparation is a separate small network test. GPU and real Docker checks
are explicit opt-in tests and fail if requested infrastructure is absent:

```bash
uv run pytest -q
RUN_MODEL_EVALUATION_CUDA_TESTS=1 uv run --locked --extra training pytest tests/test_model_evaluation_cuda.py -q
RUN_MODEL_EVALUATION_DOCKER_TESTS=1 uv run pytest tests/test_model_evaluation_docker.py -q
```

Sources: [HumanEval dataset card](https://huggingface.co/datasets/openai/openai_humaneval),
[MBPP dataset card](https://huggingface.co/datasets/google-research-datasets/mbpp),
[Qwen3-14B model card](https://huggingface.co/Qwen/Qwen3-14B).

## Recover a completed generation export after a report-writing failure

If all answers were saved but the report remained RUNNING after NF4 export
metadata failed to serialize, update the project and run:

```bash
uv run --locked llmops finalize_evaluation \
  --input /workspace/artifacts/evaluation_50
```

This is CPU-only export recovery, with no model loading, model downloads, answer
regeneration or code-test execution. It verifies the original manifest fingerprint,
frozen suite, checksummed overlap decisions and every original per-model/task cache
key. It refuses incomplete/modified evidence rather than silently regenerating it.
It can therefore recover a prior implementation's completed answers after this
export-only code correction, while ordinary inference `--resume` continues to refuse
changed implementation fingerprints. Previous exports are retained under
`_recovery/`. Missing uncommitted model-load metadata remains explicitly unavailable;
timing and memory measurements already recorded per answer are retained.

An execution-disabled run still has `evaluation_complete=false` and no code pass@1.
After recovery, use `score_evaluation` on a Docker host to execute the saved answers.
Recovering a report is not evidence of model improvement or semantic correctness.
