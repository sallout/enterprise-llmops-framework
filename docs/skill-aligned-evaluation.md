# Profile training skills and prepare an independent evaluation

The goal is to measure the capabilities targeted by training, while also checking retention of general Python skills. It is not to measure memorization of training questions or choose tasks after observing model successes. No training, API-based generation or model inference is performed during preparation.

## CLI and menu

In **Model evaluation**, select **Profile training skills and prepare aligned independent tests (CPU + Docker)**. Equivalent command:

```bash
uv run --locked llmops prepare_skill_evaluation \
  --native-plan ./artifacts/opencode_77k \
  --enterprise ./artifacts/enterprise_ready \
  --output ./artifacts/opencode_skill_evaluation \
  --sample-size 500 --max-tasks 200 --final-tasks 100 --seed 42 \
  --exclude-suite ./evaluation_suites/suite.json
```

`--max-tasks` here is the total across benchmarks and both new suites. `--final-tasks` sets the exact final subset size and must leave at least one development task. If omitted, the final subset defaults to 20%. The current requested experiment uses 200 total and 100 final tasks. Previously used suites may be repeated through `--exclude-suite`; if none is passed and the original project suite exists, it is excluded automatically. Include every other suite already used for model selection when applicable. The output must be new. Runtime failures produce a FAILED report and no approved suites. This workflow currently has no resume: a failed stream must be rerun in a new directory. It replays only the pinned selected source prefix/buffer and stops at the last committed decision; it does not download the full five-million-row dataset by default. Reader prefetch may decode extra rows. Docker and a local `python:3.12-slim` image are required for reference validation; no host-code execution fallback is permitted.

## Method and boundaries

1. Verify prepared-plan/database checksums, decision digests and enterprise split hashes.
2. Select the lowest 500 SHA-256 ranks of `(seed, accepted record id)`. This is a deterministic uniform-hash sample of the approved selection, not of the complete upstream dataset. Record all accepted conversation token counts from committed metadata separately from sample statistics.
3. Assign overlapping skill tags by documented prompt regexes; infer language from code fences, explicit prompt language or Python AST evidence. Preserve original sample prompts/answers and source IDs for review. Unknown labels are retained. A task may match several skills; counts are not disjoint classes. Heuristics are not semantic task annotation. External test-status fields are retained without claiming local verification of training examples.
4. Load pinned official HumanEval and sanitized MBPP test candidates, exclude previously used task IDs, and screen them against all accepted native rows and enterprise training messages. Native raw hashes/ordinals are verified against every committed decision, including rejected rows encountered before the stopping boundary. This does not repeat Qwen tokenization or replace `nvidia_validate`.
5. Exclude >=80% containment of 13-word prompt/reference/test ngrams and normalized Python function AST matches (ignore docstrings and entry-point name; preserve behavior). Short snippets, paraphrases, renamed implementations and pretraining exposure remain unresolved. Exact/function-identical references within the selected evaluation pool are refused before splitting; semantic duplicates are not guaranteed absent.
6. Target 80% skill-aligned selection using round-robin coverage of skills appearing in at least 2%/three sampled prompts, plus 20% seeded general Python retention selection. This is coverage-oriented, not exact proportional resampling. A tag can be underrepresented or absent if the benchmark pools do not supply suitable tasks. Actual role/category counts are reported.
7. Run selected official reference solutions against their base tests in the existing restricted Docker executor. Also replace function bodies with `return None` and require the tests to reject that mutation. Do not report external metadata as these local results. Failing references/mutations are excluded and actual counts/shortfalls reported.
8. Seed-shuffle verified tasks into disjoint development and final suites (20% final by default, or the exact `--final-tasks` count). Freeze digests and source metadata before model results. Keep final tests unused during training/quantization decisions. A small final subset has wide statistical uncertainty.

The outputs are `profile_sample.jsonl`, `profile.json`, `overlap.json`, `development_suite.json`, `final_suite.json`, and `report.json`. Suite files use the existing checksummed evaluation schema and work with `evaluate_model`/`score_evaluation`. They contain reference answers for scoring; those answers are not inserted into model-facing prompts. MBPP retains its standard visible assertion examples, so its base tests are not fully hidden. This workflow prepares/selects official tasks; it does not claim newly authored independent benchmark questions.

The current real artifacts and frozen suite copies are documented in `report.md` and `evaluation_suites/opencode_77k_100_final/README.md`. When the approved training selection changes, repeat the audit/selection and treat any already inspected final set as development. These coding suites do not replace a separate held-out enterprise knowledge evaluation.

The expanded approved selection contains 100 development and 100 final tasks in `evaluation_suites/opencode_77k_100_final/`. The original 80/20 version is historical and was superseded before model inference. The interactive form defaults to 200 total and 100 final tasks; explicit CLI limits remain configurable.
