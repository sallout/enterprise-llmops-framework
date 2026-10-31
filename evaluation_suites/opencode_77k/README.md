# Frozen coding evaluation for the 77,891-row OpenCode selection

Prepared on 31 October 2026 before running either model on these tasks.

- `development_suite.json`: 80 tasks for development comparisons.
- `final_suite.json`: 20 separate tasks; keep unused until training/quantization choices are fixed. This is a small final sample, not a definitive broad capability certification.
- `training_skill_profile.json`: overlapping heuristic counts from a reproducible 500-record sample of the 77,891 approved rows. This is not semantic annotation of all training records.

Across both suites: 41 HumanEval and 59 sanitized MBPP Python tasks; 80 selected through skill coverage and 20 through general Python retention selection. All 100 reference solutions passed the supplied base tests in Docker and their return-None mutations failed. These are test/reference checks, not model scores or proof of exhaustive test coverage. The prior suite's task IDs were excluded; three additional candidate tasks were excluded after AST matches to selected training answers. The 77,891 approved native records and the enterprise training split were screened, with exact source-identity replay checks for the native selection. No task IDs overlap between these suites; no exact/reference function-AST duplicate was found across them. Semantic clones and model-pretraining contamination remain unresolved.

The official benchmark pools provide uneven coverage. This set principally measures Python implementation problems involving sequences, arithmetic and strings. It does not establish coverage of all observed OOP, file handling, exception validation, recursion, debugging, refactoring or non-Python tasks. These remain coverage limitations, not automatically measured failures.

Run development comparison on Linux/NVIDIA using the existing evaluator and the actual compression directory:

```bash
uv run --locked --extra training llmops evaluate_model \
  --compression-run /workspace/artifacts/YOUR_COMPLETED_COMPRESSION_RUN \
  --suite ./evaluation_suites/opencode_77k/development_suite.json \
  --output /workspace/artifacts/evaluation_77k_development \
  --max-new-tokens 1024 --executor disabled
```

`--executor disabled` generates answers only. Score those answers on a host with Docker using `score_evaluation`; no GPU is needed for saved-answer scoring. Use a new evaluation directory for each experiment. Retain the evaluator's actual training-token overlap audit: the current preparation screen examines pinned raw records, whereas the inference-time audit examines consumed training inputs. The final suite is used separately after experiment choices are frozen. The enterprise assessment remains a separate existing held-out evaluation; these new files contain coding tasks only.

## Attribution and licenses

HumanEval: OpenAI, *Evaluating Large Language Models Trained on Code* (Chen et al., 2021), MIT license. Dataset: https://huggingface.co/datasets/openai/openai_humaneval ; revision `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544`. Original source and license: https://github.com/openai/human-eval .

MBPP: Google Research, *Program Synthesis with Large Language Models* (Austin et al., 2021), dataset CC BY 4.0 as declared by the source. Dataset: https://huggingface.co/datasets/google-research-datasets/mbpp ; sanitized test split, revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`. License: https://creativecommons.org/licenses/by/4.0/ . Tasks were selected and wrapped in the project's conversational format, with executable test wrappers and metadata added. This is not Google/OpenAI endorsement.

Each case retains source dataset, split, configuration, record ID, revision and license. These are the original base tests, not EvalPlus extended tests. No LLM generated the selected questions or reference answers. The audit, source sample and actual preparation report remain under `artifacts/opencode_skill_evaluation/`; the sampled NVIDIA training text is not copied into this versioned suite directory.
