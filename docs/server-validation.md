# Pipeline 2 — Validation before server use

## Locally verified checks

From the project root:

```sh
uv sync --locked
uv run pytest -q
uv build
uv run llmops pip2_code --help
```

The test suite includes synthetic source parsing, mock generation/review, validation failures, fixed-test correction, export/deduplication, CLI limits, safe diagnostics, interruption recovery, and checksummed resume. A mock execution result is not actual runtime evidence. The conditional Docker test is skipped when the daemon or configured local image is unavailable.

Source `--max-tokens` and `PIPELINE2__RESPONSE_MAX_TOKENS` are separate. Pipeline 2 now defaults to an 8192-token response budget after live 4096-token requests were truncated. A larger budget does not guarantee every model response will fit. Configuration is overridable without changing `.env`.

## Server acceptance gates

1. Install the project with the locked dependencies and pass the automated suite. Transfer source/data as required and provision credentials securely in the server environment.
2. Make Docker available and provision the trusted `python:3.12-slim` image locally (or set `PIPELINE2__PYTHON_IMAGE` to your prepared image). The application does not download images automatically.
3. Run `uv run pytest tests/test_pipeline2.py -q -rs`. The real Docker test must pass rather than skip. This executes only a small isolated synthetic fixture; it does not run repository code on the host.
4. Run a limited real Python workflow:

```sh
uv run llmops pip2_code \
  --input ./code_repos/full-stack-fastapi-template-master/backend \
  --output ./artifacts/pipeline2_server_validation \
  --max-tokens 2048 \
  --limit-units 7
```

5. Inspect `report.json`, accepted record metadata, and rejected records. Confirm no operational failures or pending units, positive actual runtime test counts for accepted coding examples, valid references/export hashes, and human-reviewed task/answer quality. `COMPLETED_WITH_WARNINGS` requires checking the actual reasons; an exit code alone does not prove dataset quality. Rejected candidates can legitimately remain rejected.
6. Repeat the exact successful command to verify checkpoint reuse, stable output hashes, and no duplicate committed examples.

## Remaining scope limits

Only Python has an isolated execution backend. Kotlin and TypeScript executable examples remain quarantined even with Docker running; those backends require implementation and real verification before the complete mixed-language executable workflow can be called ready. Knowledge tasks have explicit runtime-not-applicable/unverified metadata. Automated model review and generated test passes do not establish complete semantic correctness. No fine-tuning or Pipeline 3 is part of these checks.
