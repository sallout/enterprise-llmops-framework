# Linux/CUDA model compression and enterprise specialization

This experimental workflow implements the agreed order:

1. Limited structured **MLP channel pruning** of the original dense `Qwen/Qwen3-14B`.
2. **BF16 LoRA supervised recovery/specialization**, then merging the adapter.
3. **NF4 4-bit export**, reload and bounded technical forward/generation checks.
4. Full baseline-versus-final quality, efficiency and security evaluation comes later.

It is **not implemented/tested for macOS, MPS, CPU training or distributed GPUs**.
The existing document/data CLI still works without training dependencies. No claim
of quality retention, inference speedup or completed Qwen3-14B training is implied by
unit tests. Structural pruning is experimental and can substantially degrade quality.

## Server prerequisites

Use Linux, Python 3.12, an NVIDIA CUDA driver compatible with the locked PyTorch
CUDA runtime, and a BF16-capable GPU. The lock currently selects PyTorch **2.9.1**
with CUDA **12.8** wheels, PEFT **0.21.2**, Accelerate **1.15.0** and bitsandbytes
**0.50.2**. It preserves existing native-data dependency versions, including
Transformers **5.20.2**. Validate the actual driver using `nvidia-smi` and preflight.

Pruning and recovery use **BF16 weights**, not a 4-bit training base. The default
capacity guard requires 36 GiB free VRAM and 120 GiB disk capacity (free plus already
owned compression artifacts). These are configurable **lower-bound guards, not a
measured peak requirement or guarantee**. An 80 GB GPU is a conservative starting
choice for the 14B experiment; tune batch/context after a real smoke run. The previous
`t3.small` has no CUDA GPU and cannot execute this workflow. Exactly one CUDA device must be visible (`cuda:0`); set `CUDA_VISIBLE_DEVICES=0`
on servers exposing multiple GPUs. Multi-process runs are refused.

```bash
uv sync --locked --extra training
nvidia-smi
uv run llmops
```

Choose **Model compression & fine-tuning**. Start with the environment-check action,
then choose the pruning/training/export action. Paths, MLP fraction, calibration
count, context bound, optimizer steps and incomplete-source consent are prompted.
The generic future customization/security/prompt-injection entries remain disabled;
enterprise specialization is already part of recovery in this workflow.

Equivalent preflight (no model weights):

```bash
uv run llmops compress_model \
  --enterprise ./artifacts/enterprise_ready \
  --native-plan ./artifacts/nvidia_local \
  --output ./artifacts/compression_smoke \
  --check-only
```

Preflight checks platform, dependencies, CUDA, BF16, free capacity and immutable
input provenance. It does not prove that data replay, pruning, training or export
works on the server. It does not freeze the later experiment configuration.

First bounded GPU experiment (real 14B weights WILL download):

```bash
uv run llmops compress_model \
  --enterprise ./artifacts/enterprise_ready \
  --native-plan ./artifacts/nvidia_local \
  --output ./artifacts/compression_smoke \
  --prune-ratio 0.05 \
  --calibration-samples 16 \
  --max-tokens 4096 \
  --max-steps 10 \
  --allow-incomplete-sources \
  --resume
```

Use `--allow-incomplete-sources` only when intentionally reusing the validated accepted
subset of an incomplete producer (the current enterprise publication has that warning).
It does not mark that producer complete. The above budgets are technical-smoke choices,
not claimed sufficient training or optimal hyperparameters. Twenty NVIDIA records
remain a smoke sample. Prepare a representative larger source plan for research runs.
`--max-steps` means optimizer updates, not examples. A configurable
`--config docs/compression.example.json` supplies the complete strict configuration;
`TRAINING__*` environment settings also configure defaults. CLI values override them.
Changing configuration requires a **new output directory**.

## Input and supervision guarantees

- Reuse existing prepared enterprise train/validation/test splits and the committed
  NVIDIA native source plan. No new teacher API calls or code/policy generation.
- Pin original model weights to the exact immutable enterprise Qwen tokenizer revision.
- Index **all enterprise holdouts and train fingerprints** in a fresh global gate before
  exact approved NVIDIA replay. Existing native source/code/dependency/hash guards apply.
- Only enterprise **train** and approved NVIDIA **training rows** are used for calibration
  and recovery. No held-out records are training/calibration input. NVIDIA holdouts beyond
  the selected plan are not introduced or certified.
- Create a bounded per-source reservoir, round-robin-selected for calibration. It is
  source-balanced sampling, not certified semantic/task balance. No arbitrary token
  slicing. Over-limit conversations fail instead of being silently truncated.
- Keep original Qwen chat rendering (`enable_thinking=False`) and preserve multi-turn/tool
  formatting. Fast-tokenizer offsets mask all system/user/tool-response spans; only
  assistant spans (including tool calls and message end tokens) receive loss. Rendering
  IDs must match `apply_chat_template(tokenize=True)` exactly.
- A disk-backed supervised **token cache**, calibration token list and provenance audit
  are internal training inputs, not a full NVIDIA dataset download or new Hub dataset.
  Protect these files like enterprise data; they can reconstruct source text.
- `enterprise_repeats` deliberately upweights enterprise rows without introducing new
  unique examples. NVIDIA allocation follows the source plan. Default repeats=1 does
  **not** guarantee policy/code balance; inspect actual counts and token budgets.

## Structural pruning implementation

The installed Transformers Qwen3 implementation has a gated SiLU MLP:
`down_proj(silu(gate_proj(x)) * up_proj(x))`.
The pipeline measures RMS post-gate channel activations on unpadded calibration
records and multiplies them by each down-projection column's L2 norm. It retains
highest-scored channels, with stable index tie-breaking and original order.

The **same channel indices** slice gate/up output rows and down input columns.
Every layer retains the same aligned intermediate width. Update serialized
`config.intermediate_size`, verify exact removed parameter count, save and reload
the pruned checkpoint. Attention heads, head dimensions, residual width, vocabulary,
layer count and RoPE remain unchanged. This is actual dense dimension reduction,
not zero masks. A 10% MLP fraction is **not** 10% overall parameter reduction;
alignment may reduce the actual fraction further. At most 20% MLP pruning is allowed
by the initial bounded configuration. This is a transparent project heuristic,
not a claim to implement or reproduce NVIDIA Minitron or an optimal pruning method.

Recovery uses PEFT LoRA on attention and MLP projections over the BF16 pruned base;
only adapters are trained. Training stops on non-finite logged losses/gradient norms.
Adapter merge verifies the structural base parameter count remains unchanged.
This initial version does **not** implement teacher distillation.

## Output, resumption and evidence

```text
artifacts/compression_smoke/
├── model_4bit/              # self-contained HF checkpoint after verified export
├── report.json             # actual status, counts, versions, sizes, smoke results
├── checkpoint.json         # immutable experiment/stage fingerprints and file hashes
├── _tokenizer/             # pinned small tokenizer assets
└── _runs/<fingerprint>/
    ├── inputs/             # supervised tokens + calibration + provenance + gate index
    ├── pruned/             # pruned BF16 checkpoint
    └── recovery/
        ├── checkpoints/    # Trainer optimizer/RNG/adapter state
        ├── adapter/        # learned LoRA adapter
        └── merged/         # merged BF16 pruned model
```

LangGraph holds only stage/path references; model/data tensors are not graph state.
File hashes commit each stage only after it passes. Resume requires the same input
hashes/config/training code/runtime versions. Completed stages are reused.
Interrupted pruning/export restarts that unfinished stage; interrupted recovery uses
the latest complete optimizer/RNG checkpoint, skipping incomplete writes. If no complete
optimizer checkpoint exists, recovery explicitly restarts from the committed pruned
model. Checkpoint frequency determines lost work. Native stream replay is cached only
after complete input validation; an interrupted input stage is revalidated afresh.
No original input dataset or `.env` is modified. Dataset changes require a new run.

`model_4bit/` appears only after save/reload, source stability and technical checks pass.
It is Transformers/bitsandbytes NF4 with double quantization and BF16 compute. Some
weights (embeddings/norms/output layers) remain higher precision; “4-bit” does not mean
every tensor occupies four bits. GGUF/AWQ/vLLM-specific compatibility is not asserted.
Check `final_model_ready`, `status` and `evaluation_pending` in report.json; existence
of weights alone is not an approved result. No automatic Hub publication/deployment.

## Real server tests before a costly experiment

Run the ordinary suite, then explicitly request the small genuine CUDA integration:

```bash
uv run pytest -q
RUN_COMPRESSION_CUDA_TESTS=1 uv run --extra training pytest tests/test_compression_cuda.py -q
```

The opt-in test initializes a tiny random dense Qwen3 locally, structurally prunes it,
performs two real LoRA optimizer updates, saves/merges, quantizes to NF4, reloads and
checks forward/generation. No remote model/dataset is downloaded. If explicitly requested
without a working CUDA environment it fails. Passing it would establish basic API/kernel
compatibility, **not** 14B capacity, recovery quality or final benchmark performance.
Then run the bounded 14B command above and inspect its actual report before scaling.

## Technical references

- [Transformers Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
- [PEFT LoRA and merge](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [NVIDIA pruning/distillation examples](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning): evidence for the broader method, not certification of this implementation or Qwen3-14B results.

## Standalone 8-bit export after training

Use the completed recovery/merged BF16 checkpoint to test whether LLM.int8
preserves more of its measured development-set performance than the NF4 export:

```bash
uv run --locked --extra training llmops export_8bit \
  --compression-run /workspace/artifacts/compression_opencode_10k \
  --output /workspace/artifacts/quantization_8bit
```

The menu exposes the same operation under **Model compression & fine-tuning →
Export existing trained BF16 to 8-bit (no training)**. No additional pruning or
training runs, and `model_4bit/` is not overwritten. Output is
`/workspace/artifacts/quantization_8bit/model_8bit`; `report.json` records source
identity, runtime, outlier threshold, saved-file hashes, actual export bytes and
save/reload smoke evidence. The format is bitsandbytes **LLM.int8**, not FP8 or NF4.
Its outlier handling and non-quantized layers make it mixed precision; nominal
8-bit does not mean every tensor/operation is int8.

The source must be an unquantized dense Qwen3 checkpoint from the committed recovery
stage. Quantized sources are refused. Input/recovery stage hashes are checked
before export and again afterwards. Execution requires Linux, one BF16-capable
NVIDIA CUDA GPU and the existing training extra. Capacity guards default to 24 GiB
free VRAM and 32 GiB disk, which are conservative lower-bound checks rather than
measured peaks or guarantees. The export is loaded directly with
`BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)` and uses local source
files, saves safely, releases memory, reloads the saved model and checks actual
8-bit modules plus finite forward/generation smoke behaviour. No LLM endpoint,
remote model download, calibration optimisation or semantic certification is used.
The smoke uses an existing training-only calibration row, not the frozen test suite.

Interrupted exports can be retried with the identical command plus `--resume`;
only this export's pending staging directory is rebuilt. Already verified exports
are reused after file-hash checks. Source weights, original NF4 and datasets are
preserved. Use a new directory if source/configuration/code changes. A separate
8-bit output is not automatically included in an older compression-root backup;
preserve it independently before terminating GPU storage.

After successful export, the existing evaluator accepts the local checkpoint:
`--model recovered_int8=/workspace/artifacts/quantization_8bit/model_8bit`.
Reuse the exact development suite, generation budget and training-cache overlap
screen before comparing results with the BF16/NF4 stages. This operation does not
predict or guarantee better coding scores.

Real tiny-model int8 save/reload is covered by the optional Linux/CUDA test:

```bash
RUN_MODEL_EVALUATION_CUDA_TESTS=1 uv run --locked --extra training \
  pytest tests/test_model_evaluation_cuda.py -k int8 -q
```

That test creates tiny synthetic weights locally and downloads no pretrained
model. Ordinary macOS tests use a fake backend and do not establish CUDA success.
Implementation follows the [Transformers bitsandbytes documentation](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes).
