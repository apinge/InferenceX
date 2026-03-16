# InferenceX Qwen3.5 Single node MI355X local benchmark (no GitHub Actions)

Run a single InferenceX Qwen3.5 benchmark on single node 8 * AMD MI355X without CI.

## Requirements

- **Hardware:** AMD MI355X GPU(s) — scripts use `TP=8` (8 GPUs) by default; override with `export TP=<n>` if needed.
- **Environment:** SGLang and the benchmark client. Use the same container image as CI:
  - **BF16:** `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260215`
  - **FP8:** `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260218`
- **Models:** Ensure Hugging Face CLI/token is configured;
  - **BF16:** `Qwen/Qwen3.5-397B-A17B`
  - **FP8:** `Qwen/Qwen3.5-397B-A17B-FP8`
- **Run from repo root:** Execute from the InferenceX repo root so paths to `benchmarks/` and `utils/` resolve.

## Usage

All scripts take optional `[CONC] [ISL] [OSL]`. Defaults: CONC=4, ISL=1024, OSL=1024.

| Argument | Default | Description |
|----------|---------|-------------|
| CONC | 4 | Max concurrency for the benchmark client. |
| ISL | 1024 | Input sequence length. |
| OSL | 1024 | Output sequence length. |

### BF16 model usage

**Script:** `run_qwen3.5_bf16_mi355x_local.sh`
**Benchmark:** `benchmarks/single_node/qwen3.5_bf16_mi355x.sh`
**Default model:** `Qwen/Qwen3.5-397B-A17B`

```bash
# From InferenceX repo root

# Default: CONC=4, 1k1k (1024 in, 1024 out)
./scripts/run_qwen3.5_bf16_mi355x_local.sh

# CONC=8, 1k1k
./scripts/run_qwen3.5_bf16_mi355x_local.sh 8

# CONC=16, 1k8k (1024 in, 8192 out)
./scripts/run_qwen3.5_bf16_mi355x_local.sh 16 1024 8192

# CONC=32, 8k1k (8192 in, 1024 out)
./scripts/run_qwen3.5_bf16_mi355x_local.sh 32 8192 1024
```

### FP8 model usage

**Script:** `run_qwen3.5_fp8_mi355x_local.sh`

**Benchmark:** `benchmarks/single_node/qwen3.5_fp8_mi355x.sh`

**Default model:** `Qwen/Qwen3.5-397B-A17B-FP8`

```bash
# From InferenceX repo root

# Default: CONC=4, 1k1k (1024 in, 1024 out)
./scripts/run_qwen3.5_fp8_mi355x_local.sh

# CONC=8, 1k1k
./scripts/run_qwen3.5_fp8_mi355x_local.sh 8

# CONC=16, 1k8k (1024 in, 8192 out)
./scripts/run_qwen3.5_fp8_mi355x_local.sh 16 1024 8192

# CONC=32, 8k1k (8192 in, 1024 out)
./scripts/run_qwen3.5_fp8_mi355x_local.sh 32 8192 1024
```

## Environment overrides

| Variable | Default (BF16) | Default (FP8) | Description |
|----------|----------------|---------------|-------------|
| `INFERENCEX_WORKSPACE` | `$(pwd)/bench_workspace` | same | Directory for server log, result JSON, and GPU metrics. |
| `MODEL` | `Qwen/Qwen3.5-397B-A17B` | `Qwen/Qwen3.5-397B-A17B-FP8` | HuggingFace model name/path. |
| `TP` | 8 | 8 | Tensor parallel size (number of GPUs). |
| `RANDOM_RANGE_RATIO` | 0.8 | 0.8 | Passed to benchmark client. |
| `RUN_EVAL` | false | false | Set to `"true"` to run lm-eval after throughput. |

## Outputs

- **Result JSON:** `$WORKSPACE/${RESULT_FILENAME}.json` (and any aggregated file produced by the client).
- **Server log:** `$WORKSPACE/server.log`.
- **GPU metrics:** `$WORKSPACE/gpu_metrics.csv` (if amd-smi is available).

## Process results

To process the result with InferenceX tooling (same format as CI):

```bash
# From repo root with workspace in place
python3 utils/process_result.py   # reads result from workspace
python3 utils/summarize.py       # if you use the full pipeline
```
