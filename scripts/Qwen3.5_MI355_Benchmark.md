# Local benchmark run (no GitHub Actions)

Run a single InferenceX benchmark on your machine without CI.

## Qwen3.5 BF16 on MI355X (SGLang)

**Script:** `run_qwen3.5_bf16_mi355x_local.sh`

### Requirements

- **Hardware:** AMD MI355X GPU(s) — script uses `TP=8` (8 GPUs) by default; override with `export TP=<n>` if needed.
- **Environment:** SGLang (and its dependencies) and the benchmark client. Typically you’d use the same container image as CI, e.g.:
  - `rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260215` (see `amd-master.yaml`).
- **Model:** `Qwen/Qwen3.5-397B-A17B`. The script runs `hf download "$MODEL"`; ensure Hugging Face CLI/token is configured if the model isn’t already cached.
- **Run from repo root:** Execute from the InferenceX repo root so paths to `benchmarks/` and `utils/` resolve.

### Usage

```bash
# From InferenceX repo root
./scripts/run_qwen3.5_bf16_mi355x_local.sh [CONC] [ISL] [OSL]
```

| Argument | Default | Description |
|----------|---------|-------------|
| CONC    | 4   | Max concurrency for the benchmark client. |
| ISL     | 1024 | Input sequence length. |
| OSL     | 1024 | Output sequence length. |

**Examples:**

```bash
# Default: CONC=4, 1k1k (1024 in, 1024 out)
./scripts/run_qwen3.5_bf16_mi355x_local.sh

# CONC=8, 1k1k
./scripts/run_qwen3.5_bf16_mi355x_local.sh 8

# CONC=16, 1k8k (1024 in, 8192 out)
./scripts/run_qwen3.5_bf16_mi355x_local.sh 16 1024 8192
```

### Environment overrides

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCEX_WORKSPACE` | `$(pwd)/bench_workspace` | Directory for server log, result JSON, and GPU metrics. |
| `MODEL` | `Qwen/Qwen3.5-397B-A17B` | HuggingFace model name/path. |
| `TP` | 8 | Tensor parallel size (number of GPUs). |
| `RANDOM_RANGE_RATIO` | 0.8 | Passed to benchmark client. |
| `RUN_EVAL` | false | Set to `"true"` to run lm-eval after throughput. |

### Outputs

- **Result JSON:** `$WORKSPACE/${RESULT_FILENAME}.json` (and any aggregated file produced by the client).
- **Server log:** `$WORKSPACE/server.log`.
- **GPU metrics:** `$WORKSPACE/gpu_metrics.csv` (if amd-smi/nvidia-smi is available).

To process the result with InferenceX tooling (e.g. for the same format as CI):

```bash
# Optional: run from repo root with workspace in place
python3 utils/process_result.py   # reads result from workspace
python3 utils/summarize.py         # if you use the full pipeline
```
