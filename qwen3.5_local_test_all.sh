#!/usr/bin/env bash
#
# Run Qwen3.5 BF16 and FP8 local benchmarks on MI355X.
# From InferenceX repo root.
#
# Matrix:
#   Precisions: BF16, FP8
#   Concurrency: 4, 8, 16, 32, 64
#   Seq lengths: (ISL 1024, OSL 1024), (ISL 8192, OSL 1024)
#

set -e

CONC_LIST=(4 8 16 32 64)
# (ISL, OSL) pairs: 1k1k, 8k1k
SEQ_CONFIGS=(1024:1024 8192:1024)

for isl_osl in "${SEQ_CONFIGS[@]}"; do
  IFS=: read -r isl osl <<< "$isl_osl"
  for conc in "${CONC_LIST[@]}"; do
    echo "=== BF16 CONC=$conc ISL=$isl OSL=$osl ==="
    ./scripts/run_qwen3.5_bf16_mi355x_local.sh "$conc" "$isl" "$osl"
  done
done

for isl_osl in "${SEQ_CONFIGS[@]}"; do
  IFS=: read -r isl osl <<< "$isl_osl"
  for conc in "${CONC_LIST[@]}"; do
    echo "=== FP8 CONC=$conc ISL=$isl OSL=$osl ==="
    ./scripts/run_qwen3.5_fp8_mi355x_local.sh "$conc" "$isl" "$osl"
  done
done

echo "=== All runs finished ==="
