#!/usr/bin/env bash
#
# Run process_result.py for local Qwen3.5 MI355X benchmark result(s) (BF16 or FP8).
# With no args: process the single result specified by [CONC] [ISL] [OSL] [PRECISION] (default 4, 1024, 1024, bf16).
# With --all: recursively find all raw *.json under WORKSPACE and process each (skips agg_*.json); precision inferred from filename.
#
# Usage:
#   ./scripts/process_result_bf16_local.sh [CONC] [ISL] [OSL] [PRECISION]   # single result (PRECISION=bf16|fp8, default bf16)
#   ./scripts/process_result_bf16_local.sh --all                            # process all raw results under workspace
#
# Override workspace: INFERENCEX_WORKSPACE=/path/to/workspace
# Override single result: RESULT_FILENAME=exact_filename_without_dot_json  or  PRECISION=bf16|fp8
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="${INFERENCEX_WORKSPACE:-$REPO_ROOT/bench_workspace}"

# --- Check if raw benchmark result (has keys required by process_result.py)
is_raw_result() {
  python3 -c "
import json, sys
try:
  d = json.load(open(sys.argv[1]))
  sys.exit(0 if 'max_concurrency' in d and 'model_id' in d and 'total_token_throughput' in d else 1)
except Exception:
  sys.exit(1)
" "$1" 2>/dev/null
}

# --- Parse CONC from JSON (fallback when filename doesn't match our pattern)
get_conc_from_json() {
  python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['max_concurrency'])" "$1" 2>/dev/null || echo "4"
}

# --- Process one raw result file
process_one() {
  local json_file="$1"
  local dir base res
  dir="$(dirname "$json_file")"
  base="$(basename "$json_file" .json)"
  res=0

  # Skip already aggregated
  if [[ "$base" == agg_* ]]; then
    return 0
  fi

  # Parse optional ISL, OSL, PRECISION, CONC from filename (e.g. qwen3.5_1024_1024_bf16_sglang_..._conc4_local)
  local ISL=1024 OSL=1024 PRECISION=bf16 CONC
  CONC=$(get_conc_from_json "$json_file")
  if [[ "$base" =~ ^qwen3\.5_([0-9]+)_([0-9]+)_(bf16|fp8)_ ]]; then
    ISL="${BASH_REMATCH[1]}"
    OSL="${BASH_REMATCH[2]}"
    PRECISION="${BASH_REMATCH[3]}"
  fi
  if [[ "$base" =~ _conc([0-9]+)_local$ ]]; then
    CONC="${BASH_REMATCH[1]}"
  fi

  # Extract TP/EP/DPA settings from filename (e.g. ..._sglang_tp4-ep1-dpaFalse_...) instead of relying on environment variables
  local TP_FROM_NAME EP_FROM_NAME DPA_FROM_NAME
  TP_FROM_NAME=""
  EP_FROM_NAME=""
  DPA_FROM_NAME=""
  if [[ "$base" =~ sglang_tp([0-9]+)- ]]; then
    TP_FROM_NAME="${BASH_REMATCH[1]}"
  fi
  if [[ "$base" =~ -ep([0-9]+)- ]]; then
    EP_FROM_NAME="${BASH_REMATCH[1]}"
  fi
  if [[ "$base" =~ -dpa(True|False)_ ]]; then
    DPA_FROM_NAME="${BASH_REMATCH[1],,}"
  fi

  local IMAGE="rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315"

  export RESULT_FILENAME="$base"
  export RUNNER_TYPE="${RUNNER_TYPE:-mi355x}"
  export FRAMEWORK="${FRAMEWORK:-sglang}"
  export PRECISION
  export SPEC_DECODING="${SPEC_DECODING:-none}"
  export DISAGG="${DISAGG:-false}"
  export MODEL_PREFIX="${MODEL_PREFIX:-qwen3.5}"
  export IMAGE
  export TP="${TP_FROM_NAME:-${TP:-8}}"
  export EP_SIZE="${EP_FROM_NAME:-${EP_SIZE:-1}}"
  export DP_ATTENTION="${DPA_FROM_NAME:-${DP_ATTENTION:-false}}"
  export ISL
  export OSL

  echo "[process_result] $json_file -> agg_${base}.json (ISL=$ISL OSL=$OSL PRECISION=$PRECISION TP=$TP EP=$EP_SIZE)"
  ( cd "$dir" && python3 "$REPO_ROOT/utils/process_result.py" ) || res=$?
  return $res
}

# --- Main
if [[ "${1:-}" == "--all" ]]; then
  if [[ ! -d "$WORKSPACE" ]]; then
    echo "Error: Workspace not found: $WORKSPACE"
    exit 1
  fi
  count=0
  failed=0
  while IFS= read -r -d '' f; do
    if [[ "$(basename "$f")" == agg_* ]]; then continue; fi
    if ! is_raw_result "$f"; then continue; fi
    if process_one "$f"; then
      ((count++)) || true
    else
      ((failed++)) || true
    fi
  done < <(find "$WORKSPACE" -name '*.json' -type f -print0 2>/dev/null)
  echo "[process_result] Processed $count file(s), $failed failed."
  [[ $failed -eq 0 ]]
  exit
fi

# --- Single-result path: CONC ISL OSL PRECISION (precision default bf16)
CONC="${1:-${CONC:-4}}"
ISL="${2:-${ISL:-1024}}"
OSL="${3:-${OSL:-1024}}"
PRECISION="${4:-${PRECISION:-bf16}}"
if [[ "$PRECISION" != bf16 && "$PRECISION" != fp8 ]]; then
  echo "Error: PRECISION must be bf16 or fp8 (got: $PRECISION)"
  exit 1
fi

TP="${TP:-8}"
if [[ -z "${RESULT_FILENAME:-}" ]]; then
  export RESULT_FILENAME="qwen3.5_${ISL}_${OSL}_${PRECISION}_sglang_tp${TP}-ep1-dpaFalse_disagg-false_spec-none_conc${CONC}_local"
fi

IMAGE="${IMAGE:-rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260315}"

export RUNNER_TYPE="${RUNNER_TYPE:-mi355x}"
export FRAMEWORK="${FRAMEWORK:-sglang}"
export PRECISION
export SPEC_DECODING="${SPEC_DECODING:-none}"
export DISAGG="${DISAGG:-false}"
export MODEL_PREFIX="${MODEL_PREFIX:-qwen3.5}"
export IMAGE
export TP
export EP_SIZE="${EP_SIZE:-1}"
export DP_ATTENTION="${DP_ATTENTION:-false}"
export ISL
export OSL

if [[ ! -d "$WORKSPACE" ]]; then
  echo "Error: Workspace not found: $WORKSPACE"
  exit 1
fi
if [[ ! -f "$WORKSPACE/${RESULT_FILENAME}.json" ]]; then
  echo "Error: Raw result not found: $WORKSPACE/${RESULT_FILENAME}.json"
  exit 1
fi

cd "$WORKSPACE"
python3 "$REPO_ROOT/utils/process_result.py"

echo "[process_result] Wrote $WORKSPACE/agg_${RESULT_FILENAME}.json"
