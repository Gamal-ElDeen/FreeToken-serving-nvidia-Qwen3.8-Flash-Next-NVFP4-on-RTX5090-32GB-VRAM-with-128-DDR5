#!/usr/bin/env bash
# Reproduce the llama-benchy + aiperf suite against a running FreeToken server.
# Usage: ./scripts/run_bench.sh <case-dir> <context-class: 128k|262k>
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-Qwen3.8-Flash-Next-NVFP4}"
TOKENIZER="${TOKENIZER:-/path/to/Qwen3.8-Flash-Next-NVFP4}"  # local dir or nvidia/Qwen3.8-Flash-Next-NVFP4
RUNS="${RUNS:-3}"

CASE_DIR="${1:?usage: run_bench.sh <case-dir> <128k|262k>}"
CONTEXT="${2:?usage: run_bench.sh <case-dir> <128k|262k>}"

case "$CONTEXT" in
  128k) DEPTHS=(2048 20000 65000 130800); AIPERF_MEANS=(2000 20000 65000 130800); PREFIX=(aiperf_2k aiperf_20k aiperf_65k aiperf_130k) ;;
  262k) DEPTHS=(2048 20000 65000 130800 261800); AIPERF_MEANS=(2000 20000 65000 130800 261800); PREFIX=(aiperf_2k aiperf_20k aiperf_65k aiperf_130k aiperf_262k) ;;
  *) echo "context must be 128k or 262k" >&2; exit 1 ;;
esac

mkdir -p "$CASE_DIR"

echo "== llama-benchy -> $CASE_DIR =="
llama-benchy --base-url "$BASE_URL/v1" \
  --model "$MODEL" \
  --depth "${DEPTHS[@]}" \
  --pp 512 --tg 128 --runs "$RUNS" \
  --save-result "$CASE_DIR/results.json" --format json \
  2>&1 | tee "$CASE_DIR/llamabenchy_full.txt"

echo "== aiperf -> $CASE_DIR =="
for i in "${!AIPERF_MEANS[@]}"; do
  mean="${AIPERF_MEANS[$i]}"; prefix="${PREFIX[$i]}"
  aiperf profile \
    --model "$MODEL" \
    --url "$BASE_URL" \
    --tokenizer "$TOKENIZER" \
    --endpoint-type chat \
    --synthetic-input-tokens-mean "$mean" \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean 128 \
    --streaming \
    --artifact-dir "$CASE_DIR" \
    --profile-export-prefix "$prefix" \
    --export-level summary \
    2>&1 | tee "$CASE_DIR/${prefix}_console.txt"
done

echo "== done. capture the resolved MoE cache from your server unit: =="
echo "   journalctl -u <your-freetoken-unit> --no-pager | grep -F 'moe-cache-auto resolved'"
