#!/usr/bin/env bash
# Output-equivalence study for one model: does serving through vLLM change what the model says?
#   1. take N real agent prompts from a finished experiment,
#   2. greedy-generate them with vLLM (twice at concurrency 1, once at concurrency 8),
#   3. stop vLLM and generate the same prompts with HF transformers (+ PEFT for the released LoRA),
#   4. compare token by token.
#
#   bash scripts/equivalence.sh configs/models/privesc-llm-4b-rl.yaml runs/smoke/privesc-llm-4b-rl/<dir> [N]
#
# The HF reference loads the unquantised weights, so it is only available for Model B. For the AWQ
# Model A (no AutoAWQ reference available) the vLLM self-consistency checks still run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
[ $# -ge 2 ] || { sed -n '2,13p' "$0"; exit 2; }
MODEL="$1"; SRC="$2"; N="${3:-20}"
VPY="$(setting VLLM_VENV .venv-vllm)/bin/python"
name=$(basename "$MODEL" .yaml)
OUT=runs/equivalence/$name/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$OUT"
C="-m privrep.equivalence.compare"
$PY $C extract "$SRC" --n "$N" --out "$OUT/prompts.jsonl"
trap serve_down EXIT
serve_up "$MODEL"
$PY $C vllm --model "$MODEL" --prompts "$OUT/prompts.jsonl" --out "$OUT/vllm_c1.jsonl"
$PY $C vllm --model "$MODEL" --prompts "$OUT/prompts.jsonl" --out "$OUT/vllm_c1_again.jsonl"
$PY $C vllm --model "$MODEL" --prompts "$OUT/prompts.jsonl" --out "$OUT/vllm_c8.jsonl" --concurrency 8
$PY $C compare --a "$OUT/vllm_c1.jsonl" --b "$OUT/vllm_c1_again.jsonl" --out "$OUT/cmp_run_to_run.json" --label "vLLM run-to-run"
$PY $C compare --a "$OUT/vllm_c1.jsonl" --b "$OUT/vllm_c8.jsonl" --out "$OUT/cmp_batching.json" --label "vLLM c1 vs c8"
if grep -q "merged_from:\|^lora:" "$MODEL" || ! grep -q "quantization:" "$MODEL"; then
  serve_down
  if [ -n "$SERVE_PID" ] || curl -s -m 3 "http://127.0.0.1:$VLLM_PORT/version" >/dev/null; then
    echo "a vLLM server is still running (not started by this script); stop it to free the GPU for the HF reference" >&2; exit 4
  fi
  PYTHONPATH=. $VPY $C hf --model "$MODEL" --prompts "$OUT/prompts.jsonl" --out "$OUT/hf.jsonl"
  $PY $C compare --a "$OUT/vllm_c1.jsonl" --b "$OUT/hf.jsonl" --out "$OUT/cmp_vllm_vs_hf.json" --label "vLLM vs HF"
else
  echo "quantised checkpoint: no HF reference generated (see docs/MODELS.md)"
fi
echo "results: $OUT/cmp_*.json"
