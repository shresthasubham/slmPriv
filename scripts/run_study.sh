#!/usr/bin/env bash
# The complete study: for each model, start vLLM -> smoke gate (2 scenarios x 3 agents) -> full run
# (531 scenarios x 3 agents) -> stop vLLM. Then build the comparison reports.
#
#   bash scripts/run_study.sh                       # both models, parallel 8
#   PARALLEL=4 bash scripts/run_study.sh            # less parallelism (smaller GPU / fewer CPUs)
#   MODELS="configs/models/qwen2.5-7b-instruct-awq.yaml" bash scripts/run_study.sh
#
# Takes ~10-13 h per model on one RTX 5060 Ti 16 GB at parallel 8. Safe to run detached:
#   setsid nohup bash scripts/run_study.sh > runs/study.out 2>&1 < /dev/null &
# If interrupted, rerun a model's full run with --resume (see README, "Resuming").
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
PARALLEL=${PARALLEL:-8}
MODELS=${MODELS:-"configs/models/privesc-llm-4b-rl.yaml configs/models/qwen2.5-7b-instruct-awq.yaml"}
mkdir -p runs
LOG=runs/study_$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$LOG") 2>&1
trap serve_down EXIT
echo "== study start $(date -u) | parallel=$PARALLEL | models: $MODELS"
$PY scripts/validate.py || { echo "validation failed; fix the FAIL lines above" >&2; exit 1; }

for M in $MODELS; do
  echo "== $(date -u) model $M"
  serve_up "$M" || { echo "SKIPPING $M (server did not start)"; serve_down; continue; }
  echo "-- smoke gate"
  $PY -m privrep.orchestrator run --experiment configs/experiments/smoke.yaml --model "$M" --parallel 6 \
    || { echo "smoke gate failed for $M; skipping its full run"; serve_down; continue; }
  echo "-- full run (parallel $PARALLEL)"
  $PY -m privrep.orchestrator run --experiment configs/experiments/full.yaml --model "$M" --parallel "$PARALLEL"
  echo "-- done $M rc=$? at $(date -u)"
  serve_down
done

echo "== $(date -u) building reports"
latest() { ls -td runs/"$1"/"$2"/*/ 2>/dev/null | head -1; }
FULL=""; SMOKE=""
for M in $MODELS; do n=$(basename "$M" .yaml); FULL="$FULL $(latest full "$n")"; SMOKE="$SMOKE $(latest smoke "$n")"; done
[ -n "${FULL// /}" ]  && $PY -m privrep.evaluate $FULL  --out reports/full
[ -n "${SMOKE// /}" ] && $PY -m privrep.evaluate $SMOKE --out reports/smoke
echo "== study end $(date -u). Reports: reports/full/comparison.md"
