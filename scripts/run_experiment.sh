#!/usr/bin/env bash
# Run ONE experiment with ONE model: start vLLM for it (unless already serving it), run all planned
# runs, stop the server again (only if this script started it).
#
#   bash scripts/run_experiment.sh configs/experiments/smoke.yaml configs/models/privesc-llm-4b-rl.yaml
#   bash scripts/run_experiment.sh configs/experiments/full.yaml  configs/models/qwen2.5-7b-instruct-awq.yaml --parallel 8
#
# Extra arguments go to the orchestrator: --parallel N, --agents a,b, --reps N, --limit N, --dry-run,
# --resume runs/<exp>/<model>/<dir>
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
[ $# -ge 2 ] || { sed -n '2,10p' "$0"; exit 2; }
EXP="$1"; MODEL="$2"; shift 2
trap serve_down EXIT
serve_up "$MODEL" || exit 3
$PY -m privrep.orchestrator run --experiment "$EXP" --model "$MODEL" "$@"
