# Shared shell helpers (sourced by the other scripts). Run from the project root.
PY=${PY:-.venv/bin/python}
VLLM_PORT=8000
SERVE_PID=""

setting() {  # setting NAME DEFAULT : environment, then secrets.env, then default
  local v="${!1:-}"
  if [ -z "$v" ] && [ -f secrets.env ]; then v=$(grep -E "^$1=" secrets.env | tail -1 | cut -d= -f2- | tr -d "\"'"); fi
  echo "${v:-$2}"
}

served_name() { grep -E '^served_model_name:' "$1" | awk '{print $2}'; }

serving() {  # serving MODEL_CFG : is vLLM on :8000 serving this model?
  curl -s -m 3 "http://127.0.0.1:$VLLM_PORT/v1/models" -H "Authorization: Bearer $(setting VLLM_API_KEY '')" \
    | grep -q "\"$(served_name "$1")\""
}

serve_up() {  # serve_up MODEL_CFG : start vLLM for this model (unless already serving it) and wait until healthy
  local cfg="$1" name log
  name=$(basename "$cfg" .yaml)
  if serving "$cfg"; then echo "[serve] $name is already being served on :$VLLM_PORT"; SERVE_PID=""; return 0; fi
  if curl -s -m 3 "http://127.0.0.1:$VLLM_PORT/version" >/dev/null; then
    echo "[serve] ERROR: another model is being served on :$VLLM_PORT; stop it first (one model fits on the GPU)" >&2
    return 1
  fi
  mkdir -p runs/serve_logs
  log="runs/serve_logs/${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "[serve] starting vLLM for $name (log: $log)"
  $PY -m privrep.serve "$cfg" >"$log" 2>&1 &
  SERVE_PID=$!
  for _ in $(seq 1 180); do
    if serving "$cfg"; then echo "[serve] $name ready"; return 0; fi
    kill -0 "$SERVE_PID" 2>/dev/null || { echo "[serve] ERROR: vLLM exited; see $log" >&2; SERVE_PID=""; return 1; }
    sleep 5
  done
  echo "[serve] ERROR: vLLM did not become ready in 15 min; see $log" >&2
  return 1
}

serve_down() {  # stop the vLLM server THIS script started (never one started by someone else)
  [ -n "$SERVE_PID" ] || return 0
  echo "[serve] stopping vLLM (pid $SERVE_PID)"
  kill "$SERVE_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$SERVE_PID" 2>/dev/null || break; sleep 2; done
  kill -9 "$SERVE_PID" 2>/dev/null || true
  SERVE_PID=""
  sleep 10   # let the GPU memory be released before the next model starts
}
