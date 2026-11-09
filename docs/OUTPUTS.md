# Outputs and logs

Every orchestrator invocation creates a new directory; nothing is ever overwritten.

```
runs/<experiment>/<model>/<UTC>_<id>/
├── experiment.json        exact experiment + model config (and their SHA-256), the full shuffled plan,
│                          CLI arguments, the start-up health check, proxy port
├── environment.json       upstream path/commit + the diff actually applied, this repo's commit,
│                          pip freeze, Python, GPU + driver, Docker version, vLLM server version
├── config/                copies of the two YAML files used
├── results.jsonl          ONE LINE PER ATTEMPT, append-only (see fields below)
├── orchestrator.log       human-readable progress: "[status] agent scenario repN/attN reason"
├── events.jsonl           orchestrator start/end, circuit-breaker open/close, orchestration errors
├── health.jsonl           endpoint health checks
├── gpu_samples.jsonl      nvidia-smi every 2 s (memory used, utilisation)
├── proxy_errors.jsonl     proxy-level problems (unknown run id, agent disconnected mid-call)
└── runs/<agent>/<scenario_id>/rep<r>/attempt<a>_<runid8>/
    ├── run_spec.json      what the worker was asked to do (port, seed, limits, model)
    ├── run.json           what happened (see below)
    ├── events.jsonl       worker timeline: container_start, ssh_ready, agent_start/end, exception,
    │                      container_removed, worker_end
    ├── llm_calls.jsonl    every LLM call of this run (see below)
    ├── agent_output.log   the agent's complete output (HackSynth: its full step log)
    ├── agent_steps.jsonl  hackingBuddyGPT turns, messages, tool calls (wintermute, PrivEscAgent)
    ├── root_events.jsonl  each root process the monitor saw: pid, comm, cmdline, parent, rule, time
    ├── result.json        the upstream adapter's own result dict (original success, steps, milestones)
    ├── hbgpt.sqlite3      the per-run hackingBuddyGPT database
    ├── agent_output/      raw files the adapter wrote
    └── worker.log         the worker process's stdout/stderr
```

## `results.jsonl` fields

`key` (`agent|scenario|rep`), `run_id`, `attempt`, `agent`, `scenario_id`, `category`, `difficulty`,
`rep`, `seed`, `model_config`, `served_model_name`, `backend`, **`status`**, `recoverable`,
`status_reason`, **`orig_success`**, `orig_steps`, `agent_timeout`, **`verified_root_exec`**,
`verified_root_write`, `llm` (calls, ok, errors by class, prompt/completion tokens, latency total,
429 retries), `llm_latency_p50_s`, `llm_tok_per_s_mean`, `adapter_reported_tokens`, `image_id`,
`ts_start`, `ts_end`, `wall_s`, `worker_rc`, `run_dir`.

A key can appear several times (retries). The evaluation uses the **last** line per key.

## `run.json`

Everything in the result line plus: `verified_exec_detail` (monitor samples, events, errors),
`verified_write_detail` (changed files, new uid-0 accounts, new SUID files, the pre-run snapshot),
`adapter_error`, `container_name`, `image_tag`, `agent_duration_s`, `traceback` (if any).

## `llm_calls.jsonl` (one record per call)

| field | content |
|---|---|
| `run_id`, `agent`, `call_seq`, `role_guess` | attribution (`hacksynth.planner` / `hacksynth.summarizer` / `<agent>.llm`) |
| `ts_start`, `ts_end`, `latency_s`, `tokens_per_s` | timing |
| `messages`, `system_prompt`, `system_prompt_sha256`, `n_messages` | the full prompt as sent by the agent |
| `generation_params` | every request field except the messages (temperature, max_tokens, tools, …) |
| `injected_params` | what the proxy added (`max_tokens`, `seed`), empty if nothing |
| `status_code`, `raw_response` | exactly what the server returned |
| `parsed` | `content`, `tool_calls`, `reasoning_content`, `finish_reason`, `served_model` |
| `usage` | prompt / completion tokens |
| `error_class`, `error_message` | `ok`, `context_overflow`, `upstream_timeout`, `endpoint_unreachable`, `upstream_5xx`, `rate_limited`, `auth_error`, `model_not_found`, `oom`, `invalid_request`, `bad_gateway_response`, `transport_error` |
| `superseded`, `proxy_try`, `proxy_backoff_s` | a refused HTTP 429 attempt that the proxy retried transparently (remote APIs only) |

## Debugging a run

1. `run.json` → `status`, `status_reason`.
2. `events.jsonl` → which stage it reached.
3. `llm_calls.jsonl` → the last few calls (`error_class`, `parsed.content`, `finish_reason`).
4. `agent_output.log` → what the agent printed and which commands it ran.
5. `root_events.jsonl` → whether and when a real root process appeared.

Quick queries:

```bash
D=runs/full/privesc-llm-4b-rl/<dir>
# status counts per agent (last attempt per key)
python3 - "$D" <<'P'
import json, sys, collections
last = {}
for l in open(sys.argv[1] + "/results.jsonl"):
    r = json.loads(l); last[r["key"]] = r
print(collections.Counter((r["agent"], r["status"]) for r in last.values()))
P
# runs where the two criteria disagree
jq -c 'select(.orig_success != .verified_root_exec) | [.agent, .scenario_id, .orig_success, .verified_root_exec, .run_dir]' "$D/results.jsonl"
```

## Reports

`python -m privrep.evaluate <exp_dir> [<exp_dir> …] --out reports/<name>` writes:

* `comparison.md`: research metrics (both criteria, Wilson 95 % CI, per-repetition SR, upstream
  efficiency score and progress rate), criterion agreement, engineering metrics,
* `comparison.json`: all of the above plus per-category counts and the full upstream metric dicts,
* `per_run.csv`: one row per run (model, agent, scenario, category, rep, status, both criteria,
  steps, LLM calls, tokens, attempts, run directory).
