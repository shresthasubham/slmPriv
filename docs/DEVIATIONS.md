# Deviations from the original implementation

Baseline: upstream `yxsec/PrivEscalate` @ `cfb362d` ("Release PrivEscalate camera-ready artifact").
Each item below is either a harness fix that does not touch agent logic, a harness replacement, or a
model-setup choice the paper does not specify (the paper used API models only).

**None of them touches an agent's prompt, parser, tool or control loop.**

## A. Changes to upstream files: `patches/upstream-harness-fixes.patch`

3 files, 19 lines added, 9 removed. Every change is gated by an environment variable; with the
variable unset the file behaves exactly as upstream. `environment.json` of every experiment records
the diff that was actually applied.

| id | file(s) | change | why | effect on methodology |
|---|---|---|---|---|
| D-1 | `evaluation/adapters/wintermute.py`, `privescagent_adapter.py` | `PRIVESC_LOG_DB` sets the hackingBuddyGPT SQLite path per run | upstream used one DB per model and read back "the latest run id", so parallel runs read each other's commands and tokens | none on agents; fixes attribution |
| D-2 | same two adapters | `PRIVESC_CONTEXT_SIZE` sets `--llm.context_size` (upstream: 200 000 hard-coded / provider table) | hackingBuddyGPT trims history to this value; it must match what the server accepts (32 768 here), otherwise long runs crash with HTTP 400 | history is trimmed at the real limit instead of crashing |
| D-3 | same two adapters | `PRIVESC_RESULT_TRUNCATE=0` removes the `[:500]` slice of command output before root detection (default 500 = upstream) | a `uid=0(root)` line after the first 500 characters of a command's output was missed | **the detector is unchanged**; it now sees the full output |
| D-4 | `evaluation/runner.py` | `PRIVESC_CONTAINER_NAME` sets the container name (upstream `eval-<id>-<unix seconds>`) | two runs of one scenario started in the same second collide | none |

The harness sets D-1…D-4 for every run.

**Known upstream issues deliberately NOT changed** (they are agent behaviour, so changing them would
change the method being replicated): HackSynth's `\;` first-step quoting bug, HackSynth's per-step
`continue` on LLM errors and `raw_output[:5000]`, PrivEscAgent's `/etc/passwd` + `su` chain,
hackingBuddyGPT's stale `capability_name` labels, the regex success criteria themselves. (The
harness saves HackSynth's full `output_log` separately.)

## B. New harness (replaces `scripts/run_experiments.py`, reuses upstream components)

| id | what | upstream | here |
|---|---|---|---|
| H-1 | orchestrator | `scripts/run_experiments.py` (in-process) | `privrep.orchestrator`. Reuses upstream `ScenarioRunner`, `build_agent_info` and the three adapters unchanged (apart from §A), and sets the same `PRIVESC_*` variables |
| H-2 | run isolation | adapters in one process, shared env, DB, output dir | one subprocess per run with a minimal env + own DB, output dir, container name, port, seed, run id |
| H-3 | secrets | a `.env` at the repo root, which hackingBuddyGPT auto-loads | `secrets.env` (not on hackingBuddyGPT's dotenv search path), read only by the orchestrator, proxy and server; agents get a dummy key |
| H-4 | LLM logging | hackingBuddyGPT SQLite (wintermute, PrivEscAgent); nothing for HackSynth | a transparent proxy logs every call of all three agents |
| H-5 | seed | none | run seed recorded; for local models added as `seed` when the request has none (recorded in `injected_params`; no effect at temperature 0) |
| H-6 | SSH readiness | fixed start-up delay | the upstream delay, then wait for the sshd banner (≤ 60 s), else `infra_error` |
| H-7 | failure accounting | exceptions only; LLM outages scored as FAIL | 7 statuses; infrastructure failures are not scored, re-queued (≤ 2 retries), all attempts kept |
| H-8 | results file | JSONL rewritten on every save | append-only `results.jsonl` + per-run `run.json`; a new directory per invocation |
| H-9 | concurrency | none | lock file per (experiment, model); per-port flock leases |
| H-10 | run order | by difficulty, then alphabetical | shuffled with the experiment seed |
| H-11 | extra success criterion | – | `verified_root_exec` (process monitor) and `verified_root_write` (file evidence), reported next to the original criterion, never instead of it |

## C. Model setup

| id | choice | reason |
|---|---|---|
| M-1 | Model A = `Qwen/Qwen2.5-7B-Instruct-AWQ` (7.6 B, AWQ 4-bit) | general-purpose open instruct model that fits 16 GB with a 32k context |
| M-2 | Model B = `Qwen/Qwen3-4B-Instruct-2507` + released `sailab-vienna/privesc-llm-4b` `rl_adapter` | the published privesc fine-tune; the release contains adapters only. It comes from TU Wien (arXiv 2603.17673), not from the PrivEscalate paper |
| M-2b | Model B is served as a **merged** checkpoint (`scripts/merge_lora.py`, `lm_head` untied before merging), not through vLLM's LoRA path | through vLLM 0.24.0 LoRA the adapter (which includes an `lm_head` LoRA) did not reproduce the HF + PEFT reference (first wintermute prompt: 4 096-token runaway vs `ls /etc/passwd`). Check the merged checkpoint against the HF + PEFT reference with `scripts/equivalence.sh` (docs/MODELS.md) |
| M-3 | context 32 768 for both (`max_model_len` = `context_size`) | Qwen2.5's native limit; fits in memory; 8 192 caused overflows in an earlier run |
| M-4 | `--no-enable-prefix-caching`, `--seed 0` | removes avoidable nondeterminism |
| M-5 | `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN` | without the CUDA toolkit (`nvcc`) FlashInfer's JIT sampler cannot be built |
| M-6 | vLLM `--generation-config auto` (default) | the model's `generation_config.json` defaults apply to fields the agent does not set, as with HF `generate` |
| M-7 | proxy adds `max_tokens: 4096` **only when the request has none** (`default_max_tokens`) | hackingBuddyGPT never sets `max_tokens`; API providers cap output themselves, but a local server allows `max_model_len − prompt`, and one degenerate generation then uses the whole 600 s budget. HackSynth sets 1 024 itself and is unaffected |
| M-8 | parallel 8 (full), 6 (smoke) | throughput on one GPU. Batching makes greedy GPU decoding not bit-exact (measure it with `scripts/equivalence.sh`) |

The upstream working-tree edits of an earlier local run (`privenum.py` `timeout 45`, `MAX_STEPS = 8`)
are **not** applied: this harness uses the pinned upstream code plus §A only.
