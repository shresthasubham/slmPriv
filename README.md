# PrivEscalate replication harness: three LLM agents × local small models

A reproducible re-run of the **PrivEscalate** three-agent privilege-escalation benchmark
([yxsec/PrivEscalate](https://github.com/yxsec/PrivEscalate) @ `cfb362d`) on **small open models
served locally with vLLM**, with every LLM call logged and every run isolated.

| | |
|---|---|
| **Agents** (compared independently, unchanged) | PrivEscAgent · wintermute (hackingBuddyGPT) · HackSynth |
| **Models** | A = `Qwen/Qwen2.5-7B-Instruct-AWQ` (general instruct, 4-bit) · B = `sailab-vienna/privesc-llm-4b` `rl_adapter` on `Qwen3-4B-Instruct-2507` (privesc fine-tune) |
| **Scenarios** | the 531 upstream "core" Docker scenarios (GTFOBins / Exploit-DB / expert), zero-knowledge level 0 |
| **Budget** | 20 steps, 600 s per run, temperature 0 (the upstream values) |
| **Metrics** | the upstream `evaluation/metrics.py`, unchanged; plus an independent "verified root" criterion and engineering metrics |

The agents' prompts, parsers, tools and control loops are **not modified**. This repository adds a
new harness around them (isolation, logging, error handling, evaluation) and a 19-line, env-gated
patch to three upstream harness files. Every difference from the original is listed in
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

> **Safety.** The agents run real privilege-escalation commands. Run them **only** against the
> bundled local Docker scenarios, on a machine you control. Never point them at systems you are not
> authorised to test. See the upstream `ETHICS.md`.

---

## Contents

1. [How it works (one picture)](#how-it-works)
2. [Requirements](#requirements)
3. [Step-by-step: from zero to results](#step-by-step)
4. [Running experiments](#running-experiments)
5. [Where the results and logs are](#where-the-results-and-logs-are)
6. [Repository layout](#repository-layout)
7. [Reproducibility notes](#reproducibility-notes)
8. [Troubleshooting](#troubleshooting)
9. [Credits, citation, license](#credits-citation-license)

More detail: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) ·
[`docs/MODELS.md`](docs/MODELS.md) · [`docs/OUTPUTS.md`](docs/OUTPUTS.md) ·
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) · [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)

---

## How it works

```
                                      ┌───────────────────────── one run (own process) ─────────────────────────┐
 orchestrator ──(8 parallel slots)──▶ │ worker → upstream adapter → agent ──SSH──▶ scenario container (Docker)    │
 (queue, retries,                     │                               │                    ▲                     │
  locks, resume)                      │                               │ HTTP               │ root-process monitor │
                                      └───────────────────────────────┼────────────────────┼─────────────────────┘
                                                                      ▼                    │ (host side, every 10 ms)
                                       logging proxy  ── /r/<run_id>/<agent>/v1 ──▶  vLLM server (one model on the GPU)
                                       (writes llm_calls.jsonl for that run)
```

* The **orchestrator** shuffles all (agent × scenario × repetition) runs into one queue and runs
  N at a time. Each run gets its **own process, container, SSH port, SQLite DB, output folder and
  seed**, so runs never see each other's state.
* The **worker** calls the *original* upstream adapter (`evaluation/adapters/*.py`), which starts
  the *original* agent with the same environment variables the upstream `run_experiments.py` sets.
* The agent's LLM requests go to a **logging proxy** (the run id is in the URL), which forwards them
  to **vLLM** and stores the full request/response of every call in that run's folder.
* Success is scored twice: by the **original detector** (`uid=0(root)` in the command output) and
  by an independent **root-process monitor** that watches the container's real processes.

Full explanation: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

---

## Requirements

| | tested with | minimum |
|---|---|---|
| OS | Ubuntu 24.04, Linux 7.0 | Linux with **cgroup v2** (needed by the root monitor) |
| GPU | 1 × NVIDIA RTX 5060 Ti 16 GB, driver 615.71 | 16 GB VRAM, driver with CUDA 13 support |
| CPU / RAM | 16 cores / 31 GB | 8 cores / 16 GB (merging Model B needs ~16 GB RAM) |
| Docker | 29.1.3, rootful | rootful Docker (the monitor reads `/sys/fs/cgroup/.../docker-<id>.scope`) |
| Python | 3.12.3 | 3.10–3.13 |
| Tools | `uv` 0.13.7, `git`, `curl` | |
| Disk | | ~35 GB: models ~20 GB, scenario images ~10 GB (≈ 150 MB × 531), run logs ~2 GB per model |

The CUDA toolkit (`nvcc`) is **not** needed.

---

## Step-by-step

All commands run from the repository root.

### 1. Upstream code + harness environment

```bash
bash scripts/setup.sh
```

This:
1. clones `https://github.com/yxsec/PrivEscalate` into `./upstream` and checks out the pinned
   commit **`cfb362d`** (the upstream code is not redistributed here, see [license](#credits-citation-license)),
2. applies [`patches/upstream-harness-fixes.patch`](patches/upstream-harness-fixes.patch)
   (19 lines, 3 files, all env-gated; listed in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) §A),
3. creates `.venv` (Python 3.12) from the pinned [`requirements/harness.txt`](requirements/harness.txt)
   and installs hackingBuddyGPT from the checkout in editable mode,
4. creates `secrets.env` with a random `VLLM_API_KEY`.

> `secrets.env` is deliberately **not** called `.env`: hackingBuddyGPT auto-loads any `.env` it
> finds above its source folder (which includes this directory) and would map its keys onto the
> agent's parameters. `scripts/validate.py` checks this.

### 2. vLLM environment (separate)

```bash
bash scripts/setup_vllm.sh
```

Creates `.venv-vllm` with **vLLM 0.24.0**, torch 2.11.0+cu130, transformers 5.22.1 and peft 0.21.2
from [`requirements/vllm.txt`](requirements/vllm.txt), and prints whether CUDA is visible. It is
separate from `.venv`, so the two torch stacks never conflict.

### 3. Model weights

```bash
.venv-vllm/bin/python scripts/download_models.py          # pinned revisions only; prints SHA-256 of every weight file
.venv-vllm/bin/python scripts/merge_lora.py configs/models/privesc-llm-4b-rl.yaml    # Model B only, on CPU, about 1 min
```

| role | config | served as | weights (pinned revision) |
|---|---|---|---|
| **A** | [`qwen2.5-7b-instruct-awq.yaml`](configs/models/qwen2.5-7b-instruct-awq.yaml) | `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct-AWQ@b2503754…` |
| **B** | [`privesc-llm-4b-rl.yaml`](configs/models/privesc-llm-4b-rl.yaml) | `privesc-llm-4b` | `Qwen/Qwen3-4B-Instruct-2507@cdbee75f…` + LoRA `sailab-vienna/privesc-llm-4b/rl_adapter@e7a10aad…`, merged |

Why Model B is merged instead of served through vLLM's LoRA support, the merge procedure, expected
SHA-256 hashes and the exact `vllm serve` commands: **[`docs/MODELS.md`](docs/MODELS.md)**.

### 4. Validate everything

```bash
.venv/bin/python -m pytest -q             # 49 unit/isolation tests; no GPU, Docker or model needed
.venv/bin/python scripts/validate.py      # upstream commit + patch, configs, deps, checkpoints, merge manifest,
                                          # vLLM version, GPU, Docker, cgroup v2, secrets
```

Every check prints `PASS` / `FAIL` / `SKIP`; fix all `FAIL`s before running. Expected: `28 checks: 28 pass`.

### 5. (Optional) pre-build the scenario images

```bash
.venv/bin/python scripts/build_images.py configs/experiments/full.yaml --jobs 4
```

Otherwise each image is built the first time a run needs it (same tag, same command).

### 6. Smoke test (≈ 10 min)

```bash
bash scripts/run_experiment.sh configs/experiments/smoke.yaml configs/models/privesc-llm-4b-rl.yaml
```

Starts vLLM for the model, runs 2 scenarios × 3 agents, stops vLLM. Check the summary it prints
(`status_counts`) and look inside `runs/smoke/privesc-llm-4b-rl/<timestamp>/`.

### 7. The full study

```bash
setsid nohup bash scripts/run_study.sh > runs/study.out 2>&1 < /dev/null &
tail -f runs/study.out
```

For each model: start vLLM → smoke gate → full run (531 × 3 = 1 593 runs, parallel 8) → stop vLLM.
Then it writes the reports to `reports/full/` and `reports/smoke/`. `setsid nohup` keeps it running
after you close the terminal. **Expect ≈ 12–13 h per model** on the tested machine (most failing
runs use their full 20-step budget). Use `PARALLEL=4` on smaller machines.

### 8. Reports

```bash
.venv/bin/python -m privrep.evaluate runs/full/privesc-llm-4b-rl/<dir> runs/full/qwen2.5-7b-instruct-awq/<dir> --out reports/full
```

(`run_study.sh` does this for you.) Output: `comparison.md` (tables), `comparison.json` (all numbers)
and `per_run.csv` (one row per run).

### 9. (Optional) output-equivalence study

```bash
bash scripts/equivalence.sh configs/models/privesc-llm-4b-rl.yaml runs/smoke/privesc-llm-4b-rl/<dir> 20
```

Checks that vLLM produces the same greedy output as the Hugging Face reference (base + PEFT
adapter), run-to-run and under batching. See [`docs/MODELS.md`](docs/MODELS.md#output-equivalence).

---

## Running experiments

| experiment | file | scenarios | runs per model |
|---|---|---|---|
| smoke | [`smoke.yaml`](configs/experiments/smoke.yaml) | `T1548_001_suid_find`, `T1548_003_sudo_find` | 6 |
| pilot | [`pilot.yaml`](configs/experiments/pilot.yaml) | 33 (≤ 4 per category, fixed sample seed), **3 repetitions** | 297 |
| full | [`full.yaml`](configs/experiments/full.yaml) | all 531, 1 repetition (as upstream) | 1 593 |

Run any experiment × model directly (vLLM must already be serving that model, or use
`scripts/run_experiment.sh`, which starts and stops it):

```bash
.venv/bin/python -m privrep.orchestrator run \
    --experiment configs/experiments/pilot.yaml --model configs/models/qwen2.5-7b-instruct-awq.yaml \
    [--parallel 8] [--agents wintermute,hacksynth] [--reps 3] [--limit 20] [--dry-run]
```

**Resuming.** Every invocation writes a *new* folder `runs/<experiment>/<model>/<UTC>_<id>/`; nothing
is ever overwritten. If a run is interrupted (reboot, power cut), continue it with:

```bash
bash scripts/run_experiment.sh configs/experiments/full.yaml configs/models/<model>.yaml \
     --parallel 8 --resume runs/full/<model>/<dir>
```

Finished runs are skipped; only missing or retry-able ones are executed.

**Serving a model by hand** (one model fits on a 16 GB GPU at a time):

```bash
.venv/bin/python -m privrep.serve configs/models/privesc-llm-4b-rl.yaml --print   # show the exact vllm command
.venv/bin/python -m privrep.serve configs/models/privesc-llm-4b-rl.yaml           # run it in the foreground
```

**Adding a model.** Copy a file in `configs/models/`, pin `checkpoint.repo` + `revision` (a full
commit hash), set `served_model_name`, keep `context_size == vllm.max_model_len`, run
`scripts/download_models.py <name>` and `scripts/validate.py`. Any OpenAI-compatible endpoint also
works with `backend: remote_openai` + `base_url_env` / `api_key_env`.

---

## Where the results and logs are

```
runs/<experiment>/<model>/<UTC>_<id>/
  results.jsonl          one line per attempt (append-only): status, both success criteria, tokens, timings
  experiment.json        exact configs (+ SHA-256), the full shuffled plan, health check
  environment.json       upstream commit + applied diff, pip freeze, GPU, Docker, vLLM version
  orchestrator.log  events.jsonl  health.jsonl  gpu_samples.jsonl  proxy_errors.jsonl
  runs/<agent>/<scenario>/rep<r>/attempt<a>_<id>/
    run.json             what happened: status, reason, both criteria, LLM summary, traceback
    llm_calls.jsonl      EVERY LLM call: messages, params, raw response, tokens, latency, errors
    agent_output.log     the agent's complete output
    root_events.jsonl    root processes seen by the monitor
    result.json          the original adapter result (original success criterion)
    events.jsonl  agent_steps.jsonl  hbgpt.sqlite3  worker.log  run_spec.json
```

To see why a run failed: `run.json` → `status`, `status_reason` → `events.jsonl` → the last records
of `llm_calls.jsonl`. Every file is described in [`docs/OUTPUTS.md`](docs/OUTPUTS.md).

**Statuses.** `success` / `fail` / `context_overflow` are *scored*. `infra_error` (LLM endpoint or SSH
problems; retried up to 2×), `build_error` (scenario image does not build), `docker_error` and
`harness_error` are *not scored*: they are excluded from success rates and reported separately.

---

## Repository layout

```
.
├── README.md                  this file
├── configs/
│   ├── models/                one YAML per model (pinned weights, vLLM settings)
│   └── experiments/           smoke / pilot / full
├── privrep/                   the harness (Python package)
│   ├── orchestrator.py        queue, parallel slots, retries, locks, resume, provenance
│   ├── worker.py              one isolated run: container → agent → verification → run.json
│   ├── proxy.py               logging OpenAI-compatible proxy (llm_calls.jsonl)
│   ├── rootcheck.py           root-process monitor + file evidence (verified criterion)
│   ├── evaluate.py            original metrics (both criteria) + engineering metrics → reports
│   ├── serve.py               builds/execs the pinned `vllm serve` command
│   ├── config.py              config loading + validation, paths
│   ├── envinfo.py             environment capture
│   ├── models/base.py         OpenAI-compatible client + error taxonomy
│   └── equivalence/compare.py vLLM vs HF output-equivalence tool
├── scripts/
│   ├── setup.sh               step 1: upstream clone + patch + .venv
│   ├── setup_vllm.sh          step 2: .venv-vllm
│   ├── download_models.py     step 3: pinned weights
│   ├── merge_lora.py          step 3: merge Model B's released adapter
│   ├── validate.py            step 4: environment checks
│   ├── build_images.py        step 5: optional image pre-build
│   ├── run_experiment.sh      one experiment × one model (starts/stops vLLM)
│   ├── run_study.sh           everything: both models, smoke + full, reports
│   ├── equivalence.sh         output-equivalence study
│   └── lib.sh                 shared shell helpers
├── patches/upstream-harness-fixes.patch   the only change to upstream code
├── requirements/              pinned environments (harness.txt, vllm.txt)
├── tests/                     pytest suite (no GPU/Docker/model needed)
├── docs/                      HOW_IT_WORKS, MODELS, OUTPUTS, DEVIATIONS, TROUBLESHOOTING
└── secrets.env.example        template for secrets.env (VLLM_API_KEY)

created by the scripts, ignored by Git:
  upstream/  .venv/  .venv-vllm/  models/  runs/  reports/  secrets.env
```

---

## Reproducibility notes

* **Pinned:** upstream commit, every model revision (full commit hashes), vLLM version (checked),
  both Python environments (`uv pip freeze`), experiment seeds (run order and per-run seed).
* **Recorded per experiment:** config files and their SHA-256, the plan, the upstream diff actually
  applied, pip freeze, GPU/driver, Docker and vLLM versions (`experiment.json`, `environment.json`).
* **Decoding:** temperature 0 (greedy), as upstream. vLLM runs with `--seed 0` and prefix caching
  **off**. Greedy decoding on GPUs is still not bit-exact under different batch compositions
  (see the batching row above), so with 8 parallel runs, re-running can change individual
  trajectories. Aggregate rates are the comparable quantity; use `pilot.yaml` (3 repetitions) to
  measure the run-to-run spread.
* **Two success criteria** are always reported side by side: the upstream detector (headline, for
  comparability with the paper) and the independent verified-root monitor. Their disagreement is
  reported in a 2 × 2 table, never hidden.
* **Not scored ≠ failed:** infrastructure and build errors are excluded from rates and counted.

---

## Troubleshooting

Short list (details in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)):

| symptom | fix |
|---|---|
| vLLM fails building a FlashInfer kernel / `nvcc not found` | keep `VLLM_USE_FLASHINFER_SAMPLER=0` and `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (already in the configs) |
| `another model is being served on :8000` | only one model fits; stop the other server first |
| `another orchestrator holds …lock` | the same experiment × model is already running (by design) |
| many `build_error` for the same scenarios | those images do not build on your Docker (e.g. a package mirror changed); they are excluded from success rates and reported separately |
| `cgroup.procs not found` in `run.json` | the root monitor needs rootful Docker with cgroup v2 |
| HTTP 400 "maximum context length" | `context_size` must equal `vllm.max_model_len` (validated) |

---

## Credits, citation, license

* **PrivEscalate** (agents, scenarios, metrics, adapters): <https://github.com/yxsec/PrivEscalate>.
  Please cite it using its `CITATION.cff`. Its README states that no project-wide license is granted
  for the original code and data; therefore this repository does **not** copy it, and
  `scripts/setup.sh` fetches it from GitHub. The patch file contains only the changed lines with
  minimal context.
* **hackingBuddyGPT** (MIT) and **HackSynth** (AGPL-3.0) are part of the upstream checkout under
  their own licenses (see upstream `THIRD_PARTY_NOTICES.md`).
* **Models:** Qwen2.5-7B-Instruct-AWQ and Qwen3-4B-Instruct-2507 (Qwen, Apache-2.0);
  `sailab-vienna/privesc-llm-4b` (see its model card for license and citation, arXiv 2603.17673).
* **This harness** (`privrep/`, `scripts/`, `configs/`, `tests/`, `docs/`): license to be chosen by
  the authors before publication; add a `LICENSE` file.
