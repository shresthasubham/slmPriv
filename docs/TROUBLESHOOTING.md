# Troubleshooting

Run `.venv/bin/python scripts/validate.py` first; most problems show up there as a `FAIL` line.

## Setup

| symptom | cause / fix |
|---|---|
| `setup.sh`: `patch does not apply` | `upstream/` is not a clean checkout of `cfb362d`. Remove it (or `git -C upstream checkout -- .`) and rerun |
| `setup.sh`: `has local changes … refusing to touch it` | you edited `upstream/`; commit/stash or use a fresh clone (`PRIVREP_UPSTREAM=/other/path`) |
| `validate.py`: `no .env on hackingBuddyGPT's dotenv search path` FAIL | a `.env` file exists in this directory or a parent. Rename/move it; secrets belong in `secrets.env` |
| `pytest` exits with `upstream checkout not found` | run `scripts/setup.sh` first, or set `PRIVREP_UPSTREAM` |
| `setup_vllm.sh` fails resolving `torch==2.11.0+cu130` | the `+cu130` build comes from the PyTorch index; make sure `UV_TORCH_BACKEND=cu130` is set (the script sets it) and your driver supports CUDA 13 |
| `CUDA available: False` | driver too old for CUDA 13, or no GPU visible to the user (`nvidia-smi`) |

## Serving

| symptom | cause / fix |
|---|---|
| vLLM start-up error mentioning FlashInfer / `nvcc` / JIT | keep the `env:` block of the model config (`VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`) |
| CUDA out of memory at start-up | another process uses the GPU (`nvidia-smi`), or lower `gpu_memory_utilization` / `max_model_len` (then `context_size` too: they must stay equal) |
| `another model is being served on :8000` | one model per 16 GB GPU; stop the other server |
| `checkpoint … not found` / `MERGE_MANIFEST.json missing` | run `scripts/download_models.py` / `scripts/merge_lora.py` |
| `401 Unauthorized` from `curl localhost:8000/v1/models` | expected without the key: add `-H "Authorization: Bearer <VLLM_API_KEY>"` |
| server log: `vLLM exited` | see `runs/serve_logs/<model>_<time>.log` |

## Running

| symptom | cause / fix |
|---|---|
| `another orchestrator holds …lock` | the same experiment × model is already running (lock file in `runs/.locks/`). Two different experiments or models can run side by side, but only one model fits on the GPU |
| `endpoint health check failed` | vLLM is not serving the model named in the config (`served_model_name`) or the key differs |
| many `infra_error … llm_upstream_timeout` | the GPU is overloaded: lower `--parallel` |
| `infra_error … ssh_not_ready` | the container started but sshd did not answer within 60 s (heavy load); it is retried automatically |
| `build_error` on the same few scenarios | their Dockerfile does not build in your environment (typically a package or download URL in the upstream Dockerfile that no longer resolves). Excluded from success rates and reported separately; check with `docker build upstream/dataset/scenarios/core/<id>` |
| `run.json`: `monitor_error: cgroup.procs not found` | the verified criterion needs rootful Docker on cgroup v2 (`/sys/fs/cgroup/cgroup.controllers` must exist). Original-criterion results are unaffected |
| leftover containers after a crash | `docker ps -a --filter name=privrep- -q \| xargs -r docker rm -f` |
| the job stopped (reboot, power cut) | resume: `bash scripts/run_experiment.sh <exp> <model> --parallel 8 --resume runs/<exp>/<model>/<dir>` |
| the machine suspends overnight | disable sleep for the duration; a suspended host pauses the agents and can turn runs into timeouts |
