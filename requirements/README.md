Pinned environments (`uv pip freeze` of the environments that produced the reported results).

| file | environment | install |
|---|---|---|
| `harness.txt` | `.venv` — agents, orchestrator, proxy, evaluation (Python 3.12) | `uv pip install --python .venv/bin/python -r requirements/harness.txt` |
| `vllm.txt` | `.venv-vllm` — vLLM 0.24.0, torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2 | `UV_TORCH_BACKEND=cu130 uv pip install --python .venv-vllm/bin/python -r requirements/vllm.txt` |

The two are kept apart so the torch/CUDA stack of vLLM never conflicts with the harness.
hackingBuddyGPT is not in `harness.txt`; `scripts/setup.sh` installs it in editable mode from the
upstream checkout (`upstream/baselines/hackingBuddyGPT`).
