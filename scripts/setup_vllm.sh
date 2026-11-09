#!/usr/bin/env bash
# Step 2 of the README: create the separate vLLM environment (.venv-vllm) from the pinned lock file.
#
#   bash scripts/setup_vllm.sh
#
# torch 2.11.0+cu130 comes from the PyTorch CUDA 13.0 index (UV_TORCH_BACKEND=cu130). Needs an NVIDIA
# driver that supports CUDA 13 (tested: 615.71). The CUDA toolkit / nvcc is NOT required.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/lib.sh
VENV=$(setting VLLM_VENV .venv-vllm)
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12
UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-cu130} uv pip install --python "$VENV/bin/python" -r requirements/vllm.txt
"$VENV/bin/python" - <<'P'
import torch, vllm, transformers, peft
print(f"vllm {vllm.__version__} | torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
print("CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
P
echo "== done. Next: $VENV/bin/python scripts/download_models.py"
