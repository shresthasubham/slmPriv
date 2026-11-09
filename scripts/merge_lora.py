#!/usr/bin/env python3
"""Reproducibly merge the released LoRA adapter into its base model so vLLM can serve it.

Why: served through vLLM 0.24.0's LoRA path, `sailab-vienna/privesc-llm-4b/rl_adapter` (which
includes an `lm_head` LoRA) produced outputs that do NOT match the HF+PEFT reference. Example: the
wintermute first prompt gave a 4096-token runaway list via vLLM LoRA, but `ls /etc/passwd`
(5 tokens) via HF+PEFT. PEFT's merge_and_unload is mathematically the same model as base+adapter,
in plain weights. See docs/DEVIATIONS.md (M-2b) and docs/MODELS.md.

Run with the vLLM env python (torch, transformers, peft), on CPU (no GPU needed, ~16 GB RAM):
    .venv-vllm/bin/python scripts/merge_lora.py configs/models/privesc-llm-4b-rl.yaml
The output directory and MERGE_MANIFEST.json (inputs, versions, SHA-256 of every output file) are
written to the config's `checkpoint.local_path` (relative paths are relative to the project root).
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def snap(repo, rev):
    home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return home / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
mf = cfg["checkpoint"]["merged_from"]
out = Path(os.path.expanduser(cfg["checkpoint"]["local_path"]))
out = out if out.is_absolute() else PROJECT_ROOT / out
base = snap(mf["base_repo"], mf["base_revision"])
adapter = snap(mf["lora_repo"], mf["lora_revision"]) / mf["lora_subfolder"]
for need in (base / "config.json", adapter / "adapter_config.json"):
    if not need.exists():
        sys.exit(f"missing {need}; run scripts/download_models.py first")
if out.exists():
    sys.exit(f"refusing to overwrite existing {out}; remove it deliberately to re-merge")

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
# The base ties lm_head to embed_tokens, and the adapter has an lm_head LoRA. At runtime PEFT applies
# that delta ONLY to the output projection. A naive merge would write it into the shared tensor and
# change the input embeddings too. So untie first: lm_head gets its own copy of the base weights.
untied = bool(getattr(model.config, "tie_word_embeddings", False))
if untied:
    model.lm_head.weight = torch.nn.Parameter(model.get_input_embeddings().weight.detach().clone())
    model.config.tie_word_embeddings = False
    if hasattr(model, "_tied_weights_keys"):
        model._tied_weights_keys = []
emb_before = model.get_input_embeddings().weight.detach().clone()
model = PeftModel.from_pretrained(model, str(adapter))
model = model.merge_and_unload(safe_merge=True)
assert torch.equal(model.get_input_embeddings().weight, emb_before), "input embeddings changed by merge"
assert model.lm_head.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr()
out.mkdir(parents=True)
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
for extra in ("generation_config.json",):
    if (base / extra).exists():
        shutil.copy(base / extra, out / extra)

import peft
import transformers
manifest = {
    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "duration_s": round(time.time() - t0, 1),
    "base": {"repo": mf["base_repo"], "revision": mf["base_revision"]},
    "adapter": {"repo": mf["lora_repo"], "revision": mf["lora_revision"], "subfolder": mf["lora_subfolder"],
                "adapter_sha256": sha(adapter / "adapter_model.safetensors")},
    "method": "peft.PeftModel.merge_and_unload(safe_merge=True), dtype bfloat16, on CPU",
    "untied_lm_head": untied,
    "note": "lm_head untied from embed_tokens before merging, so the lm_head LoRA delta changes only the output projection (as PEFT does at runtime)",
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    "files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
}
(out / "MERGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
