#!/usr/bin/env python3
"""Download ONLY the pinned Hugging Face revisions named in configs/models/*.yaml.

    .venv-vllm/bin/python scripts/download_models.py            # all local models
    .venv-vllm/bin/python scripts/download_models.py privesc     # only configs whose name contains "privesc"

Run it with the vLLM env python (it needs huggingface_hub). It never runs repository code
(no trust_remote_code). It prints the SHA-256 of every downloaded weight file, so you can compare
them with docs/MODELS.md. For a merged model (Model B) it downloads the base and the LoRA adapter
named in `checkpoint.merged_from`; then run scripts/merge_lora.py.
"""
import hashlib
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


want = sys.argv[1:]
for cfg in sorted((ROOT / "configs" / "models").glob("*.yaml")):
    m = yaml.safe_load(cfg.read_text())
    if want and not any(w in m["name"] for w in want):
        continue
    if m["backend"] != "vllm_openai":
        print(f"skip {m['name']} (remote backend, nothing to download)")
        continue
    ck = m["checkpoint"]
    targets = [(ck["repo"], ck["revision"])]
    if ck.get("merged_from"):
        mf = ck["merged_from"]
        targets = [(mf["base_repo"], mf["base_revision"]), (mf["lora_repo"], mf["lora_revision"])]
    if m.get("lora"):
        targets.append((m["lora"]["repo"], m["lora"]["revision"]))
    for repo, rev in dict.fromkeys(targets):
        p = Path(snapshot_download(repo, revision=rev))
        print(f"{m['name']}: {repo}@{rev} -> {p}")
        for w in sorted(p.rglob("*.safetensors")):
            print(f"   sha256 {sha(w)}  {w.relative_to(p)}")
    if ck.get("merged_from"):
        print(f"   next: .venv-vllm/bin/python scripts/merge_lora.py {cfg.relative_to(ROOT)}")
