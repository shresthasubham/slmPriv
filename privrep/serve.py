"""Build (and optionally exec) the pinned `vllm serve` command for a model config.

    python -m privrep.serve configs/models/privesc-llm-4b-rl.yaml --print   # show the command
    python -m privrep.serve configs/models/privesc-llm-4b-rl.yaml           # exec vllm in the foreground

Checkpoints are resolved to the local Hugging Face snapshot at the pinned revision and served
offline (HF_HUB_OFFLINE=1). Nothing is downloaded at serve time. Use scripts/download_models.py
(and scripts/merge_lora.py for Model B) for that.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from .config import ConfigError, hf_snapshot, load_dotenv, load_model_config, resolve_path, vllm_bin


def build_command(cfg_path: str, port: int = 8000) -> tuple[list[str], dict]:
    m = load_model_config(cfg_path)
    if m.backend != "vllm_openai":
        raise ConfigError(f"{m.name} uses backend {m.backend}; it is not served locally")
    v = m.vllm or {}
    ck = m.checkpoint
    if ck.get("local_path"):
        model_path = resolve_path(ck["local_path"])
        if not (model_path / "MERGE_MANIFEST.json").exists():
            raise ConfigError(f"{model_path} has no MERGE_MANIFEST.json; run scripts/merge_lora.py")
    else:
        model_path = hf_snapshot(ck["repo"], ck["revision"])
    cmd = [vllm_bin(), "serve", str(model_path), "--port", str(port), "--host", "127.0.0.1",
           "--max-model-len", str(v["max_model_len"]),
           "--gpu-memory-utilization", str(v.get("gpu_memory_utilization", 0.9)),
           "--dtype", str(v.get("dtype", "auto")), "--seed", "0"]
    if ck.get("tokenizer") and not ck.get("local_path"):
        cmd += ["--tokenizer", str(hf_snapshot(ck["tokenizer"], ck["revision"]))]
    if v.get("quantization"):
        cmd += ["--quantization", v["quantization"]]
    cmd += ["--enable-prefix-caching"] if v.get("enable_prefix_caching") else ["--no-enable-prefix-caching"]
    if m.lora:
        lp = hf_snapshot(m.lora["repo"], m.lora["revision"]) / m.lora["subfolder"]
        if not (lp / "adapter_config.json").exists():
            raise ConfigError(f"LoRA adapter not found at {lp}")
        base_name = Path(ck["repo"]).name.lower()
        cmd += ["--served-model-name", base_name, "--enable-lora",
                "--lora-modules", f"{m.served_model_name}={lp}",
                "--max-lora-rank", str(m.lora.get("rank", 8)), "--max-loras", "1"]
    else:
        cmd += ["--served-model-name", m.served_model_name]
    cmd += list(v.get("extra_args") or [])
    env = {"HF_HUB_OFFLINE": "1", "VLLM_API_KEY": os.getenv(m.api_key_env, ""),
           **{k: str(val) for k, val in (v.get("env") or {}).items()}}
    return cmd, env


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="privrep.serve")
    ap.add_argument("model_config")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--print", action="store_true", help="print the command and exit")
    a = ap.parse_args(argv)
    load_dotenv()
    cmd, env = build_command(a.model_config, a.port)
    shown = " ".join(shlex.quote(c) for c in cmd)
    envs = " ".join(f"{k}={val}" for k, val in env.items() if k != "VLLM_API_KEY")
    print(f"{envs} VLLM_API_KEY=<from {load_model_config(a.model_config).api_key_env}> {shown}")
    if a.print:
        return 0
    if not env["VLLM_API_KEY"]:
        print("ERROR: API key env var for the local server is empty (set it in secrets.env)", file=sys.stderr)
        return 2
    if not Path(cmd[0]).exists():
        print(f"ERROR: vLLM not found at {cmd[0]} (create .venv-vllm or set VLLM_BIN / VLLM_VENV)", file=sys.stderr)
        return 2
    os.execve(cmd[0], cmd, {**os.environ, **env})
    return 0  # unreachable


if __name__ == "__main__":
    sys.exit(main())
