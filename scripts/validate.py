#!/usr/bin/env python3
"""Startup and configuration validation. Every check prints PASS, FAIL or SKIP with a reason.

    .venv/bin/python scripts/validate.py                    # static checks: upstream, configs, deps, checkpoints, docker, GPU
    .venv/bin/python scripts/validate.py --probe            # + live endpoint check of the model currently served
    .venv/bin/python scripts/validate.py --probe --context  # + context-truncation probe (remote backends only)

The exit code is non-zero if any check FAILs. A JSON report is written to runs/validation/<ts>.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from privrep.config import (CONFIG_ROOT, RUNS_ROOT, UPSTREAM_COMMIT, UPSTREAM_ROOT, ConfigError,  # noqa: E402
                            file_sha256, hf_snapshot, load_dotenv, load_experiment_config, load_model_config,
                            load_scenarios, resolve_path, select_scenarios, vllm_bin)

sys.path.insert(1, str(UPSTREAM_ROOT))

RESULTS: list[dict] = []


def check(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"check": name, "status": status, "detail": detail})
    print(f"[{status:4}] {name}{': ' + detail if detail else ''}")


def upstream_checks() -> bool:
    if not (UPSTREAM_ROOT / "evaluation" / "runner.py").exists():
        check("upstream checkout", "FAIL", f"{UPSTREAM_ROOT} not found (run scripts/setup.sh or set PRIVREP_UPSTREAM)")
        return False
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM_ROOT, capture_output=True, text=True).stdout.strip()
    check(f"upstream at pinned commit {UPSTREAM_COMMIT}", "PASS" if head.startswith(UPSTREAM_COMMIT) else "FAIL",
          f"{UPSTREAM_ROOT} HEAD={head[:12]}")
    patched = all("PRIVESC_" + k in (UPSTREAM_ROOT / f).read_text() for f, k in (
        ("evaluation/adapters/wintermute.py", "LOG_DB"), ("evaluation/adapters/privescagent_adapter.py", "LOG_DB"),
        ("evaluation/runner.py", "CONTAINER_NAME")))
    check("upstream patch patches/upstream-harness-fixes.patch applied", "PASS" if patched else "FAIL",
          "" if patched else "run: git -C upstream apply ../patches/upstream-harness-fixes.patch")
    # hackingBuddyGPT's find_dotenv() walks up from its source dir and auto-loads the first .env it meets
    start = UPSTREAM_ROOT / "baselines" / "hackingBuddyGPT" / "src" / "hackingBuddyGPT" / "utils"
    found = [str(d / ".env") for d in [start, *start.parents] if (d / ".env").exists()]
    check("no .env on hackingBuddyGPT's dotenv search path", "FAIL" if found else "PASS",
          f"would be auto-loaded by the agents: {found}" if found else "")
    return True


def static_checks(args) -> list:
    v = sys.version_info
    check("python 3.10-3.13", "PASS" if (3, 10) <= v[:2] <= (3, 13) else "FAIL", sys.version.split()[0])
    for mod in ("yaml", "requests", "paramiko", "openai", "hackingBuddyGPT", "rich"):
        try:
            importlib.import_module(mod)
            check(f"import {mod}", "PASS")
        except Exception as e:  # noqa: BLE001
            check(f"import {mod}", "FAIL", repr(e))
    try:
        sys.path.insert(0, str(UPSTREAM_ROOT / "baselines" / "HackSynth"))
        importlib.import_module("pentest_agent")
        check("import HackSynth pentest_agent", "PASS")
    except Exception as e:  # noqa: BLE001
        check("import HackSynth pentest_agent", "FAIL", repr(e))

    models = []
    for p in sorted((CONFIG_ROOT / "models").glob("*.yaml")):
        try:
            models.append(load_model_config(p))
            check(f"model config {p.name}", "PASS")
        except ConfigError as e:
            check(f"model config {p.name}", "FAIL", str(e))
    for p in sorted((CONFIG_ROOT / "experiments").glob("*.yaml")):
        try:
            e = load_experiment_config(p)
            n = len(select_scenarios(e))
            check(f"experiment config {p.name}", "PASS", f"{n} scenarios × {len(e.agents)} agents × {e.repetitions} reps")
        except ConfigError as e:
            check(f"experiment config {p.name}", "FAIL", str(e))
    check("corpus size 531", "PASS" if len(load_scenarios("core")) == 531 else "FAIL")

    # checkpoints and tokenizers (local backend only)
    for m in models:
        if m.backend != "vllm_openai":
            ok = bool(os.getenv(m.api_key_env)) and bool(m.base_url or os.getenv(m.base_url_env or ""))
            check(f"{m.name}: API env vars set ({m.api_key_env}, {m.base_url_env})", "PASS" if ok else "FAIL",
                  "values not shown")
            continue
        mf = m.checkpoint.get("merged_from")
        snap = hf_snapshot(mf["base_repo"], mf["base_revision"], must_exist=False) if mf else \
            hf_snapshot(m.checkpoint["repo"], m.checkpoint["revision"], must_exist=False)
        need = ["config.json", "tokenizer_config.json"]
        missing = [f for f in need if not (snap / f).exists()]
        has_w = any(snap.glob("*.safetensors"))
        check(f"{m.name}: checkpoint {snap.parent.parent.name.replace('models--', '').replace('--', '/')}@{snap.name[:8]}",
              "PASS" if snap.exists() and not missing and has_w else "FAIL",
              str(snap) if snap.exists() else "not downloaded (scripts/download_models.py)")
        if mf:
            ad = hf_snapshot(mf["lora_repo"], mf["lora_revision"], must_exist=False) / mf["lora_subfolder"]
            check(f"{m.name}: LoRA adapter {mf['lora_repo']}/{mf['lora_subfolder']}@{mf['lora_revision'][:8]}",
                  "PASS" if (ad / "adapter_config.json").exists() else "FAIL", str(ad))
            lp = resolve_path(m.checkpoint["local_path"])
            man = lp / "MERGE_MANIFEST.json"
            if man.exists():
                mj = json.loads(man.read_text())
                ok = (mj["base"]["revision"] == mf["base_revision"] and mj["adapter"]["revision"] == mf["lora_revision"]
                      and mj.get("untied_lm_head") is True)
                check(f"{m.name}: merged checkpoint (MERGE_MANIFEST matches config, lm_head untied)",
                      "PASS" if ok else "FAIL", f"{lp} model.safetensors sha256={mj['files'].get('model.safetensors', '?')[:16]}")
            else:
                check(f"{m.name}: merged checkpoint", "FAIL", f"{man} missing (run scripts/merge_lora.py)")
        tok = snap / "tokenizer_config.json"
        if tok.exists():
            tc = json.loads(tok.read_text())
            has_tpl = bool(tc.get("chat_template")) or (snap / "chat_template.jinja").exists()
            check(f"{m.name}: tokenizer has chat template", "PASS" if has_tpl else "FAIL")
        if m.lora:
            lp = hf_snapshot(m.lora["repo"], m.lora["revision"], must_exist=False) / m.lora["subfolder"]
            ac = lp / "adapter_config.json"
            if not ac.exists():
                check(f"{m.name}: LoRA {m.lora['subfolder']}", "FAIL", f"missing {ac}")
            else:
                a = json.loads(ac.read_text())
                base_ok = a.get("base_model_name_or_path") == m.checkpoint["repo"]
                check(f"{m.name}: LoRA base model matches checkpoint", "PASS" if base_ok else "FAIL",
                      f"adapter base={a.get('base_model_name_or_path')}")
                check(f"{m.name}: LoRA rank <= max-lora-rank", "PASS" if a.get("r", 0) <= m.lora.get("rank", 8) else "FAIL",
                      f"r={a.get('r')} targets={a.get('target_modules')}")
                st = lp / "adapter_model.safetensors"
                check(f"{m.name}: LoRA weights sha256", "PASS" if st.exists() else "FAIL",
                      file_sha256(st)[:16] if st.exists() else "missing")
    vbin = vllm_bin()
    if Path(vbin).exists():
        py = Path(vbin).parent / "python"
        r = subprocess.run([str(py), "-c", "import vllm;print(vllm.__version__)"], capture_output=True, text=True)
        ver = r.stdout.strip()
        want = {m.vllm["version"] for m in models if m.vllm}
        check("vLLM version matches configs", "PASS" if want == {ver} else "FAIL", f"installed={ver} configs={sorted(want)}")
    else:
        check("vLLM installed", "FAIL", f"{vbin} not found (create .venv-vllm, or set VLLM_VENV / VLLM_BIN)")
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        check("GPU visible", "PASS" if out else "FAIL", out)
    else:
        check("GPU visible", "FAIL", "nvidia-smi not found")
    try:
        dv = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True,
                            text=True, timeout=20)
        check("docker daemon reachable", "PASS" if dv.returncode == 0 else "FAIL", dv.stdout.strip() or dv.stderr.strip()[:200])
    except (OSError, subprocess.TimeoutExpired) as e:
        check("docker daemon reachable", "FAIL", repr(e))
    check("cgroup v2 (needed by root monitor)", "PASS" if Path("/sys/fs/cgroup/cgroup.controllers").exists() else "FAIL")
    check("secrets.env present with VLLM_API_KEY", "PASS" if os.getenv("VLLM_API_KEY") else "FAIL",
          "" if os.getenv("VLLM_API_KEY") else "cp secrets.env.example secrets.env and set VLLM_API_KEY")
    return models


def probe_checks(models, args) -> None:
    from privrep.models.base import make_model
    from privrep.orchestrator import health_check
    for m in models:
        try:
            model = make_model(m, timeout_s=120)
        except ConfigError as e:
            check(f"{m.name}: endpoint", "FAIL", str(e))
            continue
        hc = health_check(model)
        if not hc["ok"] and m.backend == "vllm_openai" and (
                "models_listed" not in hc or m.served_model_name not in hc.get("models_listed", [])):
            why = ("local vLLM server not running" if "models_listed" not in hc
                   else f"the local server is serving another model {hc['models_listed']}")
            check(f"{m.name}: endpoint health", "SKIP", why + " (one model fits on the GPU at a time)")
            continue
        check(f"{m.name}: endpoint health (served name listed + 1-token generation)",
              "PASS" if hc["ok"] else "FAIL", hc.get("error") or f"sample={hc.get('sample')!r} {hc.get('latency_s')}s")
        if args.context and hc["ok"] and m.backend == "remote_openai":
            # Ollama-style backends can silently truncate prompts beyond num_ctx.
            filler = "lorem ipsum " * 6000  # roughly 12k tokens
            r = model.complete([{"role": "user", "content": filler + "\nReply OK."}], max_tokens=2, temperature=0)
            pt = r.prompt_tokens
            ok = r.ok and pt is not None and pt > 10000
            check(f"{m.name}: no silent context truncation at ~12k tokens", "PASS" if ok else "FAIL",
                  f"prompt_tokens reported={pt} error={r.error_class}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--context", action="store_true")
    a = ap.parse_args()
    load_dotenv()
    if not upstream_checks():
        models = []
    else:
        models = static_checks(a)
    if a.probe:
        probe_checks(models, a)
    out = RUNS_ROOT / "validation"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{time.strftime('%Y%m%dT%H%M%S')}.json"
    f.write_text(json.dumps(RESULTS, indent=2))
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r['status']=='PASS' for r in RESULTS)} pass, {n_fail} fail, "
          f"{sum(r['status']=='SKIP' for r in RESULTS)} skip -> {f}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
