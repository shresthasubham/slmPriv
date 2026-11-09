"""Capture environment and provenance information for every experiment."""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT, UPSTREAM_ROOT


def _run(cmd: list[str], timeout: int = 20, cwd: Path = UPSTREAM_ROOT) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_info() -> dict:
    """Upstream checkout: commit + the exact diff applied to it (should equal patches/*.patch).
    Project: commit of this harness repository, plus uncommitted paths (if any)."""
    status = _run(["git", "status", "--porcelain"]) or ""
    diff = _run(["git", "diff", "HEAD", "--", "evaluation", "privescagent", "baselines", "scripts"]) or ""
    proj_status = _run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT) or ""
    return {"upstream_path": str(UPSTREAM_ROOT), "commit": _run(["git", "rev-parse", "HEAD"]),
            "dirty_paths": [l for l in status.splitlines() if ".egg-info" not in l],
            "tracked_code_diff_vs_head": diff,
            "project_path": str(PROJECT_ROOT),
            "project_commit": _run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT),
            "project_dirty_paths": [l for l in proj_status.splitlines() if not l[3:].startswith("runs/")]}


def gpu_info() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid", "--format=csv,noheader"])
    if not out:
        return []
    rows = []
    for line in out.splitlines():
        name, mem, drv, uuid = [x.strip() for x in line.split(",")]
        rows.append({"name": name, "memory_total": mem, "driver": drv, "uuid": uuid})
    return rows


def vllm_server_version(base_url_v1: str) -> Optional[str]:
    import requests
    root = base_url_v1[:-3] if base_url_v1.endswith("/v1") else base_url_v1
    try:
        r = requests.get(root + "/version", timeout=10)
        return r.json().get("version") if r.ok else None
    except (requests.RequestException, ValueError):
        return None


def collect(extra: Optional[dict] = None) -> dict:
    py = sys.executable
    freeze = _run(["uv", "pip", "freeze", "--python", py]) or _run([py, "-m", "pip", "freeze"]) or ""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": py,
        "docker": _run(["docker", "version", "--format", "{{.Server.Version}}"]),
        "gpus": gpu_info(),
        "cuda_driver_visible": bool(gpu_info()),
        "git": git_info(),
        "pip_freeze": freeze.splitlines(),
        "env_flags": {k: os.environ.get(k) for k in ("PYTHONHASHSEED",) if k in os.environ},
    }
    if extra:
        info.update(extra)
    return info


def docker_image_id(tag: str) -> Optional[str]:
    return _run(["docker", "image", "inspect", "-f", "{{.Id}}", tag])


def write(path: Path, info: dict) -> None:
    path.write_text(json.dumps(info, indent=2, default=str))
