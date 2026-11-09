"""Load and validate the replication's model and experiment configs (YAML)."""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Paths
#   PROJECT_ROOT   this repository (configs/, scripts/, runs/, secrets.env)
#   UPSTREAM_ROOT  the pinned, patched PrivEscalate checkout (agents, adapters, scenarios, metrics).
#                  Default: <project>/upstream (created by scripts/setup.sh). Override: PRIVREP_UPSTREAM.
# Secrets live in <project>/secrets.env. It is deliberately NOT named ".env": hackingBuddyGPT calls
# python-dotenv's find_dotenv(), which walks up the directory tree from its own source files, and it
# would find and load a <project>/.env, mapping its keys onto the agent's own parameters.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRETS_FILE = PROJECT_ROOT / "secrets.env"
CONFIG_ROOT = PROJECT_ROOT / "configs"
UPSTREAM_COMMIT = "cfb362d"


def _setting(name: str, default: str) -> str:
    """Environment first, then secrets.env, then the default (read at import time for paths)."""
    if os.getenv(name):
        return os.environ[name]
    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == name and v.strip():
                return v.strip().strip('"').strip("'")
    return default


def resolve_path(p: str | Path) -> Path:
    """Expand ~ and make relative paths relative to the project root."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (PROJECT_ROOT / p)


UPSTREAM_ROOT = resolve_path(_setting("PRIVREP_UPSTREAM", "upstream")).resolve()
RUNS_ROOT = resolve_path(_setting("PRIVREP_RUNS", "runs"))
SCENARIOS_ROOT = UPSTREAM_ROOT / "dataset" / "scenarios"
# short aliases used throughout the code
REPO_ROOT = UPSTREAM_ROOT
REPL_ROOT = PROJECT_ROOT


def vllm_bin() -> str:
    """vLLM executable: $VLLM_BIN, else $VLLM_VENV/bin/vllm, else <project>/.venv-vllm/bin/vllm."""
    if _setting("VLLM_BIN", ""):
        return os.path.expanduser(_setting("VLLM_BIN", ""))
    return str(resolve_path(_setting("VLLM_VENV", ".venv-vllm")) / "bin" / "vllm")


def hf_snapshot(repo: str, revision: str, must_exist: bool = True) -> Path:
    """Local Hugging Face cache path of a pinned revision."""
    home = Path(os.path.expanduser(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))))
    p = home / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if must_exist and not p.exists():
        raise ConfigError(f"checkpoint {repo}@{revision} not found at {p}; run scripts/download_models.py")
    return p

BACKENDS = {"vllm_openai", "remote_openai"}
AGENTS = {"privescagent", "wintermute", "hacksynth"}


class ConfigError(ValueError):
    """A config file is missing fields or has invalid values."""


def load_dotenv(path: Path | None = None, override: bool = False) -> None:
    """Minimal .env loader (KEY=VALUE, '#' comments). It never prints values."""
    path = path or SECRETS_FILE
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if override or k not in os.environ:
            os.environ[k] = v


@dataclass
class ModelConfig:
    name: str
    role: str
    backend: str
    served_model_name: str
    api_key_env: str
    context_size: int
    inject_seed: bool = False
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    checkpoint: dict = field(default_factory=dict)
    lora: Optional[dict] = None
    vllm: Optional[dict] = None
    notes: str = ""
    source_file: str = ""

    def resolved_base_url(self) -> str:
        url = self.base_url or (os.getenv(self.base_url_env or "", "") if self.base_url_env else "")
        if not url:
            raise ConfigError(f"{self.name}: base URL is not set (base_url or env {self.base_url_env})")
        url = url.rstrip("/")
        return url if url.endswith("/v1") else url + "/v1"

    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "")

    def to_record(self) -> dict:
        d = asdict(self)
        d["resolved_base_url_host"] = _host_only(self)
        return d


def _host_only(m: ModelConfig) -> str:
    """Return the scheme and host of the base URL, never a key or path secret."""
    try:
        from urllib.parse import urlparse
        u = urlparse(m.resolved_base_url())
        return f"{u.scheme}://{u.hostname}{':' + str(u.port) if u.port else ''}"
    except ConfigError:
        return "<unset>"


@dataclass
class ExperimentConfig:
    name: str
    scenario_type: str
    scenarios: dict
    agents: list
    level: int
    max_steps: int
    step_timeout_s: int
    repetitions: int
    seed: int
    parallel: int
    base_port: int
    proxy_port: int
    infra_retries: int = 2
    shuffle: bool = True
    ssh_ready_timeout_s: int = 60
    default_max_tokens: Optional[int] = None
    source_file: str = ""

    @property
    def run_timeout_s(self) -> int:
        # original: adapter.run(..., timeout=MAX_STEPS * 30); the adapters use max(timeout, steps*30)
        return self.max_steps * self.step_timeout_s


def _require(d: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d or d[k] is None and k not in ("revision",)]
    if missing:
        raise ConfigError(f"{where}: missing required field(s): {', '.join(missing)}")


def load_model_config(path: str | Path) -> ModelConfig:
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    where = str(path)
    _require(d, ["name", "role", "backend", "served_model_name", "api_key_env", "context_size"], where)
    if d["backend"] not in BACKENDS:
        raise ConfigError(f"{where}: backend must be one of {sorted(BACKENDS)}, got {d['backend']!r}")
    if not (d.get("base_url") or d.get("base_url_env")):
        raise ConfigError(f"{where}: one of base_url / base_url_env is required")
    if int(d["context_size"]) <= 0:
        raise ConfigError(f"{where}: context_size must be positive")
    if d["backend"] == "vllm_openai":
        v = d.get("vllm") or {}
        _require(v, ["max_model_len", "version"], where + ":vllm")
        if int(v["max_model_len"]) != int(d["context_size"]):
            raise ConfigError(
                f"{where}: context_size ({d['context_size']}) must equal vllm.max_model_len "
                f"({v['max_model_len']}); a mismatch caused the HTTP 400 crashes in the first run"
            )
        ck = d.get("checkpoint") or {}
        _require(ck, ["repo", "revision"], where + ":checkpoint")
        if d.get("lora"):
            _require(d["lora"], ["repo", "revision", "subfolder"], where + ":lora")
    known = set(ModelConfig.__dataclass_fields__) - {"source_file"}
    extra = set(d) - known
    if extra:
        raise ConfigError(f"{where}: unknown field(s): {', '.join(sorted(extra))}")
    return ModelConfig(**d, source_file=str(path))


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    d = yaml.safe_load(path.read_text())
    where = str(path)
    _require(d, ["name", "scenario_type", "scenarios", "agents", "level", "max_steps",
                 "step_timeout_s", "repetitions", "seed", "parallel", "base_port", "proxy_port"], where)
    bad = set(d["agents"]) - AGENTS
    if bad:
        raise ConfigError(f"{where}: unknown agent(s) {sorted(bad)}")
    if d["scenarios"].get("mode") not in {"all", "list", "stratified"}:
        raise ConfigError(f"{where}: scenarios.mode must be all|list|stratified")
    for k in ("max_steps", "step_timeout_s", "repetitions", "parallel"):
        if int(d[k]) < 1:
            raise ConfigError(f"{where}: {k} must be >= 1")
    known = set(ExperimentConfig.__dataclass_fields__) - {"source_file"}
    extra = set(d) - known
    if extra:
        raise ConfigError(f"{where}: unknown field(s): {', '.join(sorted(extra))}")
    return ExperimentConfig(**d, source_file=str(path))


def load_scenarios(scenario_type: str = "core") -> list[dict]:
    root = SCENARIOS_ROOT / scenario_type
    out = []
    for d in sorted(root.iterdir()):
        mf = d / "metadata.json"
        if d.is_dir() and mf.exists():
            m = json.loads(mf.read_text())
            m["_dir"] = str(d)
            out.append(m)
    return out


def select_scenarios(exp: ExperimentConfig, all_scen: list[dict] | None = None) -> list[dict]:
    all_scen = all_scen if all_scen is not None else load_scenarios(exp.scenario_type)
    mode = exp.scenarios["mode"]
    if mode == "all":
        return all_scen
    if mode == "list":
        by_id = {s["scenario_id"]: s for s in all_scen}
        missing = [i for i in exp.scenarios["ids"] if i not in by_id]
        if missing:
            raise ConfigError(f"unknown scenario id(s): {missing}")
        return [by_id[i] for i in exp.scenarios["ids"]]
    # stratified: deterministic sample of up to k per category
    k = int(exp.scenarios.get("per_category", 4))
    rng = random.Random(int(exp.scenarios.get("sample_seed", 0)))
    groups: dict[str, list[dict]] = {}
    for s in all_scen:
        groups.setdefault(s.get("category", "?"), []).append(s)
    picked = []
    for cat in sorted(groups):
        g = sorted(groups[cat], key=lambda s: s["scenario_id"])
        picked.extend(g if len(g) <= k else rng.sample(g, k))
    return sorted(picked, key=lambda s: s["scenario_id"])


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_fingerprint(*objs: Any) -> str:
    blob = json.dumps([asdict(o) if hasattr(o, "__dataclass_fields__") else o for o in objs],
                      sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
