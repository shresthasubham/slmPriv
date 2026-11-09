"""Experiment orchestrator: one experiment × one model configuration per invocation.

    python -m privrep.orchestrator run --experiment configs/experiments/smoke.yaml \
                                       --model configs/models/qwen2.5-7b-instruct-awq.yaml

It guarantees the following:
* One orchestrator per (experiment, model), enforced by a lockfile (fixes the concurrent
  HackSynth orchestrators of the first run).
* Ports are leased by per-port lockfiles and checked free before use (fixes the port collisions).
* Every invocation writes to a NEW experiment directory. `--resume DIR` only appends. Nothing is
  ever overwritten (fixes the JSONL last-writer-wins problem).
* LLM endpoint health is checked before the run and during it (circuit breaker). Infra failures
  are re-queued as new attempts, and the failed attempts are kept (fixes outages being scored as
  agent failures).
* Run order is shuffled with the experiment seed, so an outage can't wipe out whole categories.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import asdict
from pathlib import Path

from . import envinfo
from .config import (REPL_ROOT, REPO_ROOT, RUNS_ROOT, ConfigError, config_fingerprint, file_sha256, load_dotenv,
                     load_experiment_config, load_model_config, select_scenarios)
from .models.base import make_model
from .proxy import LoggingProxy, append_jsonl, utcnow

log = logging.getLogger("privrep")
LOCKS = RUNS_ROOT / ".locks"


# ---------------------------------------------------------------------------
class Lock:
    """Non-blocking exclusive flock on a file; it is released when the process exits."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fh = open(path, "a+")

    def acquire(self) -> bool:
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fh.seek(0); self.fh.truncate(); self.fh.write(f"{os.getpid()} {utcnow()}\n"); self.fh.flush()
            return True
        except BlockingIOError:
            return False

    def release(self) -> None:
        fcntl.flock(self.fh, fcntl.LOCK_UN)


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


class PortPool:
    """Ports leased across ALL orchestrators on this host via per-port flocks."""

    def __init__(self, base: int, size: int = 200):
        self.candidates = list(range(base, base + size))
        self.held: dict[int, Lock] = {}
        self.mu = threading.Lock()

    def lease(self) -> int:
        with self.mu:
            for p in self.candidates:
                if p in self.held:
                    continue
                lk = Lock(LOCKS / "ports" / f"{p}.lock")
                if lk.acquire():
                    if port_free(p):
                        self.held[p] = lk
                        return p
                    lk.release()
            raise RuntimeError("no free port in pool")

    def release(self, p: int) -> None:
        with self.mu:
            lk = self.held.pop(p, None)
            if lk:
                lk.release()


# ---------------------------------------------------------------------------
def health_check(model) -> dict:
    """The endpoint must list the served model and answer a 1-token request."""
    t0 = time.time()
    out = {"ts": utcnow(), "ok": False}
    try:
        ids = model.list_models()
        out["models_listed"] = ids
        if model.cfg.served_model_name not in ids:
            out["error"] = f"served model {model.cfg.served_model_name!r} not listed by endpoint"
            return out
        r = model.complete([{"role": "user", "content": "Reply with the single word OK."}],
                           max_tokens=4, temperature=0)
        out.update(gen_ok=r.ok, gen_error_class=r.error_class, gen_error=r.error_message[:500],
                   sample=r.text, latency_s=round(r.latency_s, 3))
        out["ok"] = r.ok
    except Exception as e:  # noqa: BLE001 - reported
        out["error"] = repr(e)
    out["duration_s"] = round(time.time() - t0, 3)
    return out


class GpuSampler(threading.Thread):
    def __init__(self, path: Path, interval_s: float = 2.0):
        super().__init__(daemon=True, name="gpu-sampler")
        self.path, self.interval_s, self.stop_ev = path, interval_s, threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                for line in r.stdout.strip().splitlines():
                    i, used, total, util = [x.strip() for x in line.split(",")]
                    append_jsonl(self.path, {"ts": utcnow(), "gpu": int(i), "mem_used_mib": int(used),
                                             "mem_total_mib": int(total), "util_pct": int(util)})
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop_ev.wait(self.interval_s)


# ---------------------------------------------------------------------------
def worker_env(model_cfg, spec: dict, proxy_base: str) -> dict:
    """A minimal environment for the worker. Nothing from .env reaches the agents.
    hackingBuddyGPT maps env vars onto its CLI parameters, so stray vars could change behaviour."""
    keep = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "USER", "LOGNAME", "TMPDIR",
                                       "DOCKER_HOST", "DOCKER_CONFIG", "XDG_RUNTIME_DIR") if k in os.environ}
    run_dir = Path(spec["run_dir"])
    env = {
        **keep,
        "PYTHONPATH": os.pathsep.join([str(REPL_ROOT), str(REPO_ROOT)]),
        "PRIVREP_UPSTREAM": str(REPO_ROOT),
        "PYTHONHASHSEED": str(spec["seed"]),
        "PYTHONUNBUFFERED": "1",
        # model identity, as the original run_experiments._setup_adapter_env sets it
        "PRIVESC_AGENT_MODEL": model_cfg.served_model_name,
        "PRIVESC_MODEL_NAME": model_cfg.name,
        "PRIVESC_TEMPERATURE": "0.0",
        # every agent talks to the logging proxy; the real key never leaves the orchestrator
        "CPA_API_URL": proxy_base, "CPA_API_PATH": "/v1/chat/completions", "CPA_API_KEY": "proxy",
        "PRIVESC_API_URL": proxy_base + "/v1", "PRIVESC_API_KEY": "proxy", "PRIVESC_PROVIDER": "openai",
        "OPENAI_API_KEY": "proxy",
        # isolation
        "PRIVESC_LOG_DB": str(run_dir / "hbgpt.sqlite3"),
        "PRIVESC_OUTPUT_DIR": str(run_dir / "agent_output"),
        "PRIVESC_CONTAINER_NAME": f"privrep-{spec['run_id'][:16]}",
        # documented harness fixes (DEVIATIONS.md D-2, D-3)
        "PRIVESC_CONTEXT_SIZE": str(model_cfg.context_size),
        "PRIVESC_RESULT_TRUNCATE": "0",
    }
    return env


class Orchestrator:
    def __init__(self, exp, model_cfg, exp_dir: Path, args):
        self.exp, self.mcfg, self.exp_dir, self.args = exp, model_cfg, exp_dir, args
        self.results = exp_dir / "results.jsonl"
        self.res_mu = threading.Lock()
        self.ports = PortPool(exp.base_port)
        self.model = make_model(model_cfg)
        # the proxy port is leased like the SSH ports, so orchestrators for different models can run side by side
        self.proxy_ports = PortPool(exp.proxy_port, 20)
        self.proxy_port = self.proxy_ports.lease()
        self.proxy = LoggingProxy(self.model, self.proxy_port, inject_seed=model_cfg.inject_seed,
                                  error_log=exp_dir / "proxy_errors.jsonl",
                                  default_max_tokens=exp.default_max_tokens)
        self.recent = deque(maxlen=6)
        self.stop_requested = False

    # -- plan -------------------------------------------------------------
    def build_plan(self, scenarios: list[dict]) -> list[dict]:
        agents = self.args.agents.split(",") if self.args.agents else self.exp.agents
        reps = self.args.reps or self.exp.repetitions
        plan = []
        for rep in range(reps):
            for agent in agents:
                for s in scenarios:
                    plan.append({"key": f"{agent}|{s['scenario_id']}|{rep}", "agent": agent,
                                 "scenario_id": s["scenario_id"], "scenario_dir": s["_dir"],
                                 "category": s.get("category", ""), "difficulty": s.get("difficulty", ""),
                                 "rep": rep, "seed": self.exp.seed + rep})
        if self.exp.shuffle:
            random.Random(self.exp.seed).shuffle(plan)
        if self.args.limit:
            plan = plan[: self.args.limit]
        return plan

    def completed_state(self) -> tuple[set, Counter]:
        done, attempts = set(), Counter()
        if self.results.exists():
            for line in self.results.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                attempts[r["key"]] += 1
                if not r.get("recoverable") or attempts[r["key"]] > self.exp.infra_retries:
                    done.add(r["key"])
        return done, attempts

    # -- execution --------------------------------------------------------
    def run_one(self, item: dict, attempt: int) -> dict:
        run_id = uuid.uuid4().hex
        port = self.ports.lease()
        run_dir = (self.exp_dir / "runs" / item["agent"] / item["scenario_id"] /
                   f"rep{item['rep']}" / f"attempt{attempt}_{run_id[:8]}")
        run_dir.mkdir(parents=True, exist_ok=False)
        spec = {**item, "run_id": run_id, "attempt": attempt, "run_dir": str(run_dir), "port": port,
                "experiment": self.exp.name, "experiment_dir": str(self.exp_dir),
                "model_config": self.mcfg.name, "served_model_name": self.mcfg.served_model_name,
                "backend": self.mcfg.backend, "level": self.exp.level, "max_steps": self.exp.max_steps,
                "run_timeout_s": self.exp.run_timeout_s, "ssh_ready_timeout_s": self.exp.ssh_ready_timeout_s,
                "context_size": self.mcfg.context_size, "inject_seed": self.mcfg.inject_seed}
        spec_path = run_dir / "run_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        self.proxy.register_run(run_id, run_dir, item["seed"])
        env = worker_env(self.mcfg, spec, self.proxy.base_url_for(run_id, item["agent"]))
        hard_timeout = self.exp.run_timeout_s + 600  # container build/start plus verification headroom
        t0 = time.time()
        try:
            with open(run_dir / "worker.log", "w") as wl:
                p = subprocess.run([sys.executable, "-m", "privrep.worker", str(spec_path)], cwd=run_dir,
                                   env=env, stdout=wl, stderr=subprocess.STDOUT, timeout=hard_timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
            subprocess.run(["docker", "rm", "-f", env["PRIVESC_CONTAINER_NAME"]], capture_output=True)
        finally:
            self.proxy.unregister_run(run_id)
            self.ports.release(port)
        try:
            meta = json.loads((run_dir / "run.json").read_text())
        except (OSError, ValueError):
            meta = {**spec, "status": "running"}
        if meta.get("status") not in ("success", "fail", "context_overflow", "infra_error", "build_error",
                                      "docker_error", "harness_error"):
            meta.update(status="harness_error", recoverable=True,
                        status_reason=f"worker exited rc={rc} without a final status")
            (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str))
        meta["worker_rc"] = rc
        meta["wall_s"] = round(time.time() - t0, 2)
        rec = {k: meta.get(k) for k in (
            "key", "run_id", "attempt", "agent", "scenario_id", "category", "difficulty", "rep", "seed",
            "model_config", "served_model_name", "backend", "status", "recoverable", "status_reason",
            "orig_success", "orig_steps", "agent_timeout", "verified_root_exec", "verified_root_write",
            "llm", "llm_latency_p50_s", "llm_tok_per_s_mean", "adapter_reported_tokens", "image_id",
            "ts_start", "ts_end", "wall_s", "worker_rc")}
        rec["run_dir"] = str(run_dir.relative_to(self.exp_dir))
        with self.res_mu:
            append_jsonl(self.results, rec)
        return rec

    def breaker(self, rec: dict) -> None:
        """Circuit breaker: after 3 consecutive LLM-infra failures, wait for the endpoint to recover."""
        self.recent.append(rec)
        tail = list(self.recent)[-3:]
        if len(tail) == 3 and all(r["status"] == "infra_error" and str(r.get("status_reason", "")).startswith("llm_")
                                  for r in tail):
            log.warning("3 consecutive LLM infra failures: pausing for endpoint health")
            append_jsonl(self.exp_dir / "events.jsonl", {"ts": utcnow(), "event": "breaker_open"})
            for _ in range(30):
                hc = health_check(self.model)
                append_jsonl(self.exp_dir / "health.jsonl", hc)
                if hc["ok"]:
                    append_jsonl(self.exp_dir / "events.jsonl", {"ts": utcnow(), "event": "breaker_closed"})
                    self.recent.clear()
                    return
                time.sleep(60)
            log.error("endpoint still down after 30 min: stopping (remaining runs stay pending; use --resume)")
            self.stop_requested = True

    def execute(self, plan: list[dict]) -> Counter:
        done, attempts = self.completed_state()
        queue = deque((it, attempts[it["key"]]) for it in plan if it["key"] not in done)
        log.info("plan: %d runs, %d already final, %d to execute", len(plan), len(done), len(queue))
        stats = Counter()
        with ThreadPoolExecutor(max_workers=self.args.parallel or self.exp.parallel) as pool:
            futs = {}
            while (queue or futs) and not self.stop_requested:
                while queue and len(futs) < (self.args.parallel or self.exp.parallel) and not self.stop_requested:
                    item, att = queue.popleft()
                    futs[pool.submit(self.run_one, item, att)] = (item, att)
                if not futs:
                    break
                fin, _ = wait(list(futs), return_when=FIRST_COMPLETED)
                for f in fin:
                    item, att = futs.pop(f)
                    try:
                        rec = f.result()
                    except Exception as e:  # noqa: BLE001 - orchestration error, logged, run re-queued
                        log.exception("orchestration error for %s", item["key"])
                        append_jsonl(self.exp_dir / "events.jsonl", {"ts": utcnow(), "event": "orchestration_error",
                                                                     "key": item["key"], "error": repr(e)})
                        stats["orchestration_error"] += 1
                        continue
                    stats[rec["status"]] += 1
                    log.info("[%s] %-12s %-40s %s %s", rec["status"], item["agent"], item["scenario_id"],
                             f"rep{item['rep']}/att{att}", rec.get("status_reason") or "")
                    if rec.get("recoverable") and att < self.exp.infra_retries:
                        queue.append((item, att + 1))
                    self.breaker(rec)
        return stats


def new_exp_dir(exp, mcfg) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = RUNS_ROOT / exp.name / mcfg.name / f"{ts}_{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def cmd_run(args) -> int:
    load_dotenv()
    exp = load_experiment_config(args.experiment)
    mcfg = load_model_config(args.model)
    lock = Lock(LOCKS / f"{exp.name}__{mcfg.name}.lock")
    if not lock.acquire():
        print(f"ERROR: another orchestrator holds {lock.path}. Refusing to run concurrently.", file=sys.stderr)
        return 2
    exp_dir = Path(args.resume) if args.resume else new_exp_dir(exp, mcfg)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(exp_dir / "orchestrator.log")])
    scenarios = select_scenarios(exp)
    orch = Orchestrator(exp, mcfg, exp_dir, args)
    plan = orch.build_plan(scenarios)
    if args.dry_run:
        print(json.dumps({"exp_dir": str(exp_dir), "n_runs": len(plan), "first": plan[:3]}, indent=2))
        shutil.rmtree(exp_dir) if not args.resume else None
        return 0

    hc = health_check(orch.model)
    append_jsonl(exp_dir / "health.jsonl", hc)
    if not hc["ok"]:
        print(f"ERROR: endpoint health check failed: {hc.get('error') or hc.get('gen_error_class')}", file=sys.stderr)
        return 3
    if not args.resume:
        cfg_dir = exp_dir / "config"
        cfg_dir.mkdir()
        shutil.copy(args.experiment, cfg_dir / Path(args.experiment).name)
        shutil.copy(args.model, cfg_dir / Path(args.model).name)
        envinfo.write(exp_dir / "environment.json", envinfo.collect({
            "vllm_server_version": envinfo.vllm_server_version(mcfg.resolved_base_url())
            if mcfg.backend == "vllm_openai" else None}))
        (exp_dir / "experiment.json").write_text(json.dumps({
            "experiment": asdict(exp), "model": mcfg.to_record(),
            "config_sha256": {Path(args.experiment).name: file_sha256(Path(args.experiment)),
                              Path(args.model).name: file_sha256(Path(args.model))},
            "fingerprint": config_fingerprint(exp, mcfg), "created": utcnow(),
            "cli": vars(args), "plan": plan, "health_check": hc, "proxy_port": orch.proxy_port},
            indent=2, default=str))
    append_jsonl(exp_dir / "events.jsonl", {"ts": utcnow(), "event": "orchestrator_start", "pid": os.getpid(),
                                            "resume": bool(args.resume)})
    try:
        orch.proxy.start()
    except OSError as e:
        print(f"ERROR: proxy port {orch.proxy_port} unavailable: {e}", file=sys.stderr)
        return 4
    gpu = GpuSampler(exp_dir / "gpu_samples.jsonl")
    gpu.start()
    try:
        stats = orch.execute(plan)
    finally:
        gpu.stop_ev.set()
        orch.proxy.stop()
        append_jsonl(exp_dir / "events.jsonl", {"ts": utcnow(), "event": "orchestrator_end"})
    print(json.dumps({"exp_dir": str(exp_dir), "status_counts": stats,
                      "stopped_early": orch.stop_requested}, indent=2))
    return 1 if orch.stop_requested else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="privrep.orchestrator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one experiment config with one model config")
    r.add_argument("--experiment", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--agents", help="comma-separated subset of the experiment's agents")
    r.add_argument("--reps", type=int, help="override the number of repetitions")
    r.add_argument("--parallel", type=int, help="override parallelism")
    r.add_argument("--limit", type=int, help="run only the first N planned runs (debugging)")
    r.add_argument("--resume", help="existing experiment dir to continue (append-only)")
    r.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        return cmd_run(args)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
