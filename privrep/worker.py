"""Execute ONE independent run in its own process.

    python -m privrep.worker <run_spec.json>

Each run gets a fresh Python process with a minimal environment (built by the orchestrator, so
nothing leaks from .env), a fresh adapter object, its own SQLite DB, output directory, container
name, port and seed. The agent's LLM traffic goes through the logging proxy under this run's id.
The worker never decides retries; it records what happened in run.json and exits.
"""
from __future__ import annotations

import json
import os
import random
import re
import socket
import sqlite3
import sys
import time
import traceback
from pathlib import Path

from .config import REPO_ROOT
from .models.base import DETERMINISTIC_ERROR_CLASSES, INFRA_ERROR_CLASSES
from .proxy import append_jsonl, utcnow

SSH_INFRA_PATTERNS = re.compile(
    r"(Error reading SSH protocol banner|NoValidConnectionsError|Unable to connect to port|"
    r"SSHException|Connection reset by peer|No existing session|paramiko\.ssh_exception)", re.I)

FINAL_STATUSES = {"success", "fail", "context_overflow", "infra_error", "build_error",
                  "docker_error", "harness_error"}


def wait_ssh(port: int, timeout_s: int) -> tuple[bool, str]:
    """Wait until sshd on localhost:port sends its banner (fixes 'SSH not ready' aborts)."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=3) as s:
                s.settimeout(3)
                banner = s.recv(64)
                if banner.startswith(b"SSH-"):
                    return True, banner.decode(errors="replace").strip()
                last = f"unexpected banner {banner!r}"
        except OSError as e:
            last = repr(e)
        time.sleep(1)
    return False, last


def export_hbgpt_sqlite(db: Path, out: Path) -> int:
    """Dump the per-run hackingBuddyGPT DB (turn sections, messages, tool calls) to JSONL."""
    if not db.exists():
        return 0
    n = 0
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        sections = [dict(r) for r in con.execute("SELECT * FROM sections ORDER BY run_id, id")]
        def step_of(run_id, mid):
            for s in sections:
                if s["run_id"] == run_id and s["from_message"] is not None and s["to_message"] is not None \
                        and s["from_message"] <= mid <= s["to_message"]:
                    return s["name"]
            return None
        for table in ("runs", "sections", "messages", "tool_calls"):
            for r in con.execute(f"SELECT * FROM {table}"):
                rec = {"record_type": f"hbgpt_{table}", **dict(r)}
                if table in ("messages", "tool_calls"):
                    mid = r["id"] if table == "messages" else r["message_id"]
                    rec["section"] = step_of(r["run_id"], mid)
                append_jsonl(out, rec)
                n += 1
    finally:
        con.close()
    return n


def summarize_llm_calls(path: Path) -> dict:
    s = {"n_calls": 0, "n_ok": 0, "errors": {}, "prompt_tokens": 0, "completion_tokens": 0,
         "latency_s_total": 0.0, "latencies": [], "tok_per_s": [], "roles": {},
         "superseded_429": 0, "proxy_backoff_s": 0.0}
    if not path.exists():
        return s
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("superseded"):
            s["superseded_429"] += 1
            s["proxy_backoff_s"] += r.get("proxy_backoff_s") or 0
            continue
        s["n_calls"] += 1
        s["roles"][r.get("role_guess", "?")] = s["roles"].get(r.get("role_guess", "?"), 0) + 1
        ec = r.get("error_class", "ok")
        if ec == "ok":
            s["n_ok"] += 1
            u = r.get("usage") or {}
            s["prompt_tokens"] += u.get("prompt_tokens") or 0
            s["completion_tokens"] += u.get("completion_tokens") or 0
            s["latencies"].append(r.get("latency_s") or 0)
            if r.get("tokens_per_s"):
                s["tok_per_s"].append(r["tokens_per_s"])
        else:
            s["errors"][ec] = s["errors"].get(ec, 0) + 1
        s["latency_s_total"] += r.get("latency_s") or 0
    return s


def decide_status(*, harness_exc: bool, docker_status: str | None, ssh_ok: bool,
                  llm: dict, adapter_result: dict | None, agent_text: str) -> tuple[str, bool, str]:
    """Return (status, recoverable, reason). Pure function (unit-tested)."""
    if harness_exc:
        return "harness_error", True, "exception in worker/adapter"
    if docker_status:
        return docker_status, docker_status == "docker_error", "container did not start"
    if not ssh_ok:
        return "infra_error", True, "ssh_not_ready"
    infra = {k: v for k, v in llm["errors"].items() if k in INFRA_ERROR_CLASSES or k == "transport_error"}
    if infra:
        return "infra_error", True, "llm_" + ",".join(sorted(infra))
    ok = bool(adapter_result and adapter_result.get("success"))
    if llm["n_calls"] == 0 and not ok:
        return "infra_error", True, "agent_made_no_llm_calls"
    if not ok and SSH_INFRA_PATTERNS.search(agent_text or "") and (
            (adapter_result or {}).get("steps", 0) <= 1 or "Traceback" in (agent_text or "")):
        return "infra_error", True, "ssh_session_failure"
    if not ok and any(k in DETERMINISTIC_ERROR_CLASSES for k in llm["errors"]):
        if "context_overflow" in llm["errors"]:
            return "context_overflow", False, "served context exceeded"
        return "fail", False, "llm invalid_request: " + ",".join(llm["errors"])
    return ("success" if ok else "fail"), False, ""


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json = run_dir / "run.json"
    events = run_dir / "events.jsonl"
    meta = {**spec, "status": "running", "ts_start": utcnow(), "pid": os.getpid()}
    run_json.write_text(json.dumps(meta, indent=2, default=str))
    append_jsonl(events, {"ts": utcnow(), "event": "worker_start", "run_id": spec["run_id"]})

    random.seed(spec["seed"])
    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.runner import ScenarioRunner, build_agent_info  # noqa: E402

    adapter_result, agent_text, tb, docker_status = None, "", None, None
    ssh_ok, ssh_info, verified_exec, verified_write = False, "", None, None
    runner = None
    try:
        scen_dir = Path(spec["scenario_dir"])
        meta_s = json.loads((scen_dir / "metadata.json").read_text())
        runner = ScenarioRunner(scen_dir, spec["port"])
        image_tag = f"privesc-eval-{meta_s['scenario_id']}".lower()
        had_image = runner._image_exists(image_tag)
        append_jsonl(events, {"ts": utcnow(), "event": "container_start", "image": image_tag, "prebuilt": had_image})
        if not runner.start():
            docker_status = "docker_error" if runner._image_exists(image_tag) else "build_error"
            raise _Abort()
        from .envinfo import docker_image_id
        meta["image_tag"], meta["image_id"] = image_tag, docker_image_id(image_tag)
        meta["container_name"] = runner.container_name
        ssh_ok, ssh_info = wait_ssh(spec["port"], spec["ssh_ready_timeout_s"])
        append_jsonl(events, {"ts": utcnow(), "event": "ssh_ready" if ssh_ok else "ssh_not_ready", "info": ssh_info})
        if not ssh_ok:
            raise _Abort()

        from . import rootcheck
        cid = os.popen(f"docker inspect -f '{{{{.Id}}}}' {runner.container_name}").read().strip()
        agent_info = build_agent_info(meta_s, spec["port"], spec["level"], {})
        uid = rootcheck.lowpriv_uid(runner.container_name, agent_info["username"])
        pre = rootcheck.file_snapshot_pre(runner.container_name)
        mon = rootcheck.RootProcessMonitor(cid, uid if uid is not None else -1, run_dir / "root_events.jsonl")
        mon.start()
        append_jsonl(events, {"ts": utcnow(), "event": "agent_start", "lowpriv_uid": uid,
                              "monitor_error": mon.error})

        adapter = _make_adapter(spec["agent"])
        t0 = time.time()
        adapter_result = adapter.run(agent_info, spec["max_steps"], timeout=spec["run_timeout_s"])
        meta["agent_duration_s"] = round(time.time() - t0, 2)
        append_jsonl(events, {"ts": utcnow(), "event": "agent_end", "orig_success": adapter_result.get("success"),
                              "steps": adapter_result.get("steps"), "adapter_error": adapter_result.get("error")})

        verified_exec = mon.stop()
        verified_write = rootcheck.file_evidence_post(runner.container_name, pre)
        verified_write["pre"] = {k: pre[k] for k in ("checksums", "uid0_accounts")}
    except _Abort:
        pass
    except Exception:  # noqa: BLE001 - recorded in full, never suppressed
        tb = traceback.format_exc()
        append_jsonl(events, {"ts": utcnow(), "event": "exception", "traceback": tb})
    finally:
        if runner is not None:
            runner.stop()
            append_jsonl(events, {"ts": utcnow(), "event": "container_removed", "name": runner.container_name})

    # ---- collect artefacts -------------------------------------------------
    out_dir = Path(os.environ["PRIVESC_OUTPUT_DIR"])
    for p in sorted(out_dir.rglob("*_interaction.log")):
        agent_text += p.read_text(errors="replace")
    if adapter_result and adapter_result.get("output_log"):  # HackSynth returns its log in-memory
        agent_text += adapter_result["output_log"]
        (run_dir / "agent_output.log").write_text(adapter_result["output_log"])
    elif agent_text:
        (run_dir / "agent_output.log").write_text(agent_text)
    n_steps_rec = export_hbgpt_sqlite(Path(os.environ["PRIVESC_LOG_DB"]), run_dir / "agent_steps.jsonl")
    llm = summarize_llm_calls(run_dir / "llm_calls.jsonl")

    status, recoverable, reason = decide_status(
        harness_exc=tb is not None, docker_status=docker_status, ssh_ok=ssh_ok,
        llm=llm, adapter_result=adapter_result, agent_text=agent_text)
    if reason == "ssh_not_ready":
        reason += ": " + ssh_info

    if adapter_result is not None:
        (run_dir / "result.json").write_text(json.dumps(adapter_result, indent=2, default=str))
    lat = sorted(llm["latencies"])
    meta.update({
        "status": status, "recoverable": recoverable, "status_reason": reason,
        "ts_end": utcnow(),
        "orig_success": bool(adapter_result and adapter_result.get("success")),
        "orig_steps": (adapter_result or {}).get("steps"),
        "adapter_error": (adapter_result or {}).get("error"),
        "agent_timeout": "TIMEOUT" in str((adapter_result or {}).get("error", "")),
        "verified_root_exec": (verified_exec or {}).get("verified_root_exec"),
        "verified_exec_detail": verified_exec,
        "verified_root_write": (verified_write or {}).get("verified_root_write"),
        "verified_write_detail": {k: v for k, v in (verified_write or {}).items() if k != "raw"},
        "llm": {k: v for k, v in llm.items() if k not in ("latencies", "tok_per_s")},
        "llm_latency_p50_s": lat[len(lat) // 2] if lat else None,
        "llm_tok_per_s_mean": (sum(llm["tok_per_s"]) / len(llm["tok_per_s"])) if llm["tok_per_s"] else None,
        "adapter_reported_tokens": {"input": (adapter_result or {}).get("input_tokens"),
                                    "output": (adapter_result or {}).get("output_tokens")},
        "hbgpt_records_exported": n_steps_rec,
        "traceback": tb,
    })
    run_json.write_text(json.dumps(meta, indent=2, default=str))
    append_jsonl(events, {"ts": utcnow(), "event": "worker_end", "status": status, "reason": reason})
    return 0


class _Abort(Exception):
    pass


def _make_adapter(agent: str):
    if agent == "wintermute":
        from evaluation.adapters.wintermute import WintermuteAdapter
        return WintermuteAdapter()
    if agent == "hacksynth":
        from evaluation.adapters.hacksynth_adapter import HackSynthAdapter
        a = HackSynthAdapter()
        if not a.available:
            raise RuntimeError("HackSynth PentestAgent failed to import")
        return a
    if agent == "privescagent":
        from evaluation.adapters.privescagent_adapter import PrivEscAgentAdapter
        return PrivEscAgentAdapter()
    raise ValueError(agent)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
