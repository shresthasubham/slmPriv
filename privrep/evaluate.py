"""Evaluation and comparison report.

    python -m privrep.evaluate <exp_dir> [<exp_dir> ...] --out reports/<name>

Research metrics are computed with the ORIGINAL `evaluation/metrics.py: compute_metrics`, unchanged,
under two success criteria:
  * orig      - the repository's own detector (adapter `success`), applied to full output (D-3)
  * verified  - the in-container process monitor (`verified_root_exec`), an added criterion (D4)
Engineering metrics (latency, tokens/s, tokens, failure rates, GPU memory) are reported separately
and are not part of the paper's metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT))
from evaluation.metrics import ScenarioResult, compute_metrics, metrics_to_dict  # noqa: E402

SCORED = {"success", "fail", "context_overflow"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def load_final(exp_dir: Path) -> list[dict]:
    """The last attempt per key; earlier attempts are kept on disk and counted as retries."""
    last, attempts = {}, Counter()
    for line in (exp_dir / "results.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        attempts[r["key"]] += 1
        last[r["key"]] = r
    out = []
    for k, r in last.items():
        r = dict(r)
        r["n_attempts"] = attempts[k]
        out.append(r)
    return out


def _milestones(exp_dir: Path, r: dict) -> dict:
    p = exp_dir / r["run_dir"] / "result.json"
    try:
        return json.loads(p.read_text()).get("milestones") or {}
    except (OSError, ValueError):
        return {}


def research_metrics(exp_dir: Path, runs: list[dict], criterion: str) -> dict:
    scored = [r for r in runs if r["status"] in SCORED]
    res = []
    for r in scored:
        llm = r.get("llm") or {}
        succ = bool(r.get("orig_success")) if criterion == "orig" else bool(r.get("verified_root_exec"))
        res.append(ScenarioResult(
            scenario_id=r["scenario_id"], success=succ, steps=r.get("orig_steps") or 0,
            scenario_type="core", difficulty=r.get("difficulty") or "medium", category=r.get("category") or "",
            info_level=0, agent_name=r["agent"], milestones=_milestones(exp_dir, r),
            input_tokens=llm.get("prompt_tokens", 0), output_tokens=llm.get("completion_tokens", 0)))
    m = metrics_to_dict(compute_metrics(res)) if res else {}
    k = sum(1 for x in res if x.success)
    by_cat = defaultdict(lambda: [0, 0])
    for x in res:
        by_cat[x.category][1] += 1
        by_cat[x.category][0] += int(x.success)
    return {"criterion": criterion, "n_scored": len(res), "n_success": k,
            "sr": round(k / len(res), 4) if res else None, "sr_wilson95": wilson(k, len(res)),
            "by_category": {c: {"success": a, "n": b} for c, (a, b) in sorted(by_cat.items())},
            "original_metrics": m}


def engineering_metrics(exp_dir: Path, runs: list[dict]) -> dict:
    lat, tps, ptok, ctok, calls, call_err = [], [], 0, 0, 0, Counter()
    for r in runs:
        p = exp_dir / r["run_dir"] / "llm_calls.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                c = json.loads(line)
            except ValueError:
                continue
            if c.get("superseded"):
                call_err["429_retried_transparently"] += 1
                continue
            calls += 1
            if c.get("error_class") == "ok":
                lat.append(c.get("latency_s") or 0)
                if c.get("tokens_per_s"):
                    tps.append(c["tokens_per_s"])
                u = c.get("usage") or {}
                ptok += u.get("prompt_tokens") or 0
                ctok += u.get("completion_tokens") or 0
            else:
                call_err[c.get("error_class")] += 1
    statuses = Counter(r["status"] for r in runs)
    total_attempts = sum(r.get("n_attempts", 1) for r in runs)
    gpu = []
    gp = exp_dir / "gpu_samples.jsonl"
    if gp.exists():
        for line in gp.read_text().splitlines():
            try:
                gpu.append(json.loads(line)["mem_used_mib"])
            except (ValueError, KeyError):
                pass
    q = lambda xs, f: round(f(xs), 4) if xs else None  # noqa: E731
    return {
        "llm_calls": calls, "llm_call_errors": dict(call_err),
        "latency_s_mean": q(lat, statistics.mean), "latency_s_median": q(lat, statistics.median),
        "latency_s_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 4) if lat else None,
        "tokens_per_s_mean": q(tps, statistics.mean), "tokens_per_s_median": q(tps, statistics.median),
        "prompt_tokens": ptok, "completion_tokens": ctok,
        "tokens_per_run_mean": round((ptok + ctok) / len(runs), 1) if runs else None,
        "final_status_counts": dict(statuses), "total_attempts_incl_retries": total_attempts,
        "infra_retry_count": total_attempts - len(runs),
        "unscored_rate": round(sum(v for k, v in statuses.items() if k not in SCORED) / len(runs), 4) if runs else None,
        "gpu_mem_used_mib_max": max(gpu) if gpu else None,
        "gpu_mem_used_mib_mean": round(statistics.mean(gpu), 1) if gpu else None,
        "gpu_note": "whole-GPU usage sampled every 2 s (includes the vLLM pre-allocated KV cache); "
                    "N/A for the remote API",
    }


def agreement(runs: list[dict]) -> dict:
    c = Counter()
    for r in runs:
        if r["status"] in SCORED:
            c[(bool(r.get("orig_success")), bool(r.get("verified_root_exec")))] += 1
    return {"orig1_ver1": c[(True, True)], "orig1_ver0": c[(True, False)],
            "orig0_ver1": c[(False, True)], "orig0_ver0": c[(False, False)]}


def per_rep_sr(runs: list[dict], criterion: str) -> dict:
    g = defaultdict(lambda: [0, 0])
    for r in runs:
        if r["status"] in SCORED:
            s = bool(r.get("orig_success")) if criterion == "orig" else bool(r.get("verified_root_exec"))
            g[r["rep"]][0] += int(s)
            g[r["rep"]][1] += 1
    return {rep: round(a / b, 4) for rep, (a, b) in sorted(g.items()) if b}


def evaluate(exp_dirs: list[Path]) -> dict:
    report = {"cells": []}
    for d in exp_dirs:
        ej = json.loads((d / "experiment.json").read_text())
        runs = load_final(d)
        by_agent = defaultdict(list)
        for r in runs:
            by_agent[r["agent"]].append(r)
        for agent, rs in sorted(by_agent.items()):
            report["cells"].append({
                "exp_dir": str(d), "experiment": ej["experiment"]["name"],
                "model_config": ej["model"]["name"], "model_role": ej["model"]["role"],
                "served_model_name": ej["model"]["served_model_name"], "backend": ej["model"]["backend"],
                "checkpoint": ej["model"].get("checkpoint"), "lora": ej["model"].get("lora"),
                "agent": agent, "n_keys": len(rs),
                "research_orig": research_metrics(d, rs, "orig"),
                "research_verified": research_metrics(d, rs, "verified"),
                "per_rep_sr_orig": per_rep_sr(rs, "orig"), "per_rep_sr_verified": per_rep_sr(rs, "verified"),
                "criterion_agreement": agreement(rs),
                "engineering": engineering_metrics(d, rs),
            })
    return report


def render_md(rep: dict) -> str:
    L = ["# Comparison report", "",
         "Research metrics use the original `evaluation/metrics.py`. SR is computed over *scored* runs",
         "(success, fail, context_overflow). Infra, build and harness errors are excluded and listed separately.", "",
         "## Research metrics (original paper metrics)", "",
         "| agent | model (role) | backend | scored | SR orig [95% CI] | SR verified [95% CI] | per-rep SR orig | efficiency score | progress rate |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in rep["cells"]:
        o, v = c["research_orig"], c["research_verified"]
        om = o.get("original_metrics") or {}
        L.append(f"| {c['agent']} | {c['model_config']} ({c['model_role']}) | {c['backend']} | {o['n_scored']} | "
                 f"{o['n_success']} = {o['sr']} {o['sr_wilson95']} | {v['n_success']} = {v['sr']} {v['sr_wilson95']} | "
                 f"{c['per_rep_sr_orig']} | {om.get('efficiency_score')} | {om.get('progress_rate')} |")
    L += ["", "## Criterion agreement (orig vs verified)", "", "| agent | model | both | orig only | verified only | neither |",
          "|---|---|---|---|---|---|"]
    for c in rep["cells"]:
        a = c["criterion_agreement"]
        L.append(f"| {c['agent']} | {c['model_config']} | {a['orig1_ver1']} | {a['orig1_ver0']} | {a['orig0_ver1']} | {a['orig0_ver0']} |")
    L += ["", "## Engineering / performance metrics (not paper metrics)", "",
          "| agent | model | LLM calls | latency mean / p95 (s) | tokens/s mean | prompt+completion tokens | tokens/run | final statuses | infra retries | GPU mem max (MiB) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in rep["cells"]:
        e = c["engineering"]
        L.append(f"| {c['agent']} | {c['model_config']} | {e['llm_calls']} | {e['latency_s_mean']} / {e['latency_s_p95']} | "
                 f"{e['tokens_per_s_mean']} | {e['prompt_tokens']}+{e['completion_tokens']} | {e['tokens_per_run_mean']} | "
                 f"{e['final_status_counts']} | {e['infra_retry_count']} | {e['gpu_mem_used_mib_max']} |")
    return "\n".join(L) + "\n"


def write_csv(rep: dict, path: Path) -> None:
    rows = []
    for c in rep["cells"]:
        for r in load_final(Path(c["exp_dir"])):
            if r["agent"] != c["agent"]:
                continue
            llm = r.get("llm") or {}
            rows.append({"model_config": c["model_config"], "agent": r["agent"], "scenario_id": r["scenario_id"],
                         "category": r.get("category"), "rep": r["rep"], "status": r["status"],
                         "status_reason": r.get("status_reason"), "orig_success": r.get("orig_success"),
                         "verified_root_exec": r.get("verified_root_exec"),
                         "verified_root_write": r.get("verified_root_write"), "steps": r.get("orig_steps"),
                         "llm_calls": llm.get("n_calls"), "prompt_tokens": llm.get("prompt_tokens"),
                         "completion_tokens": llm.get("completion_tokens"), "n_attempts": r.get("n_attempts"),
                         "run_dir": str(Path(c["exp_dir"]) / r["run_dir"])})
    if rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="privrep.evaluate")
    ap.add_argument("exp_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = evaluate([Path(d) for d in a.exp_dirs])
    (out / "comparison.json").write_text(json.dumps(rep, indent=2, default=str))
    (out / "comparison.md").write_text(render_md(rep))
    write_csv(rep, out / "per_run.csv")
    print(render_md(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
