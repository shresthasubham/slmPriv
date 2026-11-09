import json
import subprocess
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest

from privrep import orchestrator as O
from privrep.config import CONFIG_ROOT, load_experiment_config, load_model_config, select_scenarios
from privrep.evaluate import evaluate, load_final, wilson


@pytest.fixture(autouse=True)
def isolated_locks(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "LOCKS", tmp_path / "locks")


def test_lock_is_exclusive_across_processes(tmp_path):
    lk = O.Lock(tmp_path / "x.lock")
    assert lk.acquire()
    code = ("import fcntl,sys; f=open(sys.argv[1],'a+');\n"
            "try:\n fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB); print('GOT')\n"
            "except BlockingIOError: print('BLOCKED')")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "x.lock")], capture_output=True, text=True).stdout
    assert out.strip() == "BLOCKED"
    lk.release()
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "x.lock")], capture_output=True, text=True).stdout
    assert out.strip() == "GOT"


def test_two_port_pools_never_share_a_port():
    a, b = O.PortPool(46100, 6), O.PortPool(46100, 6)  # two orchestrators, same range
    got = [a.lease(), b.lease(), a.lease(), b.lease()]
    assert len(set(got)) == 4
    for p in got[::2]:
        a.release(p)
    for p in got[1::2]:
        b.release(p)


def test_port_in_use_is_skipped():
    import socket
    s = socket.socket(); s.bind(("0.0.0.0", 46200)); s.listen()
    try:
        assert O.PortPool(46200, 3).lease() != 46200
    finally:
        s.close()


def _orch(tmp_path, **kw):
    exp = load_experiment_config(CONFIG_ROOT / "experiments/smoke.yaml")
    m = load_model_config(CONFIG_ROOT / "models/qwen2.5-7b-instruct-awq.yaml")
    args = Namespace(agents=None, reps=kw.get("reps"), limit=None, parallel=None)
    return O.Orchestrator(exp, m, tmp_path, args), exp, m


def test_plan_is_deterministic_shuffled_and_complete(tmp_path):
    o, exp, _ = _orch(tmp_path, reps=2)
    p1 = o.build_plan(select_scenarios(exp))
    p2 = o.build_plan(select_scenarios(exp))
    assert [x["key"] for x in p1] == [x["key"] for x in p2]
    assert len(p1) == 2 * 3 * 2 and len({x["key"] for x in p1}) == len(p1)
    assert {x["seed"] for x in p1} == {exp.seed, exp.seed + 1}


def test_completed_state_retries(tmp_path):
    o, exp, _ = _orch(tmp_path)
    res = tmp_path / "results.jsonl"
    lines = [{"key": "a", "recoverable": True}, {"key": "b", "recoverable": False},
             {"key": "c", "recoverable": True}, {"key": "c", "recoverable": True}, {"key": "c", "recoverable": True}]
    res.write_text("".join(json.dumps(l) + "\n" for l in lines))
    done, att = o.completed_state()
    assert done == {"b", "c"} and att["a"] == 1 and att["c"] == 3


def test_worker_env_is_minimal_and_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("OTHER_PROVIDER_API_KEY", "super-secret")
    monkeypatch.setenv("LLM_MODEL", "should-not-leak")   # hackingBuddyGPT would read this
    m = load_model_config(CONFIG_ROOT / "models/qwen2.5-7b-instruct-awq.yaml")
    envs = []
    for rid in ("aaaa1111bbbb2222cccc", "dddd3333eeee4444ffff"):
        spec = {"run_dir": str(tmp_path / rid), "run_id": rid, "seed": 3}
        envs.append(O.worker_env(m, spec, f"http://127.0.0.1:1/r/{rid}/wintermute"))
    for e in envs:
        assert "super-secret" not in json.dumps(e) and "LLM_MODEL" not in e
        assert e["PRIVESC_CONTEXT_SIZE"] == "32768" and e["PRIVESC_RESULT_TRUNCATE"] == "0"
    for k in ("PRIVESC_LOG_DB", "PRIVESC_OUTPUT_DIR", "PRIVESC_CONTAINER_NAME", "CPA_API_URL"):
        assert envs[0][k] != envs[1][k]


def _fixture_exp(tmp_path):
    d = tmp_path / "exp"
    (d / "runs").mkdir(parents=True)
    (d / "experiment.json").write_text(json.dumps({
        "experiment": {"name": "t"}, "model": {"name": "m", "role": "model_a", "served_model_name": "m",
                                               "backend": "vllm_openai"}}))
    recs = []
    def add(key, sc, status, orig, ver, rec=False, cat="A1_suid_sgid"):
        rd = f"runs/{key.replace('|', '_')}_{len(recs)}"
        (d / rd).mkdir()
        (d / rd / "llm_calls.jsonl").write_text(json.dumps(
            {"error_class": "ok", "latency_s": 0.5, "tokens_per_s": 40.0,
             "usage": {"prompt_tokens": 100, "completion_tokens": 20}}) + "\n")
        (d / rd / "result.json").write_text(json.dumps({"milestones": {"enumerated": True, "exploited": False}}))
        recs.append({"key": key, "agent": "wintermute", "scenario_id": sc, "category": cat, "difficulty": "easy",
                     "rep": 0, "status": status, "recoverable": rec, "orig_success": orig, "orig_steps": 4,
                     "verified_root_exec": ver, "llm": {"n_calls": 1, "prompt_tokens": 100, "completion_tokens": 20},
                     "run_dir": rd})
    add("wintermute|s1|0", "s1", "infra_error", False, False, rec=True)   # retried...
    add("wintermute|s1|0", "s1", "success", True, True)                   # ...final attempt succeeds
    add("wintermute|s2|0", "s2", "fail", False, False)
    add("wintermute|s3|0", "s3", "success", True, False)                  # criteria disagree
    add("wintermute|s4|0", "s4", "build_error", False, False)             # excluded
    add("wintermute|s5|0", "s5", "context_overflow", False, False, cat="A2_sudo")
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d


def test_evaluate_uses_final_attempt_and_excludes_unscored(tmp_path):
    d = _fixture_exp(tmp_path)
    fin = {r["key"]: r for r in load_final(d)}
    assert fin["wintermute|s1|0"]["status"] == "success" and fin["wintermute|s1|0"]["n_attempts"] == 2
    cell = evaluate([d])["cells"][0]
    o, v = cell["research_orig"], cell["research_verified"]
    assert o["n_scored"] == 4 and o["n_success"] == 2 and o["sr"] == 0.5
    assert v["n_success"] == 1
    assert o["original_metrics"]["total_scenarios"] == 4          # computed by the original compute_metrics
    assert o["by_category"]["A2_sudo"] == {"success": 0, "n": 1}
    assert cell["criterion_agreement"] == {"orig1_ver1": 1, "orig1_ver0": 1, "orig0_ver1": 0, "orig0_ver0": 2}
    e = cell["engineering"]
    assert e["infra_retry_count"] == 1 and e["final_status_counts"]["build_error"] == 1
    assert e["latency_s_mean"] == 0.5 and e["tokens_per_s_mean"] == 40.0


def test_wilson():
    lo, hi = wilson(80, 491)
    assert 0.13 < lo < 0.14 and 0.19 < hi < 0.20
    assert wilson(0, 0) == (0.0, 0.0)
