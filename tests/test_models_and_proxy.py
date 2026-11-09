import json
import threading
from pathlib import Path

import requests

from conftest import free_port
from privrep.models.base import classify_http_error, make_model
from privrep.proxy import LoggingProxy


def test_classify_http_error():
    assert classify_http_error(200) == "ok"
    assert classify_http_error(400, "This model's maximum context length is 8192 tokens") == "context_overflow"
    assert classify_http_error(400, "bad field") == "invalid_request"
    assert classify_http_error(401) == "auth_error"
    assert classify_http_error(429) == "rate_limited"
    assert classify_http_error(503) == "upstream_5xx"
    assert classify_http_error(None, exc=requests.ConnectionError("Connection refused")) == "endpoint_unreachable"
    assert classify_http_error(None, exc=requests.ReadTimeout("read timed out")) == "upstream_timeout"


def test_adapter_complete_ok_and_errors(fake_model_cfg, upstream):
    m = make_model(fake_model_cfg)
    r = m.complete([{"role": "user", "content": "hi"}], temperature=0)
    assert r.ok and r.text == "echo:hi" and r.prompt_tokens == 11 and r.completion_tokens == 3
    assert upstream.requests[-1]["auth"] == "Bearer secret-upstream-key"
    upstream.mode = "context"
    assert m.complete([{"role": "user", "content": "x"}]).error_class == "context_overflow"
    upstream.mode = "500"
    assert m.complete([{"role": "user", "content": "x"}]).error_class == "upstream_5xx"
    assert m.list_models() == ["fake-model"]


def test_adapter_unreachable(fake_model_cfg):
    fake_model_cfg.base_url = f"http://127.0.0.1:{free_port()}/v1"
    r = make_model(fake_model_cfg).complete([{"role": "user", "content": "x"}])
    assert r.error_class == "endpoint_unreachable"


def _proxy(fake_model_cfg, tmp_path):
    p = LoggingProxy(make_model(fake_model_cfg), free_port(), inject_seed=True, error_log=tmp_path / "perr.jsonl")
    p.start()
    return p


def _post(url, content="ping", **extra):
    return requests.post(url + "/v1/chat/completions",
                         json={"model": "fake-model", "temperature": 0,
                               "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": content}],
                               **extra},
                         headers={"Authorization": "Bearer proxy"}, timeout=10)


def test_proxy_logs_full_io_and_injects_seed(fake_model_cfg, upstream, tmp_path):
    p = _proxy(fake_model_cfg, tmp_path)
    try:
        p.register_run("run1", tmp_path / "run1", seed=77)
        r = _post(p.base_url_for("run1", "wintermute"))
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "echo:ping"
        rec = json.loads((tmp_path / "run1" / "llm_calls.jsonl").read_text().splitlines()[0])
        assert rec["run_id"] == "run1" and rec["agent"] == "wintermute" and rec["call_seq"] == 1
        assert rec["messages"][0] == {"role": "system", "content": "SYS"} and rec["system_prompt"] == "SYS"
        assert rec["generation_params"]["temperature"] == 0 and rec["injected_params"] == {"seed": 77}
        assert json.loads(rec["raw_response"])["usage"]["prompt_tokens"] == 11
        assert rec["parsed"]["content"] == "echo:ping" and rec["usage"]["completion_tokens"] == 3
        assert rec["error_class"] == "ok" and rec["latency_s"] >= 0
        # the real key goes upstream; the agent's dummy key is never logged
        assert upstream.requests[-1]["auth"] == "Bearer secret-upstream-key"
        assert upstream.requests[-1]["body"]["seed"] == 77
        assert "secret-upstream-key" not in (tmp_path / "run1" / "llm_calls.jsonl").read_text()
    finally:
        p.stop()


def test_proxy_does_not_override_agent_seed(fake_model_cfg, upstream, tmp_path):
    p = _proxy(fake_model_cfg, tmp_path)
    try:
        p.register_run("r", tmp_path / "r", seed=5)
        _post(p.base_url_for("r", "hacksynth"), seed=999)
        rec = json.loads((tmp_path / "r" / "llm_calls.jsonl").read_text())
        assert rec["injected_params"] == {} and upstream.requests[-1]["body"]["seed"] == 999
    finally:
        p.stop()


def test_proxy_rejects_unknown_run(fake_model_cfg, upstream, tmp_path):
    p = _proxy(fake_model_cfg, tmp_path)
    try:
        r = _post(p.base_url_for("ghost", "wintermute"))
        assert r.status_code == 404 and not upstream.requests
        assert "unknown_run_id" in (tmp_path / "perr.jsonl").read_text()
    finally:
        p.stop()


def test_proxy_isolates_parallel_runs(fake_model_cfg, upstream, tmp_path):
    p = _proxy(fake_model_cfg, tmp_path)
    try:
        ids = [f"run{i}" for i in range(6)]
        for i in ids:
            p.register_run(i, tmp_path / i, seed=1)

        def go(i):
            for k in range(5):
                _post(p.base_url_for(i, "privescagent"), content=f"{i}-{k}")
        ts = [threading.Thread(target=go, args=(i,)) for i in ids]
        [t.start() for t in ts]
        [t.join() for t in ts]
        for i in ids:
            recs = [json.loads(l) for l in (tmp_path / i / "llm_calls.jsonl").read_text().splitlines()]
            assert len(recs) == 5
            assert all(r["run_id"] == i and r["messages"][-1]["content"].startswith(i + "-") for r in recs)
            assert sorted(r["call_seq"] for r in recs) == [1, 2, 3, 4, 5]
    finally:
        p.stop()


def test_proxy_upstream_down_is_logged_as_infra(fake_model_cfg, tmp_path):
    fake_model_cfg.base_url = f"http://127.0.0.1:{free_port()}/v1"
    p = _proxy(fake_model_cfg, tmp_path)
    try:
        p.register_run("r", tmp_path / "r", seed=1)
        r = _post(p.base_url_for("r", "wintermute"))
        assert r.status_code == 502
        rec = json.loads((tmp_path / "r" / "llm_calls.jsonl").read_text())
        assert rec["error_class"] == "endpoint_unreachable" and rec["messages"][-1]["content"] == "ping"
    finally:
        p.stop()


def test_proxy_context_overflow_passthrough(fake_model_cfg, upstream, tmp_path):
    p = _proxy(fake_model_cfg, tmp_path)
    try:
        upstream.mode = "context"
        p.register_run("r", tmp_path / "r", seed=1)
        r = _post(p.base_url_for("r", "wintermute"))
        assert r.status_code == 400  # the agent sees exactly what the upstream returned
        assert json.loads((tmp_path / "r" / "llm_calls.jsonl").read_text())["error_class"] == "context_overflow"
    finally:
        p.stop()


def test_hacksynth_role_labels(fake_model_cfg, upstream, tmp_path):
    from privrep.proxy import role_guess
    from evaluation.adapters.hacksynth_adapter import (PRIVESC_PLANNER_SYSTEM_PROMPT,
                                                       PRIVESC_SUMMARIZER_SYSTEM_PROMPT)
    mk = lambda s: {"messages": [{"role": "system", "content": s}]}  # noqa: E731
    assert role_guess("hacksynth", mk(PRIVESC_PLANNER_SYSTEM_PROMPT)) == "hacksynth.planner"
    assert role_guess("hacksynth", mk(PRIVESC_SUMMARIZER_SYSTEM_PROMPT)) == "hacksynth.summarizer"
    assert role_guess("wintermute", mk("other")) == "wintermute.llm"


def test_proxy_default_max_tokens_only_when_absent(fake_model_cfg, upstream, tmp_path):
    p = LoggingProxy(make_model(fake_model_cfg), free_port(), inject_seed=False, default_max_tokens=4096)
    p.start()
    try:
        p.register_run("r", tmp_path / "r", seed=1)
        _post(p.base_url_for("r", "wintermute"))                 # hackingBuddyGPT style: no max_tokens
        _post(p.base_url_for("r", "hacksynth"), max_tokens=1024)  # HackSynth style
        recs = [json.loads(l) for l in (tmp_path / "r" / "llm_calls.jsonl").read_text().splitlines()]
        assert recs[0]["injected_params"] == {"max_tokens": 4096} and upstream.requests[0]["body"]["max_tokens"] == 4096
        assert recs[1]["injected_params"] == {} and upstream.requests[1]["body"]["max_tokens"] == 1024
    finally:
        p.stop()


def test_proxy_retries_429_transparently(fake_model_cfg, upstream, tmp_path, monkeypatch):
    import privrep.proxy as PX
    monkeypatch.setattr(PX.time, "sleep", lambda s: None)
    p = LoggingProxy(make_model(fake_model_cfg), free_port(), max_429_retries=3)
    p.start()
    try:
        p.register_run("r", tmp_path / "r", seed=1)
        calls = {"n": 0}
        orig = p.upstream.post_raw
        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] <= 2:
                return 429, b"Too Many Requests", {"Retry-After": "1"}, None, 0.001
            return orig(*a, **k)
        p.upstream.post_raw = flaky
        r = _post(p.base_url_for("r", "hacksynth"))
        assert r.status_code == 200  # the agent never sees the 429
        recs = [json.loads(l) for l in (tmp_path / "r" / "llm_calls.jsonl").read_text().splitlines()]
        assert [x.get("superseded", False) for x in recs] == [True, True, False]
        assert recs[-1]["proxy_tries"] == 3 and recs[-1]["error_class"] == "ok"
        from privrep.worker import summarize_llm_calls
        s = summarize_llm_calls(tmp_path / "r" / "llm_calls.jsonl")
        assert s["n_calls"] == 1 and s["errors"] == {} and s["superseded_429"] == 2
    finally:
        p.stop()
