"""A transparent, logging, OpenAI-compatible reverse proxy.

Every agent sends its LLM traffic to http://127.0.0.1:<port>/r/<run_id>/<agent>/v1/...
The proxy forwards the request to the configured backend (vLLM or the remote API) and appends
one JSONL record per call to <run_dir>/llm_calls.jsonl, holding the full request, raw response,
parameters, tokens, latency and errors.

Why a proxy rather than editing the agents: it captures the input and output of all three agents
without touching their prompts, parsers or control flow. The only change to a request is the
optional `seed` injection, which is recorded in each record. Attribution goes by the run id in
the URL, so parallel runs cannot be mixed up (fixing the shared-SQLite race of the first run).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from .models.base import OpenAICompatChatModel, classify_http_error

log = logging.getLogger("privrep.proxy")

# HackSynth system-prompt prefixes (evaluation/adapters/hacksynth_adapter.py). Used only to label records.
_ROLE_PREFIXES = {
    "You are a penetration testing command generator": "hacksynth.planner",
    "You are an autonomous penetration testing agent summarizing": "hacksynth.summarizer",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append_jsonl(path: Path, rec: dict) -> None:
    """Append one record and fsync it. Never truncates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _system_prompt(body: dict) -> str:
    for m in body.get("messages") or []:
        if m.get("role") == "system":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def role_guess(agent: str, body: dict) -> str:
    sp = _system_prompt(body)
    for prefix, role in _ROLE_PREFIXES.items():
        if sp.startswith(prefix):
            return role
    return f"{agent}.llm"


class _RunInfo:
    def __init__(self, run_dir: Path, seed: Optional[int]):
        self.run_dir = run_dir
        self.seed = seed
        self.seq = 0
        self.lock = threading.Lock()

    def next_seq(self) -> int:
        with self.lock:
            self.seq += 1
            return self.seq


class LoggingProxy:
    def __init__(self, upstream: OpenAICompatChatModel, port: int, host: str = "127.0.0.1",
                 inject_seed: bool = False, error_log: Optional[Path] = None,
                 upstream_timeout_s: float = 600.0, default_max_tokens: Optional[int] = None,
                 max_429_retries: int = 8):
        self.upstream = upstream
        self.host, self.port = host, port
        self.inject_seed = inject_seed
        self.error_log = error_log
        self.upstream_timeout_s = upstream_timeout_s
        self.default_max_tokens = default_max_tokens
        self.max_429_retries = max_429_retries
        self._runs: dict[str, _RunInfo] = {}
        self._runs_lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- registry ---------------------------------------------------------
    def register_run(self, run_id: str, run_dir: Path, seed: Optional[int]) -> None:
        with self._runs_lock:
            self._runs[run_id] = _RunInfo(Path(run_dir), seed)

    def unregister_run(self, run_id: str) -> None:
        with self._runs_lock:
            self._runs.pop(run_id, None)

    def _get_run(self, run_id: str) -> Optional[_RunInfo]:
        with self._runs_lock:
            return self._runs.get(run_id)

    def base_url_for(self, run_id: str, agent: str) -> str:
        """Base URL with no /v1 suffix (hackingBuddyGPT appends /v1/chat/completions)."""
        return f"http://{self.host}:{self.port}/r/{run_id}/{agent}"

    # -- server -----------------------------------------------------------
    def start(self) -> None:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # quiet the default stderr access log
                log.debug("proxy: " + fmt, *args)

            def _send(self, status: int, body: bytes, ctype: str = "application/json"):
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError) as e:
                    # The agent process went away (e.g. killed at the run timeout) while the upstream
                    # call was in flight. The call itself is already in llm_calls.jsonl.
                    proxy._log_proxy_error("client_disconnected", f"{self.path}: {e!r}",
                                           (self._route() or ("",))[0])
                    self.close_connection = True

            def _route(self):
                parts = self.path.split("?", 1)[0].strip("/").split("/")
                # r/<run_id>/<agent>/v1/<rest...>
                if len(parts) < 4 or parts[0] != "r" or parts[3] != "v1":
                    return None
                return parts[1], parts[2], "/".join(parts[4:])

            def do_GET(self):
                route = self._route()
                if not route:
                    return self._send(404, b'{"error":{"message":"bad proxy path"}}')
                if route[2] == "models":
                    try:
                        ids = proxy.upstream.list_models()
                        return self._send(200, json.dumps({"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}).encode())
                    except Exception as e:  # noqa: BLE001 - reported to the caller and logged
                        proxy._log_proxy_error("list_models_failed", repr(e), route[0])
                        return self._send(502, json.dumps({"error": {"message": f"proxy upstream error: {e}"}}).encode())
                return self._send(404, b'{"error":{"message":"unsupported GET"}}')

            def do_POST(self):
                route = self._route()
                n = int(self.headers.get("Content-Length") or 0)
                raw_req = self.rfile.read(n) if n else b""
                if not route:
                    return self._send(404, b'{"error":{"message":"bad proxy path"}}')
                run_id, agent, rest = route
                info = proxy._get_run(run_id)
                if info is None:
                    proxy._log_proxy_error("unknown_run_id", f"POST for unregistered run {run_id}", run_id)
                    return self._send(404, json.dumps({"error": {"message": f"unknown run id {run_id}"}}).encode())
                status, body, ctype = proxy._forward(info, run_id, agent, rest, raw_req)
                self._send(status, body, ctype)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="llm-proxy", daemon=True)
        self._thread.start()
        log.info("logging proxy listening on %s:%d -> %s", self.host, self.port, self.upstream.base_url)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    # -- core -------------------------------------------------------------
    def _log_proxy_error(self, kind: str, msg: str, run_id: str = "") -> None:
        if self.error_log:
            append_jsonl(self.error_log, {"ts": utcnow(), "kind": kind, "run_id": run_id, "message": msg})

    def _forward(self, info: _RunInfo, run_id: str, agent: str, rest: str, raw_req: bytes) -> tuple[int, bytes, str]:
        seq = info.next_seq()
        ts_start = utcnow()
        rec: dict = {
            "record_type": "llm_call", "run_id": run_id, "agent": agent, "call_seq": seq,
            "ts_start": ts_start, "endpoint": rest, "backend": self.upstream.backend_id,
            "model_config": self.upstream.cfg.name, "upstream_host": self.upstream.base_url.split("/v1")[0],
        }
        injected = {}
        try:
            body = json.loads(raw_req.decode("utf-8")) if raw_req else {}
            parse_err = None
        except (UnicodeDecodeError, ValueError) as e:
            body, parse_err = None, repr(e)
        if body is None:
            rec.update(request_raw=raw_req.decode("utf-8", "replace"), request_parse_error=parse_err)
            fwd = raw_req
        else:
            if self.inject_seed and info.seed is not None and "seed" not in body and rest.endswith("completions"):
                body["seed"] = info.seed
                injected["seed"] = info.seed
            if (self.default_max_tokens and rest.endswith("completions") and "max_tokens" not in body
                    and "max_completion_tokens" not in body):
                body["max_tokens"] = self.default_max_tokens
                injected["max_tokens"] = self.default_max_tokens
            fwd = json.dumps(body).encode() if injected else raw_req
            sp = _system_prompt(body)
            rec.update(
                model_requested=body.get("model"),
                messages=body.get("messages"),
                system_prompt=sp,
                system_prompt_sha256=hashlib.sha256(sp.encode()).hexdigest()[:16],
                n_messages=len(body.get("messages") or []),
                generation_params={k: v for k, v in body.items() if k not in ("messages",)},
                role_guess=role_guess(agent, body),
            )
        rec["injected_params"] = injected

        # Transparent retry ONLY for HTTP 429 (provider quota). The agent never sees the 429, so its
        # behaviour is unchanged. Every refused try is logged as a separate `superseded` record.
        tries = 0
        while True:
            status, resp, headers, exc, latency = self.upstream.post_raw(rest, fwd, timeout_s=self.upstream_timeout_s)
            resp_text = resp.decode("utf-8", "replace")
            ec = classify_http_error(status, resp_text, exc)
            if ec != "rate_limited" or tries >= self.max_429_retries:
                break
            tries += 1
            try:
                wait_s = min(120.0, float(headers.get("Retry-After", "")))
            except ValueError:
                wait_s = min(60.0, 5.0 * 2 ** (tries - 1))
            append_jsonl(info.run_dir / "llm_calls.jsonl", {
                **rec, "ts_end": utcnow(), "status_code": status, "error_class": ec, "superseded": True,
                "proxy_try": tries, "proxy_backoff_s": wait_s, "error_message": resp_text[:500],
                "latency_s": round(latency, 4)})
            time.sleep(wait_s)
        rec["proxy_tries"] = tries + 1
        rec.update(ts_end=utcnow(), latency_s=round(latency, 4), status_code=status,
                   raw_response=resp_text, error_class=ec,
                   error_message=(repr(exc) if exc else (resp_text[:4000] if ec != "ok" else "")))
        if ec == "ok":
            try:
                r = json.loads(resp_text)
                ch = (r.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                u = r.get("usage") or {}
                rec["parsed"] = {"content": msg.get("content"), "tool_calls": msg.get("tool_calls"),
                                 "reasoning_content": msg.get("reasoning_content"),
                                 "finish_reason": ch.get("finish_reason"), "served_model": r.get("model")}
                rec["usage"] = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
                ct = u.get("completion_tokens")
                rec["tokens_per_s"] = round(ct / latency, 3) if ct and latency > 0 else None
            except (ValueError, AttributeError, IndexError, TypeError) as e:
                rec["error_class"] = "bad_gateway_response"
                rec["error_message"] = f"unparseable upstream response: {e!r}"
        append_jsonl(info.run_dir / "llm_calls.jsonl", rec)

        if exc is not None:
            err = {"error": {"message": f"proxy: upstream unreachable: {exc}", "type": "proxy_upstream_error",
                             "error_class": ec}}
            return 502, json.dumps(err).encode(), "application/json"
        return status or 502, resp, headers.get("Content-Type", "application/json")
