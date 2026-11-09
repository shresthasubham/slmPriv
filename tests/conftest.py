import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from privrep.config import UPSTREAM_ROOT  # noqa: E402

if not (UPSTREAM_ROOT / "evaluation" / "metrics.py").exists():
    pytest.exit(f"upstream checkout not found at {UPSTREAM_ROOT}: run scripts/setup.sh first "
                "(or set PRIVREP_UPSTREAM)", returncode=2)
sys.path.insert(1, str(UPSTREAM_ROOT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeUpstream:
    """A minimal OpenAI-compatible server. `mode` selects the response behaviour."""

    def __init__(self):
        self.port = free_port()
        self.mode = "ok"
        self.requests = []
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self._send(200, {"data": [{"id": "fake-model"}]})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                up.requests.append({"body": body, "auth": self.headers.get("Authorization")})
                if up.mode == "context":
                    return self._send(400, {"error": {"message": "This model's maximum context length is 8192 tokens."}})
                if up.mode == "500":
                    return self._send(500, {"error": {"message": "boom"}})
                last = body["messages"][-1]["content"]
                self._send(200, {"model": body.get("model"), "choices": [
                    {"message": {"role": "assistant", "content": f"echo:{last}"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 3}})

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream():
    u = FakeUpstream()
    yield u
    u.close()


@pytest.fixture
def fake_model_cfg(upstream, monkeypatch):
    from privrep.config import ModelConfig
    monkeypatch.setenv("FAKE_KEY", "secret-upstream-key")
    return ModelConfig(name="fake", role="test", backend="vllm_openai", served_model_name="fake-model",
                       api_key_env="FAKE_KEY", context_size=8192, inject_seed=True, base_url=upstream.base_url,
                       vllm={"max_model_len": 8192, "version": "x"}, checkpoint={"repo": "x", "revision": "y"})
