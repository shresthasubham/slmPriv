#!/usr/bin/env python3
"""Optional: pre-build the scenario Docker images an experiment needs.

    .venv/bin/python scripts/build_images.py configs/experiments/full.yaml --jobs 4

The orchestrator builds a missing image on first use anyway (upstream evaluation/runner.py), with the
same tag and command used here: `docker build -t privesc-eval-<scenario_id> -f <dir>/Dockerfile <dir>`.
Pre-building moves that cost (and any build failures) out of the timed experiment. A scenario whose
image cannot be built is reported as `build_error` in the experiment and excluded from success rates.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from privrep.config import load_experiment_config, select_scenarios  # noqa: E402


def exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def build(s: dict) -> tuple[str, str]:
    tag = f"privesc-eval-{s['scenario_id']}".lower()
    if exists(tag):
        return s["scenario_id"], "exists"
    d = Path(s["_dir"])
    r = subprocess.run(["docker", "build", "-t", tag, "-f", str(d / "Dockerfile"), str(d)],
                       capture_output=True, text=True, timeout=1800)
    return s["scenario_id"], "built" if r.returncode == 0 else "FAILED: " + r.stderr.strip()[-300:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    scen = select_scenarios(load_experiment_config(a.experiment))
    failed = []
    with ThreadPoolExecutor(a.jobs) as ex:
        for i, (sid, st) in enumerate(ex.map(build, scen), 1):
            print(f"[{i}/{len(scen)}] {sid}: {st}", flush=True)
            if st.startswith("FAILED"):
                failed.append(sid)
    print(f"\n{len(scen) - len(failed)}/{len(scen)} images ready; failed: {failed or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
