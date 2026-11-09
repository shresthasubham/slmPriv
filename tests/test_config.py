from pathlib import Path

import pytest
import yaml

from privrep.config import (CONFIG_ROOT, ConfigError, load_experiment_config, load_model_config,
                            load_scenarios, select_scenarios)

MODELS = sorted((CONFIG_ROOT / "models").glob("*.yaml"))
EXPS = sorted((CONFIG_ROOT / "experiments").glob("*.yaml"))


def test_model_configs_present_and_valid():
    roles = {load_model_config(p).role for p in MODELS}
    assert roles == {"model_a", "model_b"}


def test_merged_model_config_points_at_pinned_release():
    m = load_model_config(CONFIG_ROOT / "models/privesc-llm-4b-rl.yaml")
    mf = m.checkpoint["merged_from"]
    assert mf["lora_repo"] == "sailab-vienna/privesc-llm-4b" and mf["lora_subfolder"] == "rl_adapter"
    assert len(mf["base_revision"]) == 40 and len(mf["lora_revision"]) == 40
    assert m.lora is None  # served merged, never through vLLM's LoRA path (docs/DEVIATIONS.md M-2b)


@pytest.mark.parametrize("p", EXPS, ids=lambda p: p.name)
def test_experiment_configs_valid_and_faithful(p):
    e = load_experiment_config(p)
    # Original methodology values (scripts/run_experiments.py @ cfb362d)
    assert e.max_steps == 20 and e.level == 0 and e.run_timeout_s == 600
    assert set(e.agents) == {"privescagent", "wintermute", "hacksynth"}


def test_ports_do_not_overlap_between_experiments():
    cfgs = [load_experiment_config(p) for p in EXPS]
    proxies = [c.proxy_port for c in cfgs]
    assert len(set(proxies)) == len(proxies)
    ranges = sorted((c.base_port, c.base_port + 100) for c in cfgs)
    assert all(not (b <= pp < b + 100) for pp in proxies for b, _ in ranges)
    for (a0, a1), (b0, _) in zip(ranges, ranges[1:]):
        assert a1 <= b0


def test_context_mismatch_rejected(tmp_path):
    d = yaml.safe_load(MODELS[0].read_text())
    if d["backend"] != "vllm_openai":
        d = yaml.safe_load([p for p in MODELS if "awq" in p.name][0].read_text())
    d["context_size"] = 200000
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="must equal vllm.max_model_len"):
        load_model_config(p)


def test_unknown_field_rejected(tmp_path):
    d = yaml.safe_load(EXPS[0].read_text())
    d["max_stepz"] = 3
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="unknown field"):
        load_experiment_config(p)


def test_missing_scenario_rejected(tmp_path):
    d = yaml.safe_load((CONFIG_ROOT / "experiments/smoke.yaml").read_text())
    d["scenarios"]["ids"] = ["does_not_exist"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ConfigError, match="unknown scenario"):
        select_scenarios(load_experiment_config(p))


def test_corpus_size_matches_original():
    assert len(load_scenarios("core")) == 531


def test_stratified_selection_deterministic_and_covers_all_categories():
    e = load_experiment_config(CONFIG_ROOT / "experiments/pilot.yaml")
    a, b = select_scenarios(e), select_scenarios(e)
    assert [s["scenario_id"] for s in a] == [s["scenario_id"] for s in b]
    cats = {s["category"] for s in load_scenarios("core")}
    assert {s["category"] for s in a} == cats
