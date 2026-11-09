"""Backend output-equivalence study: does vLLM change what the model says?

Subcommands
  extract   real agent prompts from finished runs (llm_calls.jsonl) -> prompts.jsonl
  vllm      greedy generations from the running vLLM server, with token ids and logprobs
            (--concurrency N also measures batch-size effects)
  hf        reference greedy generations with HF transformers (+ PEFT for LoRA). Run it with the
            vLLM env python while the vLLM server is STOPPED (one 16 GB GPU)
  compare   token-level comparison of two generation files -> JSON + Markdown

Both backends use the same decoding, chosen to mirror the agents' requests: temperature 0
(greedy), the model's generation_config.json defaults for everything else (vLLM's default
`--generation-config auto` applies them, as HF `generate` does), the same chat template, and
the same max_new_tokens.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]   # project root
sys.path.insert(0, str(HERE))


def _jl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _wjl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
def cmd_extract(a):
    seen, out = set(), []
    for d in a.exp_dirs:
        for f in sorted(Path(d).rglob("llm_calls.jsonl")):
            for r in _jl(f):
                if r.get("error_class") != "ok" or not r.get("messages"):
                    continue
                key = json.dumps(r["messages"], sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"prompt_id": f"p{len(out):03d}", "source": f"{r['agent']}:{r['run_id'][:8]}:{r['call_seq']}",
                            "agent": r["agent"], "messages": r["messages"]})
    # spread the picks over agents
    by_agent = {}
    for p in out:
        by_agent.setdefault(p["agent"], []).append(p)
    picked = []
    while len(picked) < a.n and any(by_agent.values()):
        for ag in sorted(by_agent):
            if by_agent[ag] and len(picked) < a.n:
                picked.append(by_agent[ag].pop(0))
    _wjl(a.out, picked)
    print(f"{len(picked)} prompts -> {a.out}")


# ---------------------------------------------------------------------------
def cmd_vllm(a):
    from privrep.config import load_dotenv, load_model_config
    from privrep.models.base import make_model
    load_dotenv()
    m = make_model(load_model_config(a.model), timeout_s=600)
    prompts = _jl(a.prompts)

    def one(p):
        body = {"model": m.cfg.served_model_name, "messages": p["messages"], "temperature": 0,
                "max_tokens": a.max_tokens, "logprobs": True, "top_logprobs": 1,
                "return_tokens_as_token_ids": True}
        t0 = time.perf_counter()
        st, content, _, exc, _ = m.post_raw("chat/completions", json.dumps(body).encode())
        lat = time.perf_counter() - t0
        if exc or st != 200:
            return {"prompt_id": p["prompt_id"], "error": repr(exc) if exc else content.decode()[:2000]}
        r = json.loads(content)
        ch = r["choices"][0]
        toks = (ch.get("logprobs") or {}).get("content") or []
        ids = [int(t["token"].split(":", 1)[1]) for t in toks if t["token"].startswith("token_id:")]
        lps = [t["logprob"] for t in toks]
        if ch.get("finish_reason") == "stop" and ids:  # vLLM includes the stop token; HF output has it stripped
            ids, lps = ids[:-1], lps[:-1]
        return {"prompt_id": p["prompt_id"], "text": ch["message"]["content"], "token_ids": ids,
                "logprobs": lps, "finish_reason": ch.get("finish_reason"),
                "usage": r.get("usage"), "latency_s": round(lat, 3), "backend": "vllm",
                "concurrency": a.concurrency}
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        rows = list(ex.map(one, prompts))
    _wjl(a.out, rows)
    print(f"{sum('error' not in r for r in rows)}/{len(rows)} ok -> {a.out}")


# ---------------------------------------------------------------------------
def cmd_hf(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from privrep.config import hf_snapshot, load_model_config
    cfg = load_model_config(a.model)
    # The reference is always the RELEASED form: base weights (+ the released LoRA adapter, applied
    # at runtime by PEFT). For a merged checkpoint (Model B) that means the unmerged base + adapter,
    # so this also tests the merge.
    mf = cfg.checkpoint.get("merged_from")
    if mf:
        base = hf_snapshot(mf["base_repo"], mf["base_revision"])
        lora = {"repo": mf["lora_repo"], "revision": mf["lora_revision"], "subfolder": mf["lora_subfolder"]}
    else:
        base = hf_snapshot(cfg.checkpoint["repo"], cfg.checkpoint["revision"])
        lora = cfg.lora
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
    if lora:
        from peft import PeftModel
        lp = hf_snapshot(lora["repo"], lora["revision"]) / lora["subfolder"]
        model = PeftModel.from_pretrained(model, str(lp))
    model.eval()
    gc = GenerationConfig.from_pretrained(base)
    gen_kwargs = {"do_sample": False, "max_new_tokens": a.max_tokens,
                  "repetition_penalty": getattr(gc, "repetition_penalty", 1.0) or 1.0,
                  "output_scores": True, "return_dict_in_generate": True,
                  "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id}
    rows = []
    for p in _jl(a.prompts):
        ids = tok.apply_chat_template(p["messages"], add_generation_prompt=True, return_tensors="pt").to("cuda")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(ids, **gen_kwargs)
        lat = time.perf_counter() - t0
        new = out.sequences[0, ids.shape[1]:].tolist()
        eos = set([tok.eos_token_id] + list(getattr(gc, "eos_token_id", []) or []) if isinstance(
            getattr(gc, "eos_token_id", None), list) else [tok.eos_token_id, getattr(gc, "eos_token_id", None)])
        lps = []
        for i, sc in enumerate(out.scores):
            lp = torch.log_softmax(sc[0].float(), -1)
            lps.append(float(lp[new[i]]))
        while new and new[-1] in eos:  # vLLM does not return the stop token
            new.pop(); lps.pop()
        rows.append({"prompt_id": p["prompt_id"], "text": tok.decode(new, skip_special_tokens=True),
                     "token_ids": new, "logprobs": lps, "prompt_tokens": int(ids.shape[1]),
                     "latency_s": round(lat, 3), "backend": "hf_transformers",
                     "gen_kwargs": {k: v for k, v in gen_kwargs.items() if k not in ("output_scores", "return_dict_in_generate")}})
        print(p["prompt_id"], len(new), "tokens", round(lat, 1), "s", flush=True)
    import transformers, peft  # noqa: E401
    meta = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    for r in rows:
        r["versions"] = meta
    _wjl(a.out, rows)
    print(f"{len(rows)} -> {a.out}")


# ---------------------------------------------------------------------------
def compare_rows(A: list[dict], B: list[dict]) -> dict:
    bmap = {r["prompt_id"]: r for r in B}
    per = []
    for ra in A:
        rb = bmap.get(ra["prompt_id"])
        if not rb or "error" in ra or "error" in rb:
            per.append({"prompt_id": ra["prompt_id"], "status": "missing_or_error"})
            continue
        ta, tb = ra["token_ids"], rb["token_ids"]
        k = 0
        while k < min(len(ta), len(tb)) and ta[k] == tb[k]:
            k += 1
        lpa, lpb = ra.get("logprobs") or [], rb.get("logprobs") or []
        deltas = [abs(x - y) for x, y in zip(lpa[:k], lpb[:k])]
        per.append({"prompt_id": ra["prompt_id"], "exact_token_match": ta == tb,
                    "exact_text_match": ra["text"] == rb["text"], "len_a": len(ta), "len_b": len(tb),
                    "common_prefix_tokens": k, "first_divergence": None if ta == tb else k,
                    "max_abs_logprob_delta_on_prefix": round(max(deltas), 5) if deltas else None,
                    "mean_abs_logprob_delta_on_prefix": round(sum(deltas) / len(deltas), 6) if deltas else None,
                    "logprob_margin_at_divergence": None})
    ok = [p for p in per if "exact_token_match" in p]
    return {"n": len(per), "n_compared": len(ok),
            "exact_token_match": sum(p["exact_token_match"] for p in ok),
            "exact_text_match": sum(p["exact_text_match"] for p in ok),
            "median_common_prefix": sorted(p["common_prefix_tokens"] for p in ok)[len(ok) // 2] if ok else None,
            "max_logprob_delta": max((p["max_abs_logprob_delta_on_prefix"] or 0) for p in ok) if ok else None,
            "per_prompt": per}


def cmd_compare(a):
    rep = compare_rows(_jl(a.a), _jl(a.b))
    rep.update(a_file=a.a, b_file=a.b, label=a.label)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_prompt"}, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="privrep.equivalence.compare")
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract"); e.add_argument("exp_dirs", nargs="+"); e.add_argument("--n", type=int, default=20)
    e.add_argument("--out", required=True)
    v = sp.add_parser("vllm"); v.add_argument("--model", required=True); v.add_argument("--prompts", required=True)
    v.add_argument("--out", required=True); v.add_argument("--max-tokens", type=int, default=256)
    v.add_argument("--concurrency", type=int, default=1)
    h = sp.add_parser("hf"); h.add_argument("--model", required=True); h.add_argument("--prompts", required=True)
    h.add_argument("--out", required=True); h.add_argument("--max-tokens", type=int, default=256)
    c = sp.add_parser("compare"); c.add_argument("--a", required=True); c.add_argument("--b", required=True)
    c.add_argument("--out", required=True); c.add_argument("--label", default="")
    a = ap.parse_args(argv)
    {"extract": cmd_extract, "vllm": cmd_vllm, "hf": cmd_hf, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
