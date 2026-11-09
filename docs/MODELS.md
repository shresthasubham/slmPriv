# Models: specification, installation, serving

Two models are evaluated. Both are served locally by **vLLM 0.24.0** through its OpenAI-compatible
API on `127.0.0.1:8000`, one at a time (each fills a 16 GB GPU with a 32k context).

| | Model A | Model B |
|---|---|---|
| config | `configs/models/qwen2.5-7b-instruct-awq.yaml` | `configs/models/privesc-llm-4b-rl.yaml` |
| what it is | general-purpose instruct model | a privilege-escalation fine-tune (SFT + RL) |
| Hugging Face weights | `Qwen/Qwen2.5-7B-Instruct-AWQ` | base `Qwen/Qwen3-4B-Instruct-2507` + LoRA `sailab-vienna/privesc-llm-4b`, subfolder `rl_adapter` |
| pinned revision | `b25037543e9394b818fdfca67ab2a00ecc7dd641` | base `cdbee75f17c01a7cc42f958dc650907174af0554`, adapter `e7a10aadbcc0d0cd721533480d4fd47ee083dc63` |
| parameters | 7.6 B | 4.0 B (+ rank-8 LoRA) |
| precision on GPU | AWQ 4-bit (`--quantization awq`) | bfloat16 (merged weights) |
| served name (`model` field the agents send) | `qwen2.5-7b` | `privesc-llm-4b` |
| context (`--max-model-len` = what the agents are told) | 32 768 | 32 768 |
| weights on disk | 5.3 GB | 8.0 GB base + 68 MB adapter + 8.3 GB merged |
| license | Apache-2.0 | base Apache-2.0; adapter: see its model card |
| origin | Qwen team | TU Wien, arXiv 2603.17673 (**not** part of the PrivEscalate paper) |

Model B was trained for a hackingBuddyGPT-style (wintermute) harness. Under the PrivEscAgent and
HackSynth prompts it is out of distribution; this is reported, not corrected.

The original PrivEscalate paper evaluated API models only. Both local models are an extension, not
cells of the paper's tables.

---

## 1. Install vLLM

```bash
bash scripts/setup_vllm.sh
```

which does:

```bash
uv venv .venv-vllm --python 3.12
UV_TORCH_BACKEND=cu130 uv pip install --python .venv-vllm/bin/python -r requirements/vllm.txt
```

Tested stack: vLLM 0.24.0, torch 2.11.0+cu130, transformers 5.22.1, peft 0.21.2, safetensors 0.8.0,
accelerate 1.15.0, NVIDIA driver 615.71, RTX 5060 Ti 16 GB. The model configs pin
`vllm.version: 0.24.0`, and `scripts/validate.py` fails if the installed version differs.

Use another location with `VLLM_VENV=/path/to/venv` (or `VLLM_BIN=/path/to/bin/vllm`) in the
environment or in `secrets.env`.

**No CUDA toolkit needed.** vLLM's default FlashInfer sampler compiles a kernel at start-up with
`nvcc`. Without the toolkit that fails, so both configs set:

```yaml
env:
  VLLM_USE_FLASHINFER_SAMPLER: "0"     # use the PyTorch sampler
  VLLM_ATTENTION_BACKEND: FLASH_ATTN   # use the prebuilt FlashAttention kernels
```

---

## 2. Download the weights

```bash
.venv-vllm/bin/python scripts/download_models.py           # both models
.venv-vllm/bin/python scripts/download_models.py privesc   # only Model B
```

Only the pinned revisions are downloaded (to the Hugging Face cache, `HF_HOME`), no repository code
is executed (`trust_remote_code` is never used), and the SHA-256 of every weight file is printed.
Expected:

| file | SHA-256 |
|---|---|
| Qwen2.5-7B-Instruct-AWQ `model-00001-of-00002.safetensors` | `4ad6e70f93823a9228931615e04b5e9293854c236fb4bd5f85e27f3b600605ad` |
| Qwen2.5-7B-Instruct-AWQ `model-00002-of-00002.safetensors` | `920a8cc9d3c81668f18014a69bb041ab8bab22a31640a713f90f5e7580e2f126` |
| Qwen3-4B-Instruct-2507 `model-00001-of-00003.safetensors` | `75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6` |
| Qwen3-4B-Instruct-2507 `model-00002-of-00003.safetensors` | `0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1` |
| Qwen3-4B-Instruct-2507 `model-00003-of-00003.safetensors` | `7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d` |
| privesc-llm-4b `rl_adapter/adapter_model.safetensors` | `f4d941ec68ce361f9b40f93f42ebfe6f7242afbb43e4cd488e1375b03821eb12` |

---

## 3. Model B: merge the released adapter

```bash
.venv-vllm/bin/python scripts/merge_lora.py configs/models/privesc-llm-4b-rl.yaml
```

Runs on CPU (no GPU needed, ~16 GB RAM, about a minute) and writes
`models/privesc-llm-4b-rl-merged/` with a `MERGE_MANIFEST.json` (inputs, method, versions and the
SHA-256 of every output file). It refuses to overwrite an existing output.

**Why merge?** The adapter's LoRA targets include `lm_head`
(`r = 8, alpha = 32, targets = q,k,v,o,gate,up,down,lm_head`). Served through vLLM 0.24.0's LoRA
path (`--enable-lora --lora-modules …`), its greedy output did **not** match the Hugging Face
transformers + PEFT reference: on the first wintermute prompt, vLLM-LoRA produced a 4 096-token
runaway list, while HF + PEFT answered `ls /etc/passwd` (5 tokens). A merged checkpoint is the same
model in plain weights, and vLLM serves plain weights exactly as it serves any base model.

**How.** `peft.PeftModel.merge_and_unload(safe_merge=True)` in bfloat16, with one extra step: the
base model ties `lm_head` to the input embeddings (`tie_word_embeddings: true`). At runtime PEFT
applies the `lm_head` delta only to the output projection, but a naive merge writes it into the
*shared* tensor and so also changes the input embeddings. The script therefore gives `lm_head` its
own copy of the weights first (untie), merges, and asserts that the input embeddings are bit-for-bit
unchanged. `validate.py` checks the manifest (`untied_lm_head: true`, revisions match the config).

Expected (tested environment): `model.safetensors` SHA-256
`f19a54eb719ddf803cb5b490f3d2be5883f2911f057a36efe95f83c6b52f7e6b`. A different torch / peft
version can produce a different (but equivalent) file; compare with the
[output-equivalence](#output-equivalence) test rather than by hash in that case.

---

## 4. Serve

```bash
.venv/bin/python -m privrep.serve configs/models/privesc-llm-4b-rl.yaml --print   # print the command only
.venv/bin/python -m privrep.serve configs/models/privesc-llm-4b-rl.yaml           # run vLLM in the foreground
```

`privrep.serve` builds the command from the config and runs it offline (`HF_HUB_OFFLINE=1`, so
nothing is downloaded at serve time). The exact commands it runs:

**Model A**

```bash
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_API_KEY=<secrets.env> \
.venv-vllm/bin/vllm serve ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641 \
  --port 8000 --host 127.0.0.1 --max-model-len 32768 --gpu-memory-utilization 0.9 --dtype auto --seed 0 \
  --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641 \
  --quantization awq --no-enable-prefix-caching --served-model-name qwen2.5-7b \
  --enable-auto-tool-choice --tool-call-parser hermes
```

**Model B**

```bash
HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_API_KEY=<secrets.env> \
.venv-vllm/bin/vllm serve models/privesc-llm-4b-rl-merged \
  --port 8000 --host 127.0.0.1 --max-model-len 32768 --gpu-memory-utilization 0.9 --dtype bfloat16 --seed 0 \
  --no-enable-prefix-caching --served-model-name privesc-llm-4b \
  --enable-auto-tool-choice --tool-call-parser hermes
```

| flag | why |
|---|---|
| `--host 127.0.0.1` + `VLLM_API_KEY` | the server is only reachable locally and needs the key; the agents never get the key (the logging proxy adds it) |
| `--max-model-len 32768` | must equal the `context_size` given to the agents, so they trim history at the real limit (validated) |
| `--seed 0`, `--no-enable-prefix-caching` | removes avoidable sources of nondeterminism |
| `--gpu-memory-utilization 0.9` | leaves room on a 16 GB card; vLLM pre-allocates the KV cache |
| `--enable-auto-tool-choice --tool-call-parser hermes` | lets tool-calling requests be parsed (Qwen uses the Hermes format) |
| generation config | vLLM's default `--generation-config auto` applies each model's `generation_config.json` defaults to fields the agent does not set, as HF `generate` does |

Check that it is up:

```bash
KEY=$(grep ^VLLM_API_KEY= secrets.env | cut -d= -f2)
curl -s localhost:8000/version
curl -s localhost:8000/v1/models -H "Authorization: Bearer $KEY"
.venv/bin/python scripts/validate.py --probe       # served name listed + a 1-token generation
```

The scripts `run_experiment.sh` / `run_study.sh` start and stop the server automatically.

### Request parameters the harness adds

The logging proxy forwards each agent request unchanged, except:

* `max_tokens: 4096` **only if the request has none** (hackingBuddyGPT sends none; HackSynth sends
  1 024 itself). Without it a local server allows up to `max_model_len − prompt` output tokens, and a
  degenerate generation can consume the whole 600 s run budget. Set by `default_max_tokens` in the
  experiment config.
* `seed: <run seed>` if the request has none (`inject_seed: true`). No effect at temperature 0.

Both are recorded per call in `llm_calls.jsonl → injected_params`.

---

## Output equivalence

Question: does serving through vLLM change what the model says, compared with the reference
implementation (HF transformers, + PEFT for the released adapter)?

```bash
bash scripts/equivalence.sh configs/models/privesc-llm-4b-rl.yaml runs/smoke/privesc-llm-4b-rl/<dir> 20
```

1. extracts N distinct real agent prompts from a finished experiment (spread over the three agents),
2. generates them greedily (temperature 0, 256 tokens) with vLLM: twice at concurrency 1 (run-to-run)
   and once at concurrency 8 (batching),
3. stops vLLM and generates them with HF transformers on the same GPU. For Model B the reference is
   the **unmerged** base + released adapter through PEFT, so this also tests the merge,
4. compares token ids, text and log-probabilities (`cmp_*.json`).

Interpreting the output: identical text/tokens at concurrency 1 run-to-run is expected (greedy, fixed
seed, no prefix caching). Small differences between concurrency 1 and 8, or between vLLM and HF, come
from batched / different bf16 kernels; compare the shared-prefix length and the log-probability
differences in `cmp_*.json`, not only exact matches.

Model A is an AWQ checkpoint. A HF reference would need AutoAWQ, which is not available for this
torch/CUDA stack, so for it only the vLLM self-consistency checks (run-to-run, batching) apply:

```bash
bash scripts/equivalence.sh configs/models/qwen2.5-7b-instruct-awq.yaml runs/smoke/qwen2.5-7b-instruct-awq/<dir> 20
```
