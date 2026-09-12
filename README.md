# Qwen3.8-Flash-Next NVFP4 on RTX 5090 — FreeToken Benchmarks

**Hardware:** ASUS ROG Astral RTX 5090 32GB · 128GB DDR5 (4×32) · Intel Core Ultra 7 265K · Samsung 9100 Pro NVMe  
**Engine:** [FreeToken](https://github.com/FlashML-org/FreeToken) (`ft serve`)  
**Model:** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)  
**Date:** 2026-09-12  

Single-GPU, single-request (`--max-running-requests 1`) local serving. Six systemd configurations were measured with **llama-benchy** and **aiperf**.

---

## Headline results (honest, sustained)

| Metric | Best 128k (Case 4) | Full 262k (Case 5/6) |
|--------|-------------------:|---------------------:|
| **Decode (llama-benchy mean)** | **61–67 tok/s** | **56–58 tok/s** |
| **Decode (aiperf avg)** | **64–66 tok/s** | **55–60 tok/s** |
| **Prefill (aiperf, mid/long)** | **~3.3–3.5k tok/s** | **~3.2–3.5k tok/s** |
| **TTFT @ ~2k input** | **~1.5 s** | **~1.5–1.6 s** |
| **TTFT @ ~65k input** | **~18 s** | **~18 s** |
| **TTFT @ ~130k input** | **~35–38 s** | **~35–39 s** |
| **TTFT @ ~262k input** | — | **~72–81 s** |
| **Validated max context** | **131,072** | **262,144** |

Live `gen throughput` lines in the service journal can spike higher (e.g. 70–90 tok/s on short bursts). The tables below are **sustained** tool measurements (3-run means), not peak log lines.

> **Tokenizer caveat (transparency):** llama-benchy could not resolve the model tokenizer by name in this suite and fell back to a **gpt2 tokenizer approximation** (visible in `case*/llamabenchy_full.txt`). The `--depth` values are therefore approximate token counts, not exact Qwen tokens. Cross-case comparisons stay valid (same fallback for all six cases); absolute per-token rates carry that approximation error. aiperf used the local Qwen tokenizer, so its columns are exact. Pass `--tokenizer nvidia/Qwen3.8-Flash-Next-NVFP4` (or a local path) to reproduce without the fallback.

---

## Hardware map

```
HOST RAM 128 GB
├── ~60 GB  NVFP4 expert pool (offloaded)
├── 47.7 GiB PLE n-gram table        ← only if --ple-backend pinned
├── runtime / OS / page cache
└── optional NVMe swap safety net

RTX 5090 32 GB VRAM
├── model residuals / CUDA
├── MoE expert LRU cache (e.g. 5489 slots @128k)
├── KV cache (3.09 GiB @128k · 6.19 GiB @262k)
└── ~2–3 GiB free headroom after graphs
```

Neither the full expert set nor the 47.7 GiB FP8 PLE table fits in 32 GB VRAM alone. FreeToken keeps experts in host RAM and streams active experts over PCIe; PLE is either streamed from NVMe (`disk`) or page-locked in RAM (`pinned`).

---

## The six cases

| Case | Context | memory-ratio | MoE cache | PLE | Extra | Role |
|------|--------:|-------------:|-----------|-----|-------|------|
| **1** | 128k | 0.92 | **auto → 5740** | disk | — | Auto baseline |
| **2** | 128k | 0.92 | **fixed 5489** | disk | — | Fixed cache, disk PLE |
| **3** | 128k | 0.95 | fixed 5489 | **pinned** | LimitNOFILE | Pinned PLE |
| **4** | 128k | 0.95 | fixed 5489 | **pinned** | NOFILE + MEMLOCK + CPUAffinity 0–7 | **Best average decode** |
| **5** | **262k** | 0.93 | **auto → 4662** | pinned | low RAM → serial expert load | Max context, RAM tight |
| **6** | **262k** | 0.92 | auto | **disk** | expert-load **parallel**, prefill **10240** | Max context, RAM healthy |

All six used `--moe-strategy offload`. Full flag lists for every case: [`details.txt`](details.txt) · one-line summaries: [`data/case_configs.txt`](data/case_configs.txt) · recommended units: [`services/`](services/) · raw numbers: [`data/benchmark_results.csv`](data/benchmark_results.csv).

The benchmark runs used transient units named `freetoken-bench1.service` … `freetoken-bench6.service`.

---

## Decode speed (llama-benchy, tok/s mean)

| Context | Case1 | Case2 | Case3 | Case4 | Case5 | Case6 |
|--------:|------:|------:|------:|------:|------:|------:|
| 2k | 61.7 | **64.8** | 62.4 | 63.7 | 57.1 | 57.9 |
| 20k | 61.6 | 61.6 | 63.8 | **66.8** | 55.6 | 56.6 |
| 65k | **64.7** | 60.6 | 61.8 | 62.6 | 57.7 | 56.2 |
| 130k | 60.4 | 59.8 | **64.1** | 61.4 | 57.0 | 58.0 |
| 262k | — | — | — | — | 57.0 | **57.4** |

**128k average (2k–130k):** Case4 **63.6** > Case3 **63.0** > Case1 **62.1** > Case2 **61.7**.  
**262k:** Case5/6 stay in a narrow **56–58** band.

---

## Prefill & TTFT (aiperf)

### Prefill (Active Prefill Throughput Per User, tok/s)

| Context band | Case1 | Case2 | Case3 | Case4 | Case5 | Case6 |
|--------------|------:|------:|------:|------:|------:|------:|
| ~2k | 1.2k | 1.2k | 1.3k | 1.3k | 1.3k | 1.2k |
| ~20k | 3.0k | 3.1k | 3.3k | 3.3k | 3.3k | 3.3k |
| ~65k | 3.3k | 3.3k | **3.5k** | **3.5k** | 3.5k | 3.3k |
| ~130k | 3.2k | 3.3k | 3.5k | 3.5k | 3.5k | 3.3k |
| ~262k | — | — | — | — | 3.4k | 3.2k |

### TTFT (ms, aiperf)

| Input scale | Case1 | Case4 | Case5 | Case6 |
|-------------|------:|------:|------:|------:|
| ~2k | 1651 | **1531** | 1531 | 1611 |
| ~20k | 6578 | **6069** | 6073 | 5990 |
| ~65k | 19869 | **18336** | 18382 | 19856 |
| ~130k | 40399 | **37647** | 37627 | 39443 |
| ~262k | — | — | **78136** | 81366 |

TTFT is dominated by prefill of the synthetic prompt; pinned PLE (Case3/4) trims a bit off mid/long TTFT vs disk.

---

## What each tuning actually did

| Change | Effect on this box |
|--------|--------------------|
| **`--ple-backend pinned`** | Uses ~48 GB host RAM; small–moderate decode/TTFT win vs `disk`. Required for Case4-class results. |
| **`--ple-backend disk`** | Saves ~48 GB RAM (Case6: ~44 GB free). Decode stays ~56–58 @262k. Prefer on RAM-constrained hosts. |
| **`--moe-cache-auto`** | Picks cache size from free VRAM after KV reserve (**5740** @128k, **4662** @262k). Safest default. |
| **`--moe-cache-size 5489`** | Best **reproducible** 128k setting in this suite (Case2–4). Do **not** copy to 262k blindly. |
| **`--memory-ratio 0.95`** | Slightly more VRAM for expert cache / weights; paired well with pinned. |
| **`--num-tokens` = `--max-seq-len-override`** | Avoids oversizing the KV allocator. |
| **`--expert-load parallel`** | Faster expert-bank build when RAM is free (Case6). Case5 auto-fell back to **serial** under RAM pressure. |
| **`--max-prefill-length 10240`** | Larger prefill chunks (Case6); decode unchanged. |
| **`CPUAffinity=0-7`** | Pins to P-cores on 265K; no large decode regression in Case4. |
| **`LimitMEMLOCK=infinity`** | Helps pinned pages; use with `ple pinned`. |

**Rule of thumb**

- Daily **≤128k, max decode** → Case4 (pinned + `5489` or auto).  
- **262k** → Case6 (disk + parallel) for RAM headroom, or Case5 if you accept pinned RAM use for a tiny aiperf edge.  
- Leave **`--moe-cache-auto`** unless you freeze an entire unit file and paste the `resolved moe_cache_size=…` line from that same boot.

Capture the resolved cache for the unit you are about to freeze:

```bash
journalctl -u freetoken-qwen38-128k-case4 --no-pager | grep -F 'moe-cache-auto resolved'
```

(During this study the transient bench units were `freetoken-bench1` … `freetoken-bench6`; substitute whichever unit you actually run.)

---

## Recommended systemd unit (128k production)

See [`services/freetoken-qwen38-128k-case4.service`](services/freetoken-qwen38-128k-case4.service).

> ⚠️ Adjust before use: `User=`, `WorkingDirectory=`, venv path under `ExecStart=`, and `--model-path`. The units bind `--host 0.0.0.0` with **no auth** — change to `127.0.0.1` unless you deliberately serve your LAN.
>
> An optional GPU power-cap one-shot is provided in [`services/gpu-power-limit.service`](services/gpu-power-limit.service); adjust `--pl` to your card and enable it separately if you want it.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now freetoken-qwen38-128k-case4.service
journalctl -u freetoken-qwen38-128k-case4 -f
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next-NVFP4",
    "messages": [{"role":"user","content":"Count from 1 to 40, one number per line."}],
    "max_tokens": 80,
    "temperature": 0
  }' | jq '.usage'
```

---

## How to reproduce the benches

One-shot wrapper for both tools: [`scripts/run_bench.sh`](scripts/run_bench.sh).

```bash
# llama-benchy (append 261800 to --depth for the 262k cases 5–6)
llama-benchy --base-url http://localhost:8000/v1 \
  --model Qwen3.8-Flash-Next-NVFP4 \
  --depth 2048 20000 65000 130800 \
  --pp 512 --tg 128 --runs 3 \
  --save-result results.json --format json

# aiperf (example: ~65k prompt)
aiperf profile \
  --model Qwen3.8-Flash-Next-NVFP4 \
  --url http://localhost:8000 \
  --tokenizer /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 65000 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --streaming \
  --artifact-dir ./out \
  --profile-export-prefix aiperf_65k \
  --export-level summary
```

---

## Operational notes

1. **Pinned PLE needs ~128 GB class RAM.** On this host, Case5 peaked near **119 GB** RAM with swap use; Case6 (`disk`) stayed comfortable (~44 GB free).  
2. **Never** assume `moe_cache_size` from a 128k run works on 262k (or the reverse).  
3. `qsa_sparse` forces `page_size=64` — expected warning.  
4. Triton `_POSIX_C_SOURCE` redefine warnings at first graph capture are noisy but normal.  
5. Single-user only in this study (`max-running-requests=1`).

---

## Files

```
README.md                    # this file
data/benchmark_results.csv   # every measured row
data/case_configs.txt        # one-line config summary
details.txt                  # full per-case unit files + RAM/VRAM snapshots
services/                    # recommended units (+ optional power-cap sample)
scripts/run_bench.sh         # reproduce the suite
images/                      # charts
case1/ … case6/              # llama-benchy + aiperf raw console output, JSON summaries, server logs
```

---

## License / credit

Benchmarks run on personal hardware; results and analysis are licensed CC-BY-4.0 (see [`LICENSE`](LICENSE)). Model weights: NVIDIA / Qwen as published on Hugging Face. Engine: FlashML-org FreeToken (Apache-2.0).
