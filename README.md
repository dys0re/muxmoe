# MuxMoE

MuxMoE is an online-batching offload inference platform for serving large
Mixture-of-Experts models on a single GPU with limited VRAM.

Its central design point is online batching: independently arriving requests
share a continuous scheduler, expert cache, and PCIe prefetch pipeline while
retaining normal per-request streaming semantics. The current reference target
is Qwen 35B-A3B on a constrained single-GPU host.

It keeps the familiar SGLang server and OpenAI-compatible API, but changes the
parts that matter most for a large MoE model on a small host:

- routed expert weights can live in pinned host memory instead of occupying the
  whole GPU;
- a bounded GPU expert cache admits, evicts, and prefetches experts at runtime;
- cross-layer routing prediction overlaps expert transfers with computation;
- fragmented PCIe copies can be submitted through `cudaMemcpyBatchAsync`;
- small-batch route metadata uses one CPU snapshot, with optional overlap of
  resident-expert computation and missing-expert transfers;
- hybrid linear-attention state uses growable slabs and graph-compatible buffers;
- breakable decode CUDA Graphs keep dynamic expert operations outside capture;
- bundled speculative decoding and Mamba state rollback are integrated with the
  offload path.

This is an engineering and research fork, not an official SGLang distribution
or a drop-in replacement for every upstream configuration. The current target is
the Qwen 35B-A3B workload described below.

## Target configuration

The primary target is a GPTQ Int4 Qwen 35B-A3B checkpoint with the `moe_wna16`
runner:

| Item | Reference setup |
| --- | --- |
| GPU | NVIDIA RTX 5060 Ti, 16 GB |
| Execution | One CUDA device, one process |
| Quantization | Symmetric GPTQ Int4, `desc_act=false` |
| Host expert budget | 12.5 GiB pinned memory |
| GPU expert cache | 3.0 GiB |
| Resident layers | 6 shallow MoE layers |
| Decode graph | Breakable CUDA Graph, dynamic expert operations eager |
| Cache policy | Lease, lease length 3 |
| Expert execution | `serial` by default; `overlap` is opt-in for small GPTQ decode batches |

The figures below are configuration-specific reference points, not
model-independent or hardware-independent claims.

## Reference platform

The Qwen 35B-A3B figures in this README come from the following single-GPU
host. The platform details are part of the result: this is an old X99-class
Intel host with a relatively slow PCIe path, not a modern server platform.

| Component | Details |
| --- | --- |
| Host platform | X99-class Intel Xeon E5 v3 platform |
| CPU | Intel Xeon E5-2643 v3, 6 physical cores / 12 threads, 3.40 GHz base, up to 3.70 GHz |
| NUMA | One socket, one NUMA node |
| System memory | 31 GiB RAM plus 8 GiB zram/swap |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB, compute capability 12.0 (SM120) |
| PCIe topology | GPU attached through the Xeon E5 v3 CPU root port |
| Host-side link | PCIe Gen3 x8 in the tested configuration |
| Operating system | Fedora Linux 44, x86_64, Linux 7.0.7 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu130 |
| CUDA toolkit | 13.3 |
| CUDA host compiler | Conda GCC/G++ 15.2; system GCC 16 is not used for CUDA builds |
| Runtime environment | Conda environment `muxmoe`, one visible GPU, `MAX_JOBS=1` |

### PCIe is part of the decode bottleneck

Expert offload changes decode from a GPU-only workload into a repeated
host-to-device pipeline. Each decode step routes experts, transfers the missing
GPTQ payloads over PCIe, waits for the copies and synchronization, and then
executes the MoE kernel. On this host:

- the GPU is capable of a newer PCIe generation, but the X99/E5 v3 host path is
  limited to Gen3 x8;
- the practical Gen3 x8 payload ceiling is about 7.88 GB/s;
- active expert-copy measurements reach roughly 6.60–6.94 GB/s, or 84–88% of
  that ceiling;
- an individual GPTQ expert payload is about 1.547 MiB, so decode consists of
  many short transfers rather than one long saturated copy;
- request-wide average payload traffic is only about 1.66–1.99 GB/s because
  transfers are interleaved with routing, synchronization, and compute.

Therefore the reported decode speed is strongly platform-bound. The dominant
issue is not simply sustained bandwidth: the runtime repeatedly pays the
latency and scheduling cost of a slow Gen3 x8 link. Expert locality, cache hit
rate, and cross-layer prefetch determine how often decode has to wait for that
link. A newer Gen4/Gen5 host may improve the result, but throughput will not
scale linearly with nominal PCIe bandwidth because the traffic is bursty and
model-dependent.

## Qwen 35B-A3B performance

### Latest matched-platform results (2026-10-15)

Each cell is the median of **five measured runs after one excluded warmup**,
using a fresh server for each workload and variant. Every request generates
128 tokens with greedy sampling, `ignore_eos=true`, and MTP disabled. Aggregate
decode throughput starts after every request has emitted its first token,
excluding prefill and TTFT. Server startup, JIT compilation, and diagnostic
profiling are outside the measurements.

| Workload | Frozen baseline | Serial metadata (default) | Resident/miss overlap (opt-in) |
| --- | ---: | ---: | ---: |
| BS=1, one request | 18.51 tok/s | **18.53 tok/s** | 17.43 tok/s |
| BS=4, four related requests | 52.24 tok/s | **59.45 tok/s** | 57.80 tok/s |
| BS=8, related prompts | 68.12 tok/s | 73.08 tok/s | **81.92 tok/s** |
| BS=8, unrelated prompts | 43.32 tok/s | 44.00 tok/s | **51.40 tok/s** |

The default serial path improves BS=4 by **13.81%** and the two BS=8 workloads
by **7.28% / 1.57%**, while BS=1 stays effectively unchanged (**+0.14%**).
Overlap improves the two BS=8 workloads by **20.26% / 18.65%**, but reduces
BS=1 throughput by **5.84%**, so it remains opt-in.

All variants use the cache budgets above, `--moe-expert-drop-threshold 0`,
adaptive prefetch ratio 0.5, disabled radix cache, seed 41, and breakable decode
graphs. BS=1/4 use `mem_fraction_static=0.70` and Mamba capacities 5/12;
BS=8 uses 0.75 and capacity 8. BS=4 uses the first four related prompts with
128-token outputs; it is a different workload and timing protocol from the
older uniform 64-token BS=4 experiment.

The [published measurements](benchmark/moe_expert_cache/results/2026-10-15-rtx5060ti.json)
include prompts, sanitized server commands, every measured throughput and
latency value, and baseline comparisons. These are new matched-platform
measurements; changes are calculated against the frozen baseline in the same
table.

Validation passed 68 focused tests. A separate same-input BS=8 diagnostic
compared 128 mixed-hit direct-resident executions with serial computation:
all were elementwise identical, with maximum absolute error 0. Staging is
covered by the CUDA regression suite. Free-running BS=4/8 outputs are not
fully token-identical across versions or repeated baseline requests; these
throughput results do not establish full-model accuracy or fixed-token-trace
equivalence. All GPU/JIT runs used the RAM guard and `MAX_JOBS=1`; peak effective
system RAM was **23.10 GiB**, with no watchdog trips or OOM kills.

### Historical locality counters (2026-08-02)

The earlier related/unrelated prompt study measured the following counters at
equal cache forwards. They illustrate routing locality and are separate from
the latest throughput measurements above:

| Counter at equal cache forwards | Related | Unrelated |
| --- | ---: | ---: |
| Expert-cache hit rate | 90.64% | 85.33% |
| Cache misses | 49,283 | 98,121 |
| Demand host-to-device traffic | 74.45 GiB | 148.22 GiB |
| Cross-layer prefetch match rate | 86.82% | 86.07% |

These earlier counters show how request locality changes cache hit rates and
host-to-device traffic at the serving level.

### Historical PCIe copy and prefetch measurements (2026-08-04)

The earlier BS=1 non-MTP progression below measures whole-request throughput,
including prefill/TTFT. Its values should not be compared directly with the
latest decode-only table:

| Variant | Median request throughput | Change |
| --- | ---: | ---: |
| Previous tuned baseline | 17.8867 tok/s | — |
| Batched expert copies | 18.0528 tok/s | +0.93% |
| Full-width cross-layer prefetch | 18.3213 tok/s | +2.43% |

`cudaMemcpyBatchAsync` reduces host submission overhead. The end-to-end gain is
intentionally modest: PCIe transfers are bursty, and the main opportunity is to
fill idle gaps between MoE layers rather than to claim a higher physical DMA
bandwidth.

Full-width prefetch is an explicit trade-off. It lowers demand-side waiting,
but moves more speculative bytes and reduces immediate transfer efficiency. The
default keeps an adaptive prefetch minimum; use a full-width setting only when
latency is more important than host-to-device traffic.

### Bundled MTP status (historical)

Bundled MTP is functional with target GPTQ weights, BF16 draft weights, expert
offload, GDN state rollback/commit, and breakable decode graphs. The earlier
whole-request benchmark did not show a performance win on a 16 GB GPU:

| Configuration | Median request throughput | Acceptance |
| --- | ---: | ---: |
| Non-MTP tuned baseline | 17.89 tok/s | — |
| MTP-2 | 12.91 tok/s | 1.82–1.93 tokens, 0.82–0.93 rate |
| MTP-3 | 14.12 tok/s | 2.55–2.77 tokens, 0.78–0.89 rate |

That experiment used a smaller target expert cache, and its verification
forwards were treated as prefill by the expert-cache policy, preventing decode
admission and prefetch. MTP was not changed or retested in the 2026-10-15
optimization above; the historical slowdown is not attributed solely to draft
memory or the wider expert set. MTP is included as a working path for further
optimization, not as a claimed speedup.

## How the runtime works

### Expert offload

The offload path owns routed-expert payloads only. Router weights, attention,
shared experts, and activation scales remain device-resident. A layer is
admitted atomically: its complete supported expert payload must fit the host
budget or that layer stays device-resident.

At decode time, MuxMoE combines:

1. a bounded per-layer GPU expert cache with lease/LRU-style replacement;
2. direct execution for already-resident experts;
3. compact staging for cache misses;
4. cross-layer route prediction and adaptive prefetch;
5. batched pointer-to-pointer copies with a safe individual-copy fallback.

For ordinary GPTQ decode batches of up to eight tokens, the serial path reuses
one CPU route snapshot for deduplication and ID remapping. The opt-in overlap
path computes resident routes while misses transfer, stages only missing
experts, and combines the weighted results in the original top-k lane order.
All-hit and all-miss forwards keep a single computation path.

The cache exposes hit, miss, demand-transfer, prefetch, and wasted-prefetch
counters so that locality and PCIe behavior are visible at runtime.

### Hybrid-attention state

Qwen 35B-A3B uses a hybrid attention structure. The local state path adds a
growable Mamba state store, logical slots, slab-backed state views, checkpoint
separation, and fixed-address graph buffers. When the active slots fit a single
slab, GDN can access them directly; otherwise it gathers a compact batch view,
runs the kernel, and writes the state back.

This makes state capacity, radix/checkpoint behavior, padding, and CUDA Graph
capture explicit parts of the memory design rather than incidental tensor
allocations.

## Quick start

The editable Python package remains under `python/` for compatibility with the
SGLang server entrypoints.

```bash
conda create -n muxmoe python=3.12 -y
conda activate muxmoe
pip install -e ./python
```

Set `MODEL_PATH` to a local Qwen 35B-A3B GPTQ Int4 checkpoint, then start a
BS=8-oriented server:

```bash
export MODEL_PATH=/path/to/Qwen-35B-A3B-GPTQ-Int4
RUN_NAME="server-$(date +%Y%m%d-%H%M%S)"

./profile_output/ram_guard.py \
  --log "profile_output/${RUN_NAME}.csv" \
  --command-log "profile_output/${RUN_NAME}.log" \
  -- env MAX_JOBS=1 PYTHONPATH="$PWD/python" \
  python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --quantization moe_wna16 \
  --moe-expert-offload-gb 12.5 \
  --moe-expert-cache-gb 3.0 \
  --moe-expert-resident-shallow-layers 6 \
  --moe-expert-cache-policy lease \
  --moe-expert-cache-lease 3 \
  --moe-expert-drop-threshold 0 \
  --moe-expert-cache-execution serial \
  --max-running-requests 8 \
  --max-mamba-cache-size 8 \
  --max-total-tokens 2048 \
  --random-seed 41 \
  --mem-fraction-static 0.75 \
  --cuda-graph-backend-decode breakable \
  --cuda-graph-max-bs-decode 8 \
  --disable-radix-cache \
  --host 0.0.0.0 \
  --port 30000
```

To opt into resident/miss overlap, change the execution argument to
`--moe-expert-cache-execution overlap`. The table above shows its BS=8 benefit
and BS=1 trade-off; `serial` remains the default. This overlap path supports
ordinary small-batch GPTQ/Triton decode; prefill and other unsupported cases
retain their existing execution path.

The server exposes the usual OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen-35B-A3B-GPTQ-Int4",
    "messages": [{"role": "user", "content": "Explain MoE routing briefly."}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

For BS=1 serving, use `--max-running-requests 1` and
`--cuda-graph-max-bs-decode 1`. The cache and Mamba budgets should be reduced
or increased together with the workload; the figures above are tied to the
listed configuration.

## Memory safety

On hosts with limited RAM, run model loading, server startup, CUDA extension
builds, and other heavy workloads under a process or cgroup memory guard.

Always set:

```bash
MAX_JOBS=1
```

FlashInfer and CUDA JIT compilation can otherwise start many compiler processes
at once. On a small host, that compiler fan-out can consume more memory than the
model itself.

Do not remove the guard or raise its limits merely to make a workload complete.

## Current limitations

When expert offload is enabled, the current path requires:

- CUDA and a single node with `TP=PP=EP=DP=1`;
- a supported GPTQ Int4 or ModelOpt FP4 MoE checkpoint;
- no LoRA, weight-cache mode, or batch-overlap mode;
- the Triton MoE runner for GPTQ Int4;
- breakable or disabled decode graphs, depending on the quantization path.

Host memory capacity, pinned-memory allocation, PCIe topology, prompt locality,
and the exact routing distribution all affect performance. Other models may
still import or start, but they are not covered by the Qwen 35B-A3B figures
above.

## Development

Focused tests for the new runtime pieces live under
`test/registered/unit/layers/moe/`, `test/registered/unit/model_loader/`, and
`test/registered/unit/models/`.

## Relationship to SGLang

MuxMoE is derived from [SGLang](https://github.com/sgl-project/sglang) and
retains the upstream serving architecture, API surface, and Apache-2.0 license.
The substantial downstream work is concentrated in MoE expert residency,
offload scheduling, hybrid-attention state management, quantized loading, and
small-batch CUDA Graph integration.

Issues and pull requests should describe the exact model format, cache budgets,
batch shape, CUDA version, and PCIe topology involved.

## License

Apache License 2.0. See [LICENSE](LICENSE).
