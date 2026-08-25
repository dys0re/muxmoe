# BS=8 Domain Locality Decode Report

Date: 2026-08-02

## Current State

The local GPTQ Int4 MoE offload path now supports:

- direct resident execution for batched decode routes;
- batch-wide cross-layer expert prediction and adaptive prefetch sizing;
- fully device-resident shallow MoE layers;
- breakable decode CUDA graphs for dynamic expert-cache operations;
- tuned small-batch Triton MoE configurations and nearest-expert-count
  configuration selection.

Observed performance checkpoints are:

| Workload | Decode result |
| --- | ---: |
| BS=1, tuned cache/graph configuration | 17.89 token/s |
| BS=4, four uniform 64-token requests | 33.47 aggregate token/s |
| BS=8, related prompts | 66.51 aggregate token/s |
| BS=8, unrelated prompts | 45.04 aggregate token/s |

The BS=1 and BS=4 values use their earlier request-level benchmark harnesses.
The BS=8 values use the stricter streaming decode-only definition below, so
the rows document the current state but are not all directly comparable.

## BS=8 Configuration

Both cohorts used the same server configuration:

```text
model=/home/dry/hdd2/models/Qwen3.6-35B-A3B-GPTQ-Int4
quantization=moe_wna16
max_running_requests=8
max_mamba_cache_size=8
mem_fraction_static=0.75
moe_expert_offload_gb=12.5
moe_expert_cache_gb=3.0
moe_expert_resident_shallow_layers=6
moe_expert_cache_policy=lease
moe_expert_cache_lease=3
cuda_graph_backend_decode=breakable
cuda_graph_max_bs_decode=8
disable_radix_cache=true
```

Disabling the radix cache changes the hybrid Mamba state multiplier from five
states per request to one. This allowed BS=8 to use 0.53 GiB of Mamba state
while retaining a 2.612 GiB expert slab. Decode graphs were captured for batch
sizes 1, 2, 4, and 8.

## Method

- Each cohort contains eight concurrent requests.
- Related prompts cover eight LLM inference-system topics.
- Unrelated prompts cover eight mutually unrelated domains.
- Mean prompt length is 15.375 tokenizer tokens in both cohorts.
- Sampling uses temperature 0, `ignore_eos=true`, and exactly 128 output
  tokens per request.
- Each cohort runs once for warmup and three times for measurement.
- Each cohort uses a fresh server to prevent expert-cache state leaking across
  cohorts.
- Cohort decode timing starts after all eight requests have emitted their first
  token. Tokens emitted before this barrier and prefill/TTFT are excluded.

## Results

| Cohort | Warmup | Run 1 | Run 2 | Run 3 | Median aggregate | Median per request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Related | 65.80 | 67.20 | 66.51 | 66.07 | 66.51 | 8.31 |
| Unrelated | 45.32 | 45.04 | 44.28 | 45.57 | 45.04 | 5.63 |

Related prompts are 47.69% faster than unrelated prompts. Equivalently, the
unrelated cohort loses 32.29% of the related cohort's throughput.

## Expert-Cache Evidence

The following counters are taken at an equal 15,000 cache forwards:

| Counter | Related | Unrelated | Change |
| --- | ---: | ---: | ---: |
| Cache hit rate | 90.64% | 85.33% | -5.32 percentage points |
| Cache misses | 49,283 | 98,121 | +99.10% |
| Demand H2D | 74.45 GiB | 148.22 GiB | +99.10% |
| Prefetch H2D | 182.55 GiB | 324.85 GiB | +77.95% |
| D2D | 8.90 GiB | 10.38 GiB | +16.69% |
| Prefetch wasted | 30,717 | 48,740 | +58.67% |
| Prefetch match rate | 86.82% | 86.07% | -0.75 percentage points |

The prediction match rate is nearly unchanged. The dominant difference is the
larger batch-wide expert union for unrelated requests: it nearly doubles cache
misses and demand H2D traffic. This supports expert-locality-aware batching as
the next scheduling optimization before adding MTP or speculative decoding.

## Safety

Both servers and benchmark clients ran under `profile_output/ram_guard.py` with
`MAX_JOBS=1`. Peak system effective RAM was 21.23 GiB for the related server
and 21.30 GiB for the unrelated server, below the 26.5 GiB trip threshold.
There were no cgroup OOM kills. The summaries record `received signal 15`
because the servers were deliberately stopped after measurement, not because
the RAM threshold fired.

## Artifacts

- `bench_domain_locality_bs8.py`: streaming benchmark implementation.
- `gptq-bs8-domain-results-20260802.json`: related-prompt measurements.
- `gptq-bs8-unrelated-results-20260802.json`: unrelated-prompt measurements.
- `gptq-bs8-*-server-20260802.log`: server configuration and cache counters.
- `gptq-bs8-*.summary.json`: RAM guard summaries.
