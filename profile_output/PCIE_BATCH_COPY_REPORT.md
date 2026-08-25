# PCIe Batched Expert Copy Report

Date: 2026-08-04

## Scope

This experiment targets BS=1, non-MTP decode for
`Qwen3.6-35B-A3B-GPTQ-Int4`. It reduces host submission overhead by replacing
groups of independent expert tensor copies with CUDA 12.8+
`cudaMemcpyBatchAsync` submissions. Dependent staging-to-resident copies remain
in a separate batch because CUDA does not define ordering between operations in
one memcpy batch.

The implementation covers demand resident fills, compact staging population and
admission, and cross-layer prefetch. It falls back to individual PyTorch copies
when the CUDA API is unavailable or a tensor is incompatible. Only prefetch uses
`cudaMemcpyFlagPreferOverlapWithCompute`; the demand path uses default memcpy
attributes because the payload microbenchmark showed a GPU-time regression from
the overlap hint.

## Validation

- Expert-cache unit tests: 26 passed.
- `git diff --check`: passed.
- Python compilation for the changed source, tests, and benchmark: passed.
- CUDA API correctness probe: `cudaSuccess`, copied data correct, 4.0753 GiB
  peak effective system RAM, zero cgroup OOM kills.
- End-to-end server: zero batch-copy fallbacks and zero cgroup OOM kills.

## Exact Payload Microbenchmark

The benchmark uses the same fragmented GPTQ expert payload shapes as the target
model. Medians are:

| Submission mode | Host submit time | GPU elapsed time | Host delta | GPU delta |
| --- | ---: | ---: | ---: | ---: |
| Individual copies | 368.154 us | 1331.232 us | - | - |
| Batched, default attributes | 301.808 us | 1320.352 us | -18.0% | -0.8% |
| Batched, prefer-overlap | 241.483 us | 1359.856 us | -34.4% | +2.2% |

This isolates the benefit: batching materially reduces CPU submission time but
does not increase the physical DMA rate by a comparable amount.

## End-to-End Result

Both runs use 12.5 GiB host expert offload, a 3.0 GiB GPU expert cache, six
shallow resident MoE layers, lease 3, exact routing, breakable BS=1 decode graph,
and three deterministic 128-token generations.

| Variant | Median decode | Delta |
| --- | ---: | ---: |
| Previous tuned baseline | 17.8867 tok/s | - |
| Batched expert copies | 18.0528 tok/s | +0.93% |
| Full-width cross-layer prefetch (opt-in) | 18.3213 tok/s | +2.43% |

The three new runs were 16.6165, 18.0528, and 18.0783 tok/s. The cold first run
is retained in the result. Cumulative logging observed 12,822 batch submissions
covering 106,516 copy operations, with `copy_fallback=0`.

The full-width prefetch runs were 18.3407, 18.3213, and 18.1173 tok/s. This is
1.49% faster than batched copies with the previous adaptive minimum and 2.43%
faster than the tuned pre-batching baseline. The implementation exposes
`--moe-expert-cache-prefetch-min-ratio` in `(0, 1]`. The default remains 0.5
because full-width prefetch trades lower transfer efficiency for higher decode
throughput; 1.0 enables that tradeoff explicitly.

At 12,000 cumulative offloaded-layer forwards, full-width prefetch changed the
traffic and cache counters as follows:

| Counter | Adaptive minimum | Full-width |
| --- | ---: | ---: |
| Demand H2D | 21.745 GiB | 14.019 GiB |
| Prefetch H2D | 14.513 GiB | 27.838 GiB |
| Total H2D | 36.258 GiB | 41.857 GiB |
| Immediately useful H2D | 34.424 GiB | 34.277 GiB |
| Immediate transfer efficiency | 94.94% | 81.89% |
| Prefetch selected | 56,668 | 91,576 |
| Prefetch useful | 8,393 | 13,410 |
| Prefetch wasted | 1,214 | 5,018 |

Demand traffic fell 35.5%, while total traffic increased 15.4%. Immediately
useful traffic fell 0.43%, and immediate transfer efficiency fell by 13.05
percentage points. The prefetch admission useful rate also fell from 87.4% to
72.8%. Full-width prefetch therefore improves latency by spending idle PCIe
capacity on speculative transfers; it does not improve effective PCIe
utilization and is not the default.

## PCIe Observation

A separate already-loaded 128-token request took 7.080 seconds. During its
decode-active samples, `nvidia-smi dmon` reported GPU PCIe RX values of 4776,
1719, 1464, 1208, 3851, and 1036 MB/s:

- active-sample mean: 2342 MB/s;
- observed peak: 4776 MB/s;
- expert-cache counters imply approximately 1.66 GB/s average payload traffic
  over the complete request;
- relative to a PCIe Gen3 x8 payload ceiling of about 7.88 GB/s, the request-wide
  effective utilization is approximately 21%, the active-sample mean is about
  30%, and the observed one-second peak is about 61%.

The PCIe traffic is therefore bursty rather than bandwidth-saturated. Each MoE
layer issues short transfers and then waits on routing, synchronization, and
compute. Reducing API submission overhead yields only a small end-to-end gain;
the larger opportunity is to create a longer pipeline across layer boundaries,
increase useful prefetch distance, or coalesce expert storage into larger
contiguous transfers without increasing bytes moved.

With full-width prefetch, the complete benchmark's 22 active one-second samples
averaged 2361 MB/s and peaked at 4364 MB/s. A separate already-loaded request
took 6.980 seconds; its six active samples averaged 2024 MB/s and peaked at
2999 MB/s. Instantaneous DMA throughput did not increase. The throughput gain
comes from filling more inter-layer PCIe idle periods and reducing demand-side
waits, at the cost of lower useful-byte efficiency.

## Artifacts

- `profile_output/bench_pcie_expert_copy.py`
- `profile_output/pcie-expert-copy-benchmark-v2-20260804.json`
- `profile_output/pcie-expert-copy-bench-v2-20260804.summary.json`
- `profile_output/pcie-batch-api-probe-20260804.summary.json`
- `profile_output/pcie-batch-e2e-benchmark-20260804.json`
- `profile_output/pcie-batch-e2e-bench-20260804.summary.json`
- `profile_output/pcie-batch-e2e-server-20260804.summary.json`
- `profile_output/pcie-batch-decode-only-dmon-20260804.log`
- `profile_output/pcie-batch-decode-only-20260804.summary.json`
- `profile_output/pcie-prefetch-full-benchmark-20260804.json`
- `profile_output/pcie-prefetch-full-bench-20260804.summary.json`
- `profile_output/pcie-prefetch-full-dmon-20260804.log`
- `profile_output/pcie-prefetch-full-decode-only-dmon-20260804.log`
- `profile_output/pcie-prefetch-full-decode-only-20260804.summary.json`
- `profile_output/pcie-prefetch-full-server-20260804.summary.json`

The server reached 19.7076 GiB peak effective system RAM and 18.0022 GiB peak
cgroup memory. It was terminated explicitly after measurement; the summary's
signal-triggered status is not an OOM (`oom_kill=0`).

The full-width server reached 20.7057 GiB peak effective system RAM and 18.0019
GiB peak cgroup memory. It was also terminated explicitly after measurement and
recorded zero cgroup OOM kills.
