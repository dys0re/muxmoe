# Qwen3.6-35B-A3B BS=1 MTP report

## Result

BS=1 bundled MTP now starts and completes repeated generation requests on the
RTX 5060 Ti with GPTQ Int4 target weights, BF16 MTP weights, expert offload,
GDN state rollback/commit, and breakable decode CUDA graphs enabled.

The validated target model is:

```text
/home/dry/hdd2/models/Qwen3.6-35B-A3B-GPTQ-Int4
```

Both tested linear-chain configurations completed three 128-token greedy
generations without a scheduler exception:

| Configuration | Median decode TPS | Accept length | Accept rate |
|---|---:|---:|---:|
| Non-MTP tuned baseline | 17.89 | 1.00 | - |
| MTP-2: 1 draft step, 2 verify tokens | 12.91 | 1.82-1.93 | 0.82-0.93 |
| MTP-3: 2 draft steps, 3 verify tokens | 14.12 | 2.55-2.77 | 0.78-0.89 |

MTP-3 is 21.1% slower than the tuned non-MTP baseline. MTP-2 is 27.8%
slower. The implementation is functional, but MTP is not a performance win
with the current 16 GiB VRAM budget: the BF16 draft layer reduces the target
expert cache from 3.0 GiB to 1.5 GiB, and wider target verification activates
and transfers more unique experts per forward.

MTP-2 reproduced the historical 128-token greedy baseline text exactly.
MTP-3 was deterministic across its three runs but diverged from the baseline
at a later greedy choice. The target verify path changes MoE/GDN execution
shape from one token to three tokens, so floating-point kernel ordering can
change a close argmax. This remains a quality-validation consideration for
wider MTP chains.

## Fixes

The implementation required four independent fixes:

1. Detect GPTQ dynamic negative selectors that exclude `mtp.*` and construct
   the bundled MTP module in BF16 instead of allocating GPTQ-packed shapes.
2. Share the target embedding and LM head before target KV-pool sizing. This
   reduced the minimum viable `mem_fraction_static` from 0.8725 to 0.7487 by
   releasing about 2 GiB of duplicate vocabulary parameters before profiling.
3. Size breakable CUDA graph output buffers from the warmup output row count.
   The graph key is request batch size (`1`), while target verify returns
   `bs * draft_token_num` rows (`3`); the previous buffer silently truncated
   logits and hidden states to one row.
4. Restore `MambaPool.get_speculative_mamba2_params_all_layers()`. An earlier
   state-adapter change had replaced the method declaration and left its body
   unreachable, preventing GDN state commit after verification.

Expert offload plus speculative decoding remains restricted to the validated
case: BS=1, Qwen3.5-family bundled NEXTN/EAGLE, top-k 1, a single MTP layer,
and the same checkpoint path for target and draft.

## Validated configuration

Common server settings:

```text
MAX_JOBS=1
PYTHONPATH=/home/dry/data/sglang/python
--quantization moe_wna16
--moe-expert-offload-gb 12.5
--moe-expert-cache-gb 1.5
--moe-expert-resident-shallow-layers 6
--max-running-requests 1
--max-mamba-cache-size 1
--max-total-tokens 1024
--disable-radix-cache
--mem-fraction-static 0.85
--cuda-graph-backend-decode breakable
--cuda-graph-max-bs-decode 1
--speculative-algorithm NEXTN
--speculative-draft-model-path <target model path>
--speculative-draft-model-quantization unquant
--speculative-eagle-topk 1
```

MTP-3 used `--speculative-num-steps 2 --speculative-num-draft-tokens 3`.
MTP-2 used `--speculative-num-steps 1 --speculative-num-draft-tokens 2`.

## Verification

CPU-only focused tests:

```text
33 passed
```

Covered MTP quantization selection, expert-offload argument validation,
breakable graph output sizing, speculative Mamba pool access, and expert cache
behavior.

GPU/RAM-guard artifacts:

| Run | Benchmark | RAM summary | Peak effective RAM | cgroup OOM kills |
|---|---|---|---:|---:|
| MTP-3 v8 | `gptq-bs1-mtp-v8-benchmark-20260803.json` | `gptq-bs1-mtp-v8-server-20260803.summary.json` | 20.153 GiB | 0 |
| MTP-2 v9 | `gptq-bs1-mtp2-v9-benchmark-20260803.json` | `gptq-bs1-mtp2-v9-server-20260803.summary.json` | 20.144 GiB | 0 |

Both servers were deliberately stopped with SIGINT after the benchmarks. The
RAM summaries therefore record signal 2 as the trigger and return code -9;
neither run crossed the 26.5 GiB system trip point or incurred a cgroup OOM.

Ruff and Black were not installed in the `muxmoe` environment. `git diff
--check`, Python compilation, and the focused pytest suite passed.
