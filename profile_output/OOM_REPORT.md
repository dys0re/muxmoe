# Qwen3.5-4B-NVFP4 主存 OOM 排查报告

## 结论

系统 OOM 不是由 4B NVFP4 权重加载本身造成的，而是权重加载完成后的
FlashInfer SM120 FP4 GEMM JIT 编译造成的。

`flashinfer/jit/cpp_ext.py::_get_num_workers()` 仅在环境变量
`MAX_JOBS` 为数字时才向 Ninja 传入 `-j`。当前环境未设置 `MAX_JOBS`，
因此 Ninja 在 12 线程 CPU 上并行启动了 14 个 `nvcc/cicc`。目标
`fp4_gemm_cutlass_sm120` 含 17 个 CUDA 编译单元，每个 `cicc` 可占用约
2–6 GiB 主存，足以在数十秒内耗尽 31 GiB RAM 和 8 GiB zram。

## 直接证据

### 上一启动周期的内核 OOM 记录

内核 OOM task table 在第一次 OOM 时记录到：

| 指标 | 数值 |
|---|---:|
| 同时存活的 `cicc` | 14 |
| 这些 `cicc` 的合计 RSS | 25.351 GiB |
| 这些 `cicc` 的合计 swap | 1.714 GiB |
| 全系统剩余 swap | 196 KiB |
| OOM killer 杀死对象 | `cicc`, PID 84472 |

31 秒后再次 OOM：

| 指标 | 数值 |
|---|---:|
| 同时存活的 `cicc` | 13 |
| 这些 `cicc` 的合计 RSS | 25.389 GiB |
| 全系统剩余 swap | 28 KiB |
| OOM killer 杀死对象 | `cicc`, PID 84604 |

更前一个启动周期连续发生 6 次同类 OOM；`cicc` 数从 13 降到 8，
但其合计 RSS 始终约为 24.46–25.20 GiB。

### JIT 构建图

本机缓存中的：

`~/.cache/flashinfer/0.6.15.post1/120f/cached_ops/fp4_gemm_cutlass_sm120/build.ninja`

包含 17 个 `cuda_compile` 目标。此前失败构建的 `.ninja_log` 只完成了
入口对象，未生成最终 `.so`，所以后续每次启动仍会重新触发大规模编译。

### 权重加载实测

在 20 GiB cgroup 和 26.5 GiB 整机看门狗下运行本地 SGLang 副本：

| 阶段/指标 | 数值 |
|---|---:|
| 权重加载耗时 | 26.14 s |
| SGLang 报告的 GPU 权重增量 | 3.82 GB |
| 整次运行的整机有效 RAM 峰值 | 8.184 GiB |
| 整次运行的 scope 内存峰值 | 9.205 GiB |
| cgroup OOM | 0 |

日志已经出现 `Load weight end`，随后才进入 FlashInfer autotune/JIT。
本次因系统默认 GCC 16 不受 CUDA 13.3 支持而提前报错，尚未进入高内存的
FP4 GEMM 编译。

## 隔离复现

直接触发 `get_gemm_sm120_module_cutlass_fp4()`，使用 GCC 15，并保留默认
Ninja 并行度：

| 指标 | 数值 |
|---|---:|
| cgroup 硬上限 | 14.0 GiB |
| 撞到硬上限所需时间 | 22.7 s |
| cgroup `oom` / `oom_kill` | 6 / 4 |
| 整机有效 RAM 峰值 | 17.394 GiB |
| 看门狗绝对上限 | 28.0 GiB |

这在不危及整机的情况下复现了同一故障。

对照组设置 `MAX_JOBS=1`：

| 指标 | 数值 |
|---|---:|
| 同时运行的 `cicc` | 1 |
| 120 s 测试窗内 scope 峰值 | 6.630 GiB |
| 整机有效 RAM 峰值 | 9.886 GiB |
| cgroup OOM | 0 |

单个编译单元尚能增长到约 6 GiB，因此这台 32 GiB 主机应使用
`MAX_JOBS=1`，不建议用 2。

## 第二个环境问题

当前 `/usr/bin/gcc` 是 GCC 16.1.1；CUDA 13.3 的 Linux host compiler
支持范围止于 GCC 15。默认启动会先在 FlashInfer attention JIT 处报：

`unsupported GNU version! gcc versions later than 15 are not supported`

本机已有 `/usr/bin/gcc-15`、`/usr/bin/g++-15`，muxmoe 环境也包含
GCC 15.2 工具链。

## 推荐的安全启动方式

所有后续运行都通过看门狗，并固定单路 JIT 和 GCC 15：

```bash
./profile_output/ram_guard.py \
  --log profile_output/qwen35_guarded.csv \
  --command-log profile_output/qwen35_guarded.log \
  -- env \
  MAX_JOBS=1 \
  CC=/usr/bin/gcc-15 \
  CXX=/usr/bin/g++-15 \
  NVCC_CCBIN=/usr/bin/g++-15 \
  PYTHONPATH=/home/dry/data/sglang/python \
  CUDA_VISIBLE_DEVICES=0 \
  /home/dry/miniconda3/envs/muxmoe/bin/python \
  -m sglang.launch_server \
  --model-path /home/dry/hdd2/models/Qwen3.5-4B-NVFP4
```

看门狗默认提供两层保护：

- 整个命令树的 cgroup：`MemoryHigh=18 GiB`、`MemoryMax=20 GiB`、
  `MemorySwapMax=1 GiB`。
- 整机有效 RAM 每 100 ms 采样；26.5 GiB 时杀死整个 scope，绝对红线
  为 28 GiB。

## 本地代码改动判断

工作树原有的 `python/sglang/srt/models/qwen3_5.py` 改动：

- 让 Qwen3.5 attention/GDN projection 使用 `modelopt_fp4`；
- 给 fused projection loader 增加 `weight_scale_2`。

本次实测证明该版本可以完成 NVFP4 权重加载，且 GPU 权重增量只有
3.82 GB。这些改动不是系统 OOM 的来源；本次排查未修改该文件。

## 剖析仪表与产物

| 文件 | 类型 | 内容 |
|---|---|---|
| `profile_output/ram_guard.py` | 新建 | cgroup 硬限制、整机 RAM 看门狗、100 ms CSV 采样 |
| `profile_output/load_modified.{csv,log,summary.json}` | 生成 | 完整 SGLang 权重加载与失败点 |
| `profile_output/fp4_jit_unbounded.{csv,log,summary.json}` | 生成 | 未限制 Ninja 并行度的隔离 OOM |
| `profile_output/fp4_jit_max_jobs_1.{csv,log,summary.json}` | 生成 | `MAX_JOBS=1` 对照 |
| `profile_output/env_import.{csv,summary.json}` | 生成 | muxmoe/SGLang/依赖版本与导入内存 |
| `profile_output/watchdog_smoke.{csv,summary.json}` | 生成 | 看门狗正常路径冒烟测试 |
| `profile_output/watchdog_hard_limit.{csv,summary.json}` | 生成 | 低阈值保护测试 |
| `profile_output/__pycache__/ram_guard.*.pyc` | 生成 | Python 语法检查缓存 |
| `~/.cache/flashinfer/0.6.15.post1/120f/cached_ops/fp4_gemm_cutlass_sm120/` | JIT 缓存更新 | FlashInfer 自动生成的 Ninja 文件和部分对象缓存 |

