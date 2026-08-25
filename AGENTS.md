# SGLang 本地开发约定

本仓库是用于本地魔改和验证的 SGLang 副本。修改时优先保证
`Qwen3.5-4B-NVFP4` 在本机 RTX 5060 Ti（SM120）上的正确性和整机安全，
同时尽量保持与上游 SGLang 的结构和代码风格一致。

## 当前目标环境

- 仓库：`/home/dry/data/sglang`
- Conda 环境：`muxmoe`
- 本地 Python 源码：`/home/dry/data/sglang/python`
- 主要模型：`/home/dry/hdd2/models/Qwen3.5-4B-NVFP4`
- GPU：RTX 5060 Ti 16 GB，compute capability 12.0
- 主机内存：约 31 GiB RAM 和 8 GiB zram
- 系统 CUDA：`/usr/local/cuda`，当前为 CUDA 13.3
- Conda 编译器：GCC/G++ 15.2.0
- 系统默认 GCC：16.1.1，不受当前 CUDA 13.3 支持

环境版本可能变化。遇到环境相关问题时先重新核对实际版本，不要仅依赖本文
记录。

## 不可违反的内存安全规则

任何可能加载 SGLang 模型、初始化 GPU engine、触发 CUDA/FlashInfer JIT
或运行 GPU benchmark/test 的命令，都必须通过
`profile_output/ram_guard.py` 启动。

同时必须显式设置：

```bash
MAX_JOBS=1
```

禁止：

- 无看门狗启动 SGLang server、Engine、Runtime 或模型加载脚本；
- 无看门狗直接触发 FlashInfer、Torch CUDA extension 或 Ninja CUDA 构建；
- 把 `MAX_JOBS` 调到 2 或更高；
- 提高或绕过 28 GiB 整机 RAM 红线；
- 为了让任务通过而扩大 cgroup 内存上限；
- 同时运行多个 GPU/JIT 实验。

看门狗默认在整机有效 RAM 达到 26.5 GiB 时终止完整命令树，并额外设置
18 GiB `MemoryHigh`、20 GiB `MemoryMax` 和 1 GiB `MemorySwapMax`。
如果看门狗缺失、无法建立 cgroup 或保护机制报错，先修复保护机制，不能
直接运行目标命令。

安全启动模板：

```bash
./profile_output/ram_guard.py \
  --log profile_output/<run-name>.csv \
  --command-log profile_output/<run-name>.log \
  -- /home/dry/miniconda3/bin/conda run --no-capture-output -n muxmoe \
  env \
  MAX_JOBS=1 \
  PYTHONPATH=/home/dry/data/sglang/python \
  CUDA_VISIBLE_DEVICES=0 \
  /home/dry/miniconda3/envs/muxmoe/bin/python \
  -m sglang.launch_server \
  --model-path /home/dry/hdd2/models/Qwen3.5-4B-NVFP4
```

每次使用新的、可识别的 `<run-name>`，保留 CSV、命令日志和
`*.summary.json`。若看门狗或 cgroup 杀死任务，不要自动放宽限制重试；
先检查日志、峰值内存和 `memory.events`。

纯文本检索、Git 检查、静态分析和不会初始化 CUDA 的轻量单元测试不要求
套看门狗。不能确认某项测试是否会加载模型或触发 JIT 时，按需要看门狗处理。

## Conda 与编译器

必须通过 `conda activate muxmoe` 或 `conda run -n muxmoe` 获得环境变量。
仅直接执行 `/home/dry/miniconda3/envs/muxmoe/bin/python` 不等于激活环境，
会丢失 `CC` 和 `CXX`。

激活后应满足：

```text
CC=/home/dry/miniconda3/envs/muxmoe/bin/x86_64-conda-linux-gnu-cc
CXX=/home/dry/miniconda3/envs/muxmoe/bin/x86_64-conda-linux-gnu-c++
```

FlashInfer 会读取 `CC` 并向 NVCC 添加 `-ccbin "$CC"`。检查编译器版本时
使用 `"$CC" --version` 和 `"$CXX" --version`；裸 `gcc --version` 显示
系统 GCC 16 并不能代表 Conda 构建实际使用的编译器。

不要使用 `conda run -n muxmoe bash -lc ...`，本机登录 shell 配置可能重新
激活 `base`。优先直接把目标可执行文件交给 `conda run`。

除非用户明确要求，不要修改 Conda 环境、系统 CUDA、`site-packages` 或
系统编译器来实现框架功能。实现应落在本仓库源码中。

## 已知 OOM 根因

`Qwen3.5-4B-NVFP4` 的权重本身不是此前系统 OOM 的来源。权重加载完成后，
FlashInfer 为 SM120 编译 `fp4_gemm_cutlass_sm120`，在未设置 `MAX_JOBS`
时由 Ninja 并发启动约 14 个 `nvcc/cicc`；这些进程曾合计占用约 25 GiB
RSS 并耗尽 swap。

单个 `cicc` 也可能使用约 6 GiB RAM，所以本机固定使用 `MAX_JOBS=1`。
详细证据和基线见 `profile_output/OOM_REPORT.md`。不要删除
`~/.cache/flashinfer` 来“排除缓存问题”，除非用户明确授权；清缓存会重新
触发昂贵 JIT。

## 本地源码与现有改动

运行或测试时必须设置：

```bash
PYTHONPATH=/home/dry/data/sglang/python
```

必要时打印 `sglang.__file__`，确认导入的是本仓库源码而非已安装副本。

开始修改前执行 `git status --short` 和相关文件的 `git diff`。工作树中的
既有修改和未跟踪文件均视为用户资产，不得回退、覆盖或清理。

当前 `python/sglang/srt/models/qwen3_5.py` 有用户已有的 NVFP4 改动：

- packed loader 处理 `weight_scale_2`；
- `modelopt_fp4` quant config 传入 linear-attention/GatedDeltaNet；
- `modelopt_fp4` quant config 传入 full-attention projections。

这些改动已经证明可以完成目标 checkpoint 的权重加载，不是系统 OOM
来源。除非用户明确要求或新的证据证明实现有误，不得删除或绕过它们。

## 修改原则

- 先沿调用链确认真实根因，再改最小必要范围；
- 优先修改 `python/sglang` 下的本地实现，不给依赖包打临时补丁；
- 遵循相邻代码的抽象、命名、类型注解和日志风格；
- 不做与当前任务无关的重构、格式化或依赖升级；
- 不吞掉异常，不用宽泛 fallback 掩盖量化权重、shape 或 backend 错误；
- 量化相关修改必须核对参数名、shape、dtype、scale 语义、packed shard
  映射以及 tensor-parallel 行为；
- 不凭 checkpoint 文件名推断量化格式，以配置和实际 tensor metadata
  为准；
- 对行为变化补充聚焦的回归测试；测试不能依赖已安装的非本地 SGLang。

`docs_new/` 下另有更具体的 `AGENTS.md`；修改该目录时同时遵守其文档规则。

## 验证顺序

按成本从低到高验证：

1. 检查 diff，确认没有覆盖用户已有改动；
2. 对改动文件运行语法、静态检查和最小相关单元测试；
3. 验证配置解析、模型类选择、权重名称映射和量化方法绑定；
4. 只有确有必要时，运行带看门狗的模型加载或 GPU 测试；
5. 最后才运行端到端 server smoke test，并记录日志和内存峰值。

Python 代码遵循仓库的 Black、isort 和 Ruff 配置。不要对整个仓库运行会
自动修改文件的格式化命令；只处理本次改动涉及的文件。

交付时说明：

- 修改了哪些文件和行为；
- 运行了哪些验证；
- GPU/JIT 运行的看门狗日志及 RAM 峰值；
- 尚未验证的路径和剩余风险。
