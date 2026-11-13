# Profile 特定源码行对应 CUDA Kernel 的方法

以 `sglang/srt/models/deepseek_v4.py:1042` 的 `o, _ = self.wo_b(o.flatten(1))` 为例，记录如何在 SGLang 服务上定位一行 Python 代码对应的 CUDA kernel。

## 原理

SGLang 服务内置 `/start_profile`、`/stop_profile` HTTP 接口，底层使用 `torch.profiler.profile`。开启后每个 rank 会输出一份 Chrome trace（`*.trace.json.gz`）。定位某行 Python 代码对应的 kernel 关键在**注入 anchor**，把该行代码前后包在一段可识别的 profiler 区间里，然后在 trace 里筛出这个区间内的 GPU kernel。

## 关键坑：`torch.compile` / piecewise CUDA graph

DeepSeek-V4-Flash 启动带 `--piecewise-cuda-graph-compiler inductor`，对应的 forward 会被 inductor 编译。**编译区间内**的 `torch.profiler.record_function(...)` 不会产生事件。解决方式：**同时加 `torch.cuda.nvtx.range_push/pop`**，NVTX 是 CUDA runtime 级别的 range，能穿透编译边界，torch profiler 会把它记为 `user_annotation`/`gpu_user_annotation` 事件。

## 步骤

### 1. 在目标代码行加 anchor

在 `deepseek_v4.py:1042` 用 nvtx + record_function 双保险包一层：

```python
torch.cuda.nvtx.range_push("WO_B_ANCHOR_L1042")
with torch.profiler.record_function("WO_B_ANCHOR_L1042"):
    o, _ = self.wo_b(o.flatten(1))
torch.cuda.nvtx.range_pop()
```

改完记得 profile 结束后**恢复原状**。

### 2. 启动服务并声明 profiler 输出目录

```bash
export SGLANG_TORCH_PROFILER_DIR=/some/path/profile_wo_b
mkdir -p "$SGLANG_TORCH_PROFILER_DIR"
sh deploy_pcg.sh > server.log 2>&1 &
```

等到日志出现 `The server is fired up and ready to roll!`。

### 3. 发送目标负载并同步抓取 profile

用 `input_ids` 直接控制 token 数（避免 tokenizer 展开导致数量对不上）。核心步骤：`POST /start_profile` → `POST /generate` → `POST /stop_profile`。

```python
import json, random, urllib.request

HOST = "http://127.0.0.1:30101"
PROFILE_DIR = "/some/path/profile_wo_b"

def post(path, payload=None, timeout=1800):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(HOST + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout).read().decode(errors="replace")

random.seed(0)
input_ids = [random.randint(10, 129000) for _ in range(64000)]

post("/start_profile", {
    "output_dir": PROFILE_DIR,
    "activities": ["CPU", "GPU"],
    "with_stack": False,
    "record_shapes": True,
})
post("/generate", {
    "input_ids": input_ids,
    "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
})
post("/stop_profile")
```

抓完每个 rank 会输出一个 `<ts>-TP-{i}-DP-{i}-EP-{i}.trace.json.gz`。**只有承接该请求的 DP rank 里会有非空 anchor 事件**（其他 rank 是 IDLE），可以按文件大小快速判断。

### 4. 从 trace 找 anchor 区间内的 kernel

思路：
- anchor 事件按 `cat=user_annotation`（CPU 侧）和 `cat=gpu_user_annotation`（GPU 侧）出现。
- **首选 CPU-correlation 方法**：CPU-anchor 覆盖的时间区间里，`cpu_op` 事件的 `args["External id"]` 就是 correlation id；GPU 侧 `kernel` 事件的 `args["External id"]` 相同即为对应 kernel。这种方法**精确对应**，不会漏也不会混入邻居。
- **兜底 GPU 时间窗方法**：直接按时间过滤 GPU-anchor 区间内的 `kernel` 事件。用于 CPU-correlation 全为 0 的情况（见下方"CUDA graph 泄漏"）。
- 两种方式互相验证。

#### CUDA graph / inductor 对 anchor 的影响（重要）

当目标代码位于 `--piecewise-cuda-graph-compiler inductor` 编译的静态图内（例如 `fused_rope_inplace` 这种被 inductor 吞进去的自定义 op）时：

1. **CPU-correlation 会返回 0 个 kernel**。原因是 `record_function` 被 inductor 编译边界吞掉，anchor 内看不到关联到 GPU kernel 的 `cpu_op`。此时只能退回 GPU 时间窗方法。
2. **GPU 时间窗会把 stream 上相邻的 kernel 也框进来**。`nvtx.range_push/pop` 在 CUDA graph replay 时的 GPU 时间点由 graph 排布决定，push/pop 之间会包含前后邻居 op 的 kernel。观察上：GPU anchor 时长会被拉长到几十 µs（远大于目标 kernel 本身几 µs）。
3. **判据 —— 用出现次数过滤**：一次 forward 只会调用 anchor 内的代码一次，所以真正对应的 kernel 出现次数应精确等于 anchor 次数（例如 705 次 anchor → 目标 kernel 恰好 705 次）；**数量远小于 anchor 次数的条目属于泄漏**（例如某个 GEMM 只出现 22/9 次而非 705 次，说明只在部分 graph replay 窗口里被"蹭"进来），应剔除。

分析脚本骨架：

```python
import gzip, json, bisect
from collections import Counter

ANCHOR = "WO_B_ANCHOR_L1042"

with gzip.open("<rank-TP-1>.trace.json.gz", "rt") as f:
    evs = json.load(f).get("traceEvents", [])

cpu_anchors = [(e["ts"], e["ts"]+e["dur"], e["pid"])
               for e in evs
               if e.get("cat")=="user_annotation" and e.get("name")==ANCHOR and e.get("dur",0)>0]
gpu_anchors = [(e["ts"], e["ts"]+e.get("dur",0))
               for e in evs
               if e.get("cat")=="gpu_user_annotation" and e.get("name")==ANCHOR and e.get("dur",0)>0]
N = len(cpu_anchors)
print(f"anchor count = {N}")

# 首选：CPU-correlation
corr = set()
for e in evs:
    if e.get("cat") != "cpu_op": continue
    ts = e.get("ts")
    for s, t, pid in cpu_anchors:
        if s <= ts <= t and e.get("pid") == pid:
            cid = (e.get("args") or {}).get("External id")
            if cid is not None: corr.add(cid)
            break

kernels_corr = Counter()
for e in evs:
    if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset"):
        cid = (e.get("args") or {}).get("External id")
        if cid in corr:
            kernels_corr[e["name"]] += 1

# 兜底：GPU 时间窗（当 kernels_corr 为空且代码在 CUDA graph 里时使用）
starts = sorted(s for s,_ in gpu_anchors)
ends_by_s = {s:t for s,t in gpu_anchors}
kernels_win = Counter()
for e in evs:
    if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset"):
        ts = e.get("ts")
        idx = bisect.bisect_right(starts, ts) - 1
        if idx >= 0 and starts[idx] <= ts <= ends_by_s[starts[idx]]:
            kernels_win[e["name"]] += 1

use = kernels_corr if kernels_corr else kernels_win
for name, c in use.most_common():
    # 出现次数 << N 的很可能是 CUDA graph 时间窗泄漏，人工剔除
    tag = "" if c >= N * 0.9 else "  <-- likely leak from neighbor"
    print(f"  {c:6d}  {name[:180]}{tag}")
```

### 5. 解读结果

对 `wo_b`（RowParallelLinear + FP8 block-scale），一次调用得到 3 个 kernel：

- `at::native::elementwise_kernel<... direct_copy_kernel_cuda ...>` — `o.flatten(1)` 需要 contiguous 触发的 bf16 拷贝
- `per_token_group_quant_8bit_kernel<..., __nv_bfloat16, Float8_e4m3fn, ...>` — `sgl_kernel::sgl_per_token_group_quant_8bit_v2`，输入 FP8 逐 128-组量化
- `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<..., (deep_gemm::GemmType)0, e4m3_t, e4m3_t, bfloat16_t, EpilogueIdentity>` — `sglang::deep_gemm_fp8_fp8_bf16_nt`，主 FP8 GEMM。DeepGEMM 会按 M 维度 JIT 出多个 `BLOCK_M` 实例（例如 prefill M=4096 用 `BLOCK_M=128`；小 M 走 `BLOCK_M=192` 或 `16`）

若 `attn_tp_size>1`（未启用 DP-attention），还会多一个 NCCL all-reduce kernel。

#### 案例：不同类型代码行的观察

以 `deepseek_v4.py` 里几段代码为例（`--piecewise-cuda-graph-compiler inductor` + `--enable-dp-attention` + 64k 输入 → 705 次 anchor 调用）：

- **纯 metadata（无 kernel）**：`o.view(...)` — CPU anchor 705、GPU anchor **0**、kernel **0 个**。`view` 只改 stride/shape，不动 storage，不 launch kernel。
- **单 kernel 融合 op（CPU-correlation 失效，走 GPU 时间窗+次数过滤）**：`fused_rope_inplace(...)` — CPU-correlation 0 个（被 inductor 吞掉）；GPU 时间窗看到 `deepseek_rope_kernel` × **705**（=anchor 次数，即真正对应），另有 `nvjet_sm100_tst_*` × 22 和 × 9（<< 705，是相邻 GEMM 泄漏进来的，剔除）。**结论：1 个 kernel**。
- **batched GEMM（CPU-correlation 正常）**：`torch.einsum("tgd,grd->tgr", o, wo_a)` — CPU-correlation 命中 705 次，全部落到同一族 `nvjet_sm100_tst_*` cuBLASLt kernel 的三种 tile 实例（`128x256…2x1_v`×660、`128x256…2x2_h`×44、`64x8_64x16…v`×1，合计 705）。这三个都是同一 GEMM 算子按 M 维形状挑不同 tile 的结果，语义上仍是 **1 个 kernel（1 次 batched-GEMM launch/调用）**，只是 cuBLASLt heuristic 在不同 shape 下选了不同模板。

**cuBLASLt kernel 名字解读（`nvjet_sm100_tst_<Mtile>x<Ntile>_<warp>_<cluster>_[2cta]_<split>_bz_TNT`）**：
- `sm100` = Blackwell / B200；`tst` = Tensor Core 模板
- `128x256` / `64x8` = CTA tile (M×N)；`64x6` / `64x16` = warp/K pipeline
- `2x1` / `2x2` / `4x1` = CTA cluster；`2cta` = Blackwell dual-CTA MMA
- `v` / `h` = vertical / horizontal split-K
- `bz` = batched（batch>1）；`TNT` = A/B/C 转置状态

即使 kernel 名字不同，只要 correlation 落在同一 anchor 上，就是**同一个语义算子的多个模板实例化**，不是"多个 kernel"。

## 常见问题

- **anchor 事件为空**：多半是 record_function 被 inductor/torch.compile 吃掉；加上 `torch.cuda.nvtx.range_push/pop` 再抓一次。
- **CPU-correlation 得到 0 个 kernel，但 GPU anchor 有事件**：代码在 CUDA graph 里，correlation 链断了；改用 GPU 时间窗方法 + 次数过滤（见 §4 判据）。
- **GPU 时间窗混入不相关的 GEMM/邻居 kernel**：CUDA graph replay 会让 nvtx 时间窗覆盖前后邻居；按"次数 ≈ anchor 次数"过滤，剔除远少于 anchor 次数的条目。
- **anchor 只在部分 rank 出现**：单请求只走一个 DP rank，属于正常，看文件大小挑那个 rank 分析即可。
- **`prompt_tokens` 与预期不符**：走 `text` 字段会经过 tokenizer 且可能被 auto-truncate；用 `input_ids` 直接给 token 数最稳。
- **服务重启慢**：DeepGEMM JIT 预编译 + CUDA graph 抓取会几分钟；可先 `python -m sglang.compile_deep_gemm ...` 预热。
- **profile 文件过大**：`with_stack=False`、缩短窗口、只保留感兴趣 rank 的 trace。

## 清理

Profile 结束后：
1. 恢复源码里为 anchor 加的 nvtx / record_function
2. 删除多余 trace 文件、临时脚本
3. 需要时 `pkill -9 -f sglang` 关掉服务
