# SAGE 代码梳理（Protocol SAGE-v1.0）

本文把论文第 3、4 节映射到代码模块，作为后续实现的蓝图。所有数值参数以
`configs/sage_v1.0.json` 为准（论文 4.4 节说明 JSON 为权威配置）。

## 1. 整体流程

```
离线（不消耗目标查询）                在线（每张生成图，一个查询计数器）
───────────────────────────          ──────────────────────────────────────────────
dev 集 real / fake 图像               x_o ──► ASA: 坐标下降求 α* ──► x_spec = T(x_o; α*)
   │                                            │                    │
   ▼                                            ▼                    ▼ 1 次查询
SAP: 平均功率谱 P̄_r, P̄_f            a(u,v)=mean_c|α*·ΔP_c|      D(x_spec)=Real 且 LPIPS≤δ ? → 成功返回
   ΔP_c = P̄_r,c − P̄_f,c              W = (1-ζ)a/Σa + ζ/M                 │否
   （保存为 .npz，冻结）                       │                    ▼
                                               │         Real 但超失真 → 作为起点
                                               │         Fake → 参考池逐个验证 + 二分(x_o↔x_ref)
                                               ▼                    │
                                        SBQR: 谱加权随机方向 + 正交投影 + 向 x_o 收缩
                                        本地拒绝不改善 L2 的候选 → 查询 → 仅接受 Real
                                        每 50 个提议自适应 γ, λ；归档所有 Real 候选
                                               ▼
                                        返回 LPIPS≤δ 中 L2 最小者，或失败+诊断
```

## 2. 目录结构

已实现，见 [README.md](../README.md) 的 Layout 一节。与最初计划的差异：

- 基线 BA / HSJA / QEBA-S 在 `sage/baselines.py` 中自行实现（不依赖 foolbox），以便所有方法
  共用同一个 `QueryOracle`（预算、缓存、检查点完全一致）；默认参数照搬参考实现并写在文件头。
- 所有攻击写成 Python generator，由 `sage/query.py:run_tasks` 把多个攻击的待查询图像拼成一个
  batch 送进检测器（单卡 A100 上同时跑 8–32 个攻击），不改变任何单次攻击的查询计数。
- 第三方代码（DIRE、UFD 的 CLIP）按文件路径加载，避免两个仓库都有 `networks` 包导致冲突。

## 3. 关键实现要点（容易出错的地方）

1. **FFT 约定**：`torch.fft.fft2(x, norm="ortho")`，不 shift；频率用 `torch.fft.fftfreq`；
   ρ 的 8 个等宽带覆盖 `[0, √0.5]`，最后一带包含外端点；DC 不改，最后原样写回。
2. **共轭对称**：校正只改幅度、保留相位，径向掩码天然对称；逆变换后取实部，
   并断言 `imag.abs().max()` 很小（记录数值误差）。SBQR 的随机谱可直接用
   `irfft2` / 或对实高斯噪声做 FFT 再乘 √W，天然满足共轭对称 + 自共轭点为实数。
3. **ASA 目标**：每次网格评估都要完整重建 + 一次 LPIPS；8 带 × 11 值 × ≤3 轮 ≈ 264 次
   LPIPS 前向，可在 GPU 上把 11 个候选 batch 起来。平局取较小 α。
4. **查询计数**：所有检测器调用必须经过 `QueryCounter`；同一张图同一次运行内用
   图像字节哈希做标签缓存，缓存命中不计数；ASA 验证、参考验证、二分、细化共用 5000 预算。
5. **DIRE 每次查询都要对攻击后的 RGB 重新做反演+重建**，不能复用干净图的 DIRE 图；
   重建随机性固定（DDIM 反演本身确定性，再固定 seed / 关闭 cudnn 非确定算法）。
6. **成功判定**：`D(x)=Real` 且 `LPIPS(x, x_o) ≤ 0.05`；在 500/1k/2k/5k 检查点记录，
   首次成功即停并向后继承。
7. **基线公平性**：同一参考池、同一参考顺序、同一查询包装器、同一停止模式。

## 4. 计算量预估（先在验证集 pilot 上实测）

- UFD 一次查询 ≈ 一次 CLIP ViT-L/14 前向，A100 上 batch=1 约几毫秒级。
- DIRE 一次查询 = 20 步 DDIM 反演 + 20 步 DDIM 采样（256² ADM UNet，fp16）≈ 40 次 UNet 前向，
  单张大约 0.5–1 s 级。最坏情况 5000 查询 ≈ 1–1.5 GPU·h / run。
  实验块 1 的 DIRE 部分最多 4 方法 × 400 图 × 3 seeds = 4800 run，**上界数千 GPU·h**。
  - 实际会被首次成功早停、eligibility 大幅压低，但务必先跑论文 4.4 的 40 张验证集 pilot
    测 wall time 再提交大规模作业。
  - 优化：SBQR / BA / HSJA 的查询本质是串行的，但可以**多张图并行成 batch**
    （每张图各自状态机，一次把多个候选拼成 batch 送进 DIRE），这是 DIRE 上最主要的提速手段。

## 5. 实现顺序建议

1. `spectral.py` + 单元测试（T(x;0)=x、实部误差、频带覆盖、DC 不变）
2. 检测器适配器 `ufd.py` / `dire.py`，在 val 集复现干净准确率，冻结阈值（0.5）
3. `build_manifest.py` 跑通，`sap.py` 生成 prior
4. `query.py` + `init.py` + `sbqr.py` + `asa.py` → `attack.py`
5. 基线接入共享包装器
6. 40 张 pilot → 测时间 → 锁定 execution lock（哈希、commit、版本）→ 正式实验
