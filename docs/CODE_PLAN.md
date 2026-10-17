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

## 2. 目录结构（计划）

```
SAGE/
├── configs/sage_v1.0.json          # 冻结的协议参数（论文 4.4）
├── envs/requirements.txt           # pip 依赖（torch 单独装）
├── scripts/
│   ├── umhpc/
│   │   ├── setup_env.sh            # 登录节点：建 conda 环境、克隆第三方仓库、预下载权重
│   │   ├── env.sh                  # 每个作业 source：环境变量、缓存路径
│   │   └── check_env.sbatch        # 在 A100 节点上做 GPU/依赖/检测器自检
│   └── data/
│       ├── build_manifest.py       # 扫描→哈希→去重→划分(seed 20261008)→预处理到 256 画布
│       └── verify_manifest.py      # 校验 manifest 与磁盘文件一致
├── sage/
│   ├── spectral.py                 # 酉 FFT、径向频带 Ω_b、DC 处理、共轭对称、T(x;α)   (Eq.2,4–6)
│   ├── sap.py                      # 由 dev 集计算 P̄_r, P̄_f, ΔP，保存 prior.npz           (Eq.3)
│   ├── asa.py                      # 带内归一化 KL + λ·LPIPS，网格坐标下降                 (Eq.7–8)
│   ├── weights.py                  # a(u,v)、W(u,v)，以及 uniform / low-freq 对照权重     (Eq.10–11)
│   ├── sbqr.py                     # 谱偏置边界细化 + 步长自适应 + 归档                    (Eq.12–14)
│   ├── init.py                     # ASA 验证、参考池验证、二分括定                        (Eq.9)
│   ├── query.py                    # QueryCounter / 标签缓存 / 预算检查 / 检查点记录（所有方法共用）
│   ├── perceptual.py               # LPIPS(alex, v0.1, 输入 2x−1)、PSNR、RMSE
│   ├── detectors/
│   │   ├── base.py                 # 统一接口: predict(x[0,1] 256×256) -> {Real, Fake}
│   │   ├── ufd.py                  # UniversalFakeDetect: CLIP ViT-L/14 + fc_weights.pth
│   │   └── dire.py                 # DIRE: ADM DDIM 反演+重建(20步) → |x−recon| → ResNet50
│   ├── baselines/
│   │   ├── boundary.py             # BA（foolbox 3.3.4 默认参数，经共享查询包装器）
│   │   ├── hsja.py                 # HSJA（同上）
│   │   └── qeba.py                 # QEBA-S 空间变体（移植官方实现）
│   └── attack.py                   # SAGE 完整流程（论文 3.7 节 6 步）
├── experiments/
│   ├── screen_clean.py             # 干净样本筛选 I_D、FPR（单独计费，不计入预算）
│   ├── run_primary.py              # 实验块 1：4 方法 × 2 检测器 × 400 图 × 3 seeds
│   ├── run_controls.py             # 实验块 2：100 图子集，共享起点 + uniform/lowfreq/learned
│   ├── run_transfer.py             # 实验块 3：SD-v1-only prior → SD-v2
│   └── analyze.py                  # ASR@Q、惩罚查询数(5001)、bootstrap 10k、出图
└── slurm/                          # 按 (detector, method, seed, shard) 切分的数组作业
```

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
