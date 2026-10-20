# 数据集与权重准备（UMHPC）

目标：按论文 Table 1 从 DIRE 的 **DiffusionForensics** 基准中选出
1000 real + 500 SD-v1 + 500 SD-v2，生成冻结的 manifest。

| Subset      | Real | SD-v1 | SD-v2 | Total |
|-------------|-----:|------:|------:|------:|
| Development |  500 |   250 |   250 |  1000 |
| Validation  |  100 |    50 |    50 |   200 |
| Test        |  400 |   200 |   200 |   800 |

## 1. 选哪个子集

DiffusionForensics 的目录是 `images/{train,val,test}/{lsun_bedroom,imagenet,celebahq}/{real,adm,...}`
（另有 `recons/`、`dire/` 两个平行目录，**我们不需要**——DIRE 图必须对攻击后的图在线重算）。

推荐用 **`images/test/lsun_bedroom/`** 下的：

- `real/` — LSUN bedroom 真实图
- `sdv1_new/`（或 `sdv1/`）— Stable Diffusion v1
- `sdv2/` — Stable Diffusion v2

理由：SD-v1 与 SD-v2 在 lsun_bedroom 下同场景同分辨率，real 与 fake 内容域一致，
可减小 SAP 先验混入“场景差异”的风险（论文第 6 节的主要威胁之一）；
且 DIRE 有对应的 `lsun_adm.pth` 检测器和 ADM `lsun_bedroom.pt` 重建模型。

> 文件夹名请以实际下载内容为准（`ls` 一下）。DIRE 自带的 `compute_dire.sh` 示例里还出现了
> `imagenet/sdv1`；如果 lsun_bedroom 下没有 sdv2，就要改用 imagenet 子集并换成
> `256x256_diffusion_uncond.pt` + `imagenet_adm.pth`，并在论文中如实记录。

**与检测器训练数据的重叠**：DIRE 的 `lsun_adm.pth` 是在 `train/lsun_bedroom/{real,adm}` 上训练的；
UniversalFakeDetect 在 ProGAN/LSUN 数据上训练。我们只用 `test/` 划分，并且把
`train/lsun_bedroom/real` 作为 `--exclude` 传给 manifest 脚本做哈希+pHash 重叠检查。
UFD 训练集（CNNDetection ProGAN 数据，LSUN 多类）若无法下载比对，就在论文里写明“未解决的潜在重叠”。

## 2. 下载（在你自己的电脑上）

DiffusionForensics 和 DIRE 检测器权重只放在网盘上（密码 `dire`），需要浏览器手动下载，
链接见 <https://github.com/ZhendongWang6/DIRE#readme>（RecDrive / BaiduDrive / OneDrive）。

只需要下载：

```
DiffusionForensics/images/test/lsun_bedroom/real/
DiffusionForensics/images/test/lsun_bedroom/sdv1_new/   (或 sdv1/)
DiffusionForensics/images/test/lsun_bedroom/sdv2/
DiffusionForensics/images/train/lsun_bedroom/real/      (仅用于重叠检查，可选)
DIRE 检测器权重: lsun_adm.pth
```

记录下载日期、链接、压缩包文件名和 sha256（写进 `--dataset-version`）。

## 3. 上传到 UMHPC

只上传需要的约 2.2 GB（不是整个 141 GB），**上传压缩包、在 UMHPC 上解压**：

```bash
bash scripts/data/upload_to_umhpc.sh
```

脚本会：本地算 sha256 → 建一个 SSH 主连接（只输一次密码）→ 用
`rsync --partial --append-verify` 逐个上传（断线自动重试，重跑脚本从断点续传）→
在 UMHPC 上校验 sha256 → 解压到 `~/SAGE/data/raw/DiffusionForensics/images/...` →
把 `lsun_adm.pth` 放到 `~/SAGE/weights/dire/`。

上传的文件：`images/test/lsun_bedroom/{real,sdv1_new,sdv2}.tar.gz`、
`images/train/lsun_bedroom/real.tar.gz`（重叠检查用，`INCLUDE_TRAIN_REAL=0` 可跳过）、
`checkpoints/lsun_adm.pth`。注意 DIRE 的 `test/lsun_bedroom/real.tar.gz` 实际是未压缩 tar，
`tar -xf` 会自动识别。

## 4. 生成 manifest（登录节点或 CPU 作业均可，无需 GPU）

```bash
source ~/SAGE/scripts/umhpc/env.sh
python scripts/data/build_manifest.py \
  --real  $DATA_ROOT/raw/DiffusionForensics/images/test/lsun_bedroom/real \
  --sdv1  $DATA_ROOT/raw/DiffusionForensics/images/test/lsun_bedroom/sdv1_new \
  --sdv2  $DATA_ROOT/raw/DiffusionForensics/images/test/lsun_bedroom/sdv2 \
  --exclude $DATA_ROOT/raw/DiffusionForensics/images/train/lsun_bedroom/real \
  --dataset-version "DiffusionForensics via RecDrive, downloaded 2026-10-xx, <zip sha256>" \
  --out $DATA_ROOT/sage_v1
python scripts/data/verify_manifest.py $DATA_ROOT/sage_v1
```

脚本做的事情（对应论文 4.1 / 4.4）：

1. 扫描图像，计算原始文件 sha256；
2. 在 256 画布上算 64-bit pHash，**精确重复 + 近重复（Hamming ≤ 4）** 合并成组，每组保留一张；
   同一内容同时出现在 real 和 fake 中则整组删除；
3. 与 `--exclude` 目录（检测器训练数据）做重叠检查并删除；
4. 用 split seed **20261008** 按类别随机划分为 Table 1；
5. 预处理到 **256×256 RGB**：中心裁成正方形 → PIL bicubic 缩放 → **无损 PNG**
   （与 DIRE `reshape_image` 的顺序一致；SD 图原始 512/768，LSUN 本来就是 256）；
6. 标记：
   - `ref_order`：development/real 的固定参考顺序（所有方法共用，初始化最多查前 50 个）；
   - `in_ablation`：test 中每个生成器 50 张（seed **20261009**），用于组件对照实验；
   - `in_pilot`：validation 中每个生成器 20 张，用于 40 张 pilot。

输出：

```
$DATA_ROOT/sage_v1/
├── images/{development,validation,test}/{real,sdv1,sdv2}/*.png
├── manifest.csv          # 每张图: id, 类别, 子集, 各种 hash, 标记
├── removed.csv           # 被删除的重复/重叠图及原因
└── manifest_meta.json    # 种子、来源、计数、manifest 自身 sha256
```

**manifest 一旦生成即冻结**：脚本拒绝覆盖已有 manifest；要改就换新目录并升协议版本号。

## 5. 各实验用到的数据

| 用途 | 数据 |
|---|---|
| SAP 混合先验 | development: real 500 + sdv1 250 + sdv2 250 |
| SAP SD-v1-only 先验（迁移实验） | development: real 500 + sdv1 250 |
| 参考池（初始化） | development/real，按 `ref_order` |
| 阈值/实现检查、pilot | validation（pilot = `in_pilot`） |
| 干净 FPR | test/real（**不是**参考池） |
| 攻击目标 | test/sdv1 + test/sdv2（再经各检测器的 eligibility 筛选） |
| 组件对照 | test 中 `in_ablation==1` |
| 迁移目标 | test/sdv2 |

注意：所有生成图在 SD 数据集中的 prompt 基本相同（卧室场景），没有可用的 prompt 分组信息，
这一点按论文要求在 manifest/论文中写明。
