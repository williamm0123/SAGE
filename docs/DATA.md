# 数据集与权重准备（UMHPC）

目标：按论文 Table 1 从 DIRE 的 **DiffusionForensics** 基准中选出
1000 real + 500 SD-v1 + 500 SD-v2，生成冻结的 manifest。

| Subset      | Real | SD-v1 | SD-v2 | Total |
|-------------|-----:|------:|------:|------:|
| Development |  500 |   250 |   250 |  1000 |
| Validation  |  100 |    50 |    50 |   200 |
| Test        |  400 |   200 |   200 |   800 |

## 1. 数据位置（UMHPC，已解压）

```
DIRE_ROOT=/scr/user/eric123/datasets/DIRE          # scripts/umhpc/env.sh 的默认值
$DIRE_ROOT/images/{train,val,test}/{lsun_bedroom,imagenet,celebahq}/{real,adm,...}
$DIRE_ROOT/{recons,dire}/...                       # 不使用：DIRE 图必须对攻击后的图在线重算
```

项目代码：`/scr/user/eric123/project/SAGE`（git pull）；处理后的 manifest 写到
`$PROJECT_ROOT/data/sage_v1`，权重在 `$PROJECT_ROOT/weights`。

## 2. 用哪些子集

只有 `test/lsun_bedroom/` 同时有 SD-v1 与 SD-v2（train/val 的 lsun_bedroom 只有 adm/iddpm/pndm/stylegan）：

| 类别 | 目录 | 张数 | 原始格式 |
|---|---|---:|---|
| real | `images/test/lsun_bedroom/real` + `images/val/lsun_bedroom/real` | 1000 + 1000 | JPEG，短边 256 |
| sdv1 | `images/test/lsun_bedroom/sdv1_new` | 1000 | PNG 512² |
| sdv2 | `images/test/lsun_bedroom/sdv2` | 1000 | PNG 768² |
| 重叠检查 | `images/train/lsun_bedroom/real`（DIRE `lsun_adm.pth` 的训练集） | 40000 | JPEG |

**为什么 real 要并入 val**：`test/lsun_bedroom/real` 恰好 1000 张，其中 15 张与 DIRE 训练集
近重复（pHash 汉明距离 0–4，多数为 0 且同尺寸），排除后只剩 985 < Table 1 需要的 1000，
manifest 脚本会直接报错退出。`val/lsun_bedroom/real` 是 DIRE 的验证集（非训练集），
与训练集重叠的 16 张同样会被排除。两个目录先合并、再按 split seed 划分；
`manifest_meta.json` 的 `selected_by_source` 记录每个子集各来自哪个目录，论文中需如实说明。
（本地 dry run：删除 1 张重复 + 30 张训练集重叠，Table 1 计数完全满足。）

`sdv1_new2` 也存在，但沿用 `sdv1_new`（可用 `SDV1_DIR=sdv1_new2` 覆盖，需升协议版本）。

**与检测器训练数据的重叠**：DIRE 的 `lsun_adm.pth` 在 `train/lsun_bedroom/{real,adm}` 上训练，
已用 `--exclude` 做 sha256 + pHash 排除；UniversalFakeDetect 在 ProGAN/LSUN 数据上训练，
无法比对的部分在论文中写明“未解决的潜在重叠”。另外 real 为 JPEG、生成图为 PNG 且需下采样，
这是论文第 6 节所说的采集/压缩差异，预处理无法消除。

## 3. DIRE 检测器权重

`setup_env.sh` 会自动从 `$DIRE_ROOT/checkpoints/lsun_adm.pth`（若存在）复制到
`$WEIGHTS_ROOT/dire/`；否则从 DIRE README 的网盘链接手动下载后放入该目录。

## 4. 生成 manifest

`slurm/01_prepare.sbatch` 在 `$SAGE_DATA/manifest.csv` 不存在时自动执行下面的命令；也可以在登录节点手动跑（纯 CPU，约 2 分钟）：

```bash
cd /scr/user/eric123/project/SAGE && source scripts/umhpc/env.sh
python scripts/data/build_manifest.py \
  --real  $DIRE_ROOT/images/test/lsun_bedroom/real $DIRE_ROOT/images/val/lsun_bedroom/real \
  --sdv1  $DIRE_ROOT/images/test/lsun_bedroom/sdv1_new \
  --sdv2  $DIRE_ROOT/images/test/lsun_bedroom/sdv2 \
  --exclude $DIRE_ROOT/images/train/lsun_bedroom/real \
  --dataset-version "DiffusionForensics (DIRE benchmark) at $DIRE_ROOT" \
  --out $DATA_ROOT/sage_v1
python scripts/data/verify_manifest.py $DATA_ROOT/sage_v1
```

脚本做的事情（对应论文 4.1 / 4.4）：

1. 扫描图像（每类可给多个目录，先合并），计算原始文件 sha256；
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
