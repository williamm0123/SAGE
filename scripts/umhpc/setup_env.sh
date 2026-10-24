#!/usr/bin/env bash
# One-time environment setup for SAGE on UMHPC (DICC, gpu-a100 partition, A100 80GB, sm_80).
#
# Run on the LOGIN node (needs internet; compute nodes may not have it), from the repo root:
#   cd /scr/user/eric123/project/SAGE && bash scripts/umhpc/setup_env.sh
#
# What it does:
#   1. installs Miniforge (if no conda found) and creates the `sage` env (Python 3.10)
#   2. installs PyTorch 2.4.1 + CUDA 12.1 wheels (bundle their own CUDA runtime; only need driver >= 530)
#   3. clones DIRE / UniversalFakeDetect / QEBA at recorded commits
#   4. pre-downloads public weights (ADM lsun_bedroom, CLIP ViT-L/14, LPIPS AlexNet)
#   5. writes an environment lock (versions, commits, sha256) to $PROJECT_ROOT/locks/
set -euo pipefail

export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export SAGE_ENV=${SAGE_ENV:-sage}
# Prefer the installation that already holds $SAGE_ENV (e.g. ~/miniforge3 while ~/miniconda3 is the
# active base); otherwise the active conda; otherwise ~/miniforge3 (installed by setup_env.sh).
if [ -z "${CONDA_ROOT:-}" ]; then
  _active=${CONDA_EXE:+$(dirname "$(dirname "$CONDA_EXE")")}
  for _c in "$_active" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [ -n "$_c" ] && [ -d "$_c/envs/$SAGE_ENV" ]; then CONDA_ROOT=$_c; break; fi
  done
  CONDA_ROOT=${CONDA_ROOT:-${_active:-$HOME/miniforge3}}
  unset _c _active
fi
export CONDA_ROOT
export DIRE_ROOT=${DIRE_ROOT:-/scr/user/eric123/datasets/DIRE}
export WEIGHTS_ROOT=${WEIGHTS_ROOT:-$PROJECT_ROOT/weights}
export THIRD_PARTY=${THIRD_PARTY:-$PROJECT_ROOT/third_party}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}

# Commits inspected when this script was written (2026-10-17). Change only deliberately and re-lock.
DIRE_COMMIT=1f89fff15f0e478307cdea9abdfe06a52fe7838a
UFD_COMMIT=030495aea3300a8b54c0ec37ec7fe1dd7e63c619
QEBA_COMMIT=8ee4a586a1ac5b172da92cc1c8efa9f0e543269c

mkdir -p "$WEIGHTS_ROOT"/{torch,clip,hf,guided_diffusion,dire,ufd} "$THIRD_PARTY" "$PROJECT_ROOT/locks" "$PIP_CACHE_DIR"

echo "== [1/5] conda env =="
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  installer=$(mktemp -d)/Miniforge3.sh
  curl -fsSL -o "$installer" \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash "$installer" -b -p "$CONDA_ROOT"
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
ENV_PREFIX=$CONDA_ROOT/envs/$SAGE_ENV
if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" -c conda-forge python=3.10 git
fi
echo "using conda env $ENV_PREFIX"
conda activate "$ENV_PREFIX"
python -m pip install --upgrade pip wheel

echo "== [2/5] PyTorch + deps =="
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$PROJECT_ROOT/envs/requirements.txt"

echo "== [3/5] third-party code =="
clone_at() {  # url dir commit
  if [ ! -d "$2/.git" ]; then git clone "$1" "$2"; fi
  if [ -n "${3:-}" ]; then git -C "$2" fetch -q origin && git -C "$2" checkout -q "$3"; fi
}
clone_at https://github.com/ZhendongWang6/DIRE                  "$THIRD_PARTY/DIRE"                "$DIRE_COMMIT"
clone_at https://github.com/WisconsinAIVision/UniversalFakeDetect "$THIRD_PARTY/UniversalFakeDetect" "$UFD_COMMIT"
clone_at https://github.com/AI-secure/QEBA                       "$THIRD_PARTY/QEBA"                "$QEBA_COMMIT"   # reference for the QEBA-S port
python -m pip install -e "$THIRD_PARTY/DIRE/guided-diffusion"
cp -n "$THIRD_PARTY/UniversalFakeDetect/pretrained_weights/fc_weights.pth" "$WEIGHTS_ROOT/ufd/"

echo "== [4/5] public weights =="
fetch() {  # url dest
  [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"
}
# ADM unconditional LSUN-bedroom model used by DIRE reconstruction for lsun_bedroom images
fetch https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt \
      "$WEIGHTS_ROOT/guided_diffusion/lsun_bedroom.pt"
# ImageNet unconditional ADM (only needed if you later evaluate the imagenet subset)
if [ "${FETCH_IMAGENET_ADM:-0}" = 1 ]; then
  fetch https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt \
        "$WEIGHTS_ROOT/guided_diffusion/256x256_diffusion_uncond.pt"
fi

TORCH_HOME=$WEIGHTS_ROOT/torch CLIP_CACHE=$WEIGHTS_ROOT/clip \
THIRD_PARTY=$THIRD_PARTY python - <<'PY'
import importlib.util, os, sys, lpips
init = os.path.join(os.environ["THIRD_PARTY"], "UniversalFakeDetect/models/clip/__init__.py")
spec = importlib.util.spec_from_file_location("ufd_clip", init, submodule_search_locations=[os.path.dirname(init)])
clip = importlib.util.module_from_spec(spec); sys.modules["ufd_clip"] = clip; spec.loader.exec_module(clip)
clip.load("ViT-L/14", device="cpu", download_root=os.environ["CLIP_CACHE"])
lpips.LPIPS(net="alex", version="0.1")  # pulls torchvision AlexNet into $TORCH_HOME
lpips.LPIPS(net="vgg", version="0.1")   # VGG16 backbone for AEROBLADE's LPIPS layer-2 distance
print("CLIP ViT-L/14 + LPIPS(alex, vgg) cached")
PY

# AEROBLADE autoencoders (SD1 VAE, SD2 VAE = stabilityai/sd-vae-ft-mse, Kandinsky 2.1 MoVQ) at the revisions
# pinned in configs/sage_v1.0.json, into $HF_HOME (compute nodes run with HF_HUB_OFFLINE=1).
HF_HOME=$WEIGHTS_ROOT/hf HF_HUB_OFFLINE=0 PYTHONPATH=$PROJECT_ROOT python - <<'PY'
import torch
import safetensors.torch  # noqa: F401  diffusers 0.30 uses safetensors.torch without importing it
from diffusers import AutoencoderKL, VQModel
from sage.config import load_config
for spec in load_config()["detectors"]["aeroblade"]["autoencoders"]:
    kw = dict(revision=spec["revision"], torch_dtype=torch.float16, use_safetensors=True)
    kw.update({k: spec[k] for k in ("subfolder", "variant") if spec.get(k)})
    (VQModel if spec["type"] == "vq" else AutoencoderKL).from_pretrained(spec["repo"], **kw)
    print("cached", spec["repo"], spec["revision"][:10])
PY

# DIRE classifier: the DiffusionForensics download ships it under checkpoints/
for c in "$DIRE_ROOT/checkpoints/lsun_adm.pth" "$DIRE_ROOT/../checkpoints/lsun_adm.pth"; do
  if [ ! -s "$WEIGHTS_ROOT/dire/lsun_adm.pth" ] && [ -s "$c" ]; then cp "$c" "$WEIGHTS_ROOT/dire/lsun_adm.pth"; fi
done
if [ ! -s "$WEIGHTS_ROOT/dire/lsun_adm.pth" ]; then
  cat <<EOF

  [MANUAL] DIRE classifier checkpoint not found.
  Download lsun_adm.pth from the links in https://github.com/ZhendongWang6/DIRE#readme
  (RecDrive / BaiduDrive / OneDrive, password: dire) and copy it to:
      $WEIGHTS_ROOT/dire/lsun_adm.pth
  e.g. from your PC:  rsync -avP lsun_adm.pth <user>@umhpc.dicc.um.edu.my:$WEIGHTS_ROOT/dire/

EOF
fi

echo "== [5/5] environment lock =="
lock=$PROJECT_ROOT/locks/env_$(date +%Y%m%d_%H%M%S)
mkdir -p "$lock"
python -m pip freeze > "$lock/pip_freeze.txt"
conda env export -p "$ENV_PREFIX" > "$lock/conda_env.yml"
for d in DIRE UniversalFakeDetect QEBA; do
  echo "$d $(git -C "$THIRD_PARTY/$d" rev-parse HEAD)"
done > "$lock/third_party_commits.txt"
find "$WEIGHTS_ROOT" -type f \( -name '*.pt' -o -name '*.pth' \) -print0 \
  | xargs -0 -r sha256sum > "$lock/weights_sha256.txt"
echo "Lock written to $lock"
echo "Next: (cd $PROJECT_ROOT && python -m pytest -q tests) && sbatch scripts/umhpc/check_env.sbatch"
