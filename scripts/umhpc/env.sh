# Source this at the top of every UMHPC job / interactive shell:
#   source /scr/user/eric123/project/SAGE/scripts/umhpc/env.sh
# Paths can be overridden by exporting them before sourcing.

# Repository root = two levels above this file (works wherever the repo was cloned).
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

# DiffusionForensics (DIRE benchmark), already extracted on UMHPC:
#   $DIRE_ROOT/images/{train,val,test}/{lsun_bedroom,imagenet,celebahq}/{real,adm,sdv1_new,sdv2,...}
export DIRE_ROOT=${DIRE_ROOT:-/scr/user/eric123/datasets/DIRE}
export DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}       # frozen 256x256 manifest goes to $DATA_ROOT/sage_v1
export WEIGHTS_ROOT=${WEIGHTS_ROOT:-$PROJECT_ROOT/weights}
export THIRD_PARTY=${THIRD_PARTY:-$PROJECT_ROOT/third_party}

# Compute nodes may have no internet: every model cache points at pre-downloaded files.
export TORCH_HOME=$WEIGHTS_ROOT/torch          # LPIPS AlexNet backbone
export CLIP_CACHE=$WEIGHTS_ROOT/clip           # CLIP ViT-L/14 for UniversalFakeDetect
export HF_HOME=$WEIGHTS_ROOT/hf
export HF_HUB_OFFLINE=1
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$PROJECT_ROOT/.cache/pip}

# Third-party code is loaded by file path inside sage/detectors.py (DIRE and UFD both define `networks`).
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}

# Determinism for the primary evaluation (paper Sec. 4.2)
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ROOT/envs/$SAGE_ENV"   # by path: works whichever conda is on PATH
