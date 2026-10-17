# Source this at the top of every UMHPC job / interactive shell:
#   source $PROJECT_ROOT/scripts/umhpc/env.sh
# Paths can be overridden by exporting them before sourcing.

export PROJECT_ROOT=${PROJECT_ROOT:-$HOME/SAGE}
export CONDA_ROOT=${CONDA_ROOT:-$HOME/miniforge3}
export SAGE_ENV=${SAGE_ENV:-sage}

# Large data lives outside $HOME if your home quota is small (check with `quota -s` / `df -h ~`).
export DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data}
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
conda activate "$SAGE_ENV"
