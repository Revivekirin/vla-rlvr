#!/usr/bin/env bash
# Execute in a child shell: do not source this file into an interactive Slurm shell.
set -eo pipefail
if [[ $# -lt 2 || ( "$1" != original && "$1" != plus ) ]]; then
  echo 'Usage: bash experiments/server_env.sh {original|plus} COMMAND [ARGS...]' >&2
  exit 2
fi
benchmark_mode=$1
shift
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
conda_root=${VLA_CONDA_ROOT:-/home/jhkim/miniconda3}
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${VLA_CONDA_ENV:-ript_vla_openvla_oft_stage0}"
cd "$repo_root"

# RLIMIT_NOFILE is per process, inherited by both benchmark evaluators.
# This cannot raise the administrator/Slurm hard limit.
fd_hard=$(ulimit -Hn)
ulimit -Sn "$fd_hard"
printf 'Evaluation environment: %s; open files soft=%s hard=%s\n' \
  "$CONDA_PREFIX" "$(ulimit -Sn)" "$fd_hard"

export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Preserve CUDA_VISIBLE_DEVICES supplied by Slurm; never select a physical GPU here.
if [[ "$benchmark_mode" == plus ]]; then
  export MAGICK_HOME=${VLA_MAGICK_HOME:-$conda_root/envs/liberoplus_imagemagick}
  if [[ ! -d "$MAGICK_HOME/lib" ]]; then
    echo "Missing ImageMagick installation: $MAGICK_HOME (see experiments/libero_plus/README.md)" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
# Each existing Python entrypoint selects its own LIBERO import/config paths.
exec "$@"
