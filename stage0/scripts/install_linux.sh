#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
test "$(uname -s)" = Linux || { echo 'Requires Linux with NVIDIA CUDA; this script does not support macOS.'; exit 2; }
nvidia-smi
command -v nvcc
test -d "$ROOT/runtime/openvla-oft" || python3 "$ROOT/stage0/scripts/prepare.py"
# Activate a dedicated Python 3.10 environment before invoking this script.
python -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
python -m pip install 'pip==24.3.1' 'setuptools==75.6.0' 'wheel==0.45.1' 'packaging==24.2' 'ninja==1.11.1.3'
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -c "$ROOT/stage0/constraints.txt" -e "$ROOT/runtime/openvla-oft" -e "$ROOT/runtime/ript-vla" -e "$ROOT/vendor/LIBERO" 'imageio[ffmpeg]'
# Released RIPT INSTALL.md requests this version. CUDA toolkit + compatible
# compiler required; no silent CPU or alternative-attention substitution.
python -m pip install 'flash-attn==2.7.4.post1' --no-build-isolation
python -m pip check
python -m pip freeze > "$ROOT/stage0/evidence/linux-pip-freeze.txt"
python -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
