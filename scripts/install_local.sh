#!/usr/bin/env bash
# One-time setup for running the project on your own Linux / WSL2 machine with an NVIDIA GPU.
#   bash scripts/install_local.sh
# Options (environment variables):
#   PYTHON=python3.12      Python to build the venv with (3.10-3.13; TensorRT has no wheels for newer yet)
#   TRT_VERSION=10.16.1.11 TensorRT version (default: the one used for the published results)
#   TORCH_INDEX=<url>      override the PyTorch wheel index (default: picked from your driver's CUDA version)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
TRT_VERSION="${TRT_VERSION:-10.16.1.11}"
fail() { echo "ERROR: $*" >&2; exit 1; }

echo "== 1/6 NVIDIA GPU and driver"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found. Install the NVIDIA driver (on WSL2: the Windows driver) first."
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader \
  || nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
DRV_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
[ -n "$DRV_CUDA" ] || fail "could not read the driver's CUDA version from nvidia-smi"
echo "driver supports CUDA up to $DRV_CUDA"

echo "== 2/6 System packages"
missing=()
for c in g++ cmake git pkg-config wget unzip; do command -v "$c" >/dev/null || missing+=("$c"); done
pkg-config --exists opencv4 2>/dev/null || missing+=("libopencv-dev")
if ((${#missing[@]})); then
  echo "Missing: ${missing[*]}"
  echo "Install with:  sudo apt-get update && sudo apt-get install -y build-essential cmake git pkg-config wget unzip python3-venv libopencv-dev"
  exit 1
fi
CMAKE_VER=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+')
"$PY" - "$CMAKE_VER" <<'EOF' || fail "CMake >= 3.18 required (found $CMAKE_VER)"
import sys; a, b = map(int, sys.argv[1].split('.')); sys.exit(0 if (a, b) >= (3, 18) else 1)
EOF
echo "g++ $(g++ -dumpversion), cmake $CMAKE_VER, OpenCV $(pkg-config --modversion opencv4)"

echo "== 3/6 Python virtual environment (.venv)"
command -v "$PY" >/dev/null || fail "$PY not found"
PYV=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
"$PY" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)' \
  || fail "Python $PYV found; use 3.10-3.13, e.g. PYTHON=python3.12 bash scripts/install_local.sh"
"$PY" -m venv .venv || fail "venv creation failed (sudo apt-get install python3-venv)"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip

echo "== 4/6 PyTorch (matched to your driver)"
if [ -z "${TORCH_INDEX:-}" ]; then
  MAJ=${DRV_CUDA%%.*}; MIN=${DRV_CUDA#*.}
  if   [ "$MAJ" -ge 13 ]; then IDX=cu130
  elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; then IDX=cu128
  elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; then IDX=cu126
  else fail "driver CUDA $DRV_CUDA is too old; update the NVIDIA driver (needs CUDA >= 12.6)"; fi
  TORCH_INDEX="https://download.pytorch.org/whl/$IDX"
fi
echo "using $TORCH_INDEX"
pip install -q torch torchvision --index-url "$TORCH_INDEX"

echo "== 5/6 Project packages + TensorRT $TRT_VERSION"
pip install -q -r requirements.txt
CU=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
if ! pip install -q "tensorrt-cu${CU}==${TRT_VERSION}"; then
  echo "WARNING: tensorrt-cu${CU}==${TRT_VERSION} not available; installing the latest TensorRT 10.x instead"
  pip install -q "tensorrt-cu${CU}>=10.0,<11"
fi

echo "== 6/6 Check"
python - <<'EOF'
import torch, tensorrt, ultralytics
assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'
print('GPU        ', torch.cuda.get_device_name(0))
print('torch      ', torch.__version__, '(CUDA', torch.version.cuda + ')')
print('tensorrt   ', tensorrt.__version__)
print('ultralytics', ultralytics.__version__)
EOF
echo
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  python scripts/run_all.py --quick      # ~20 min smoke run"
echo "  python scripts/run_all.py              # full run"
