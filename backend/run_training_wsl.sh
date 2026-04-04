#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x "$SCRIPT_DIR/.venv-wsl/bin/python" ]]; then
  PYTHON_SITE_PACKAGES="$("$SCRIPT_DIR/.venv-wsl/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  if [[ -d "$PYTHON_SITE_PACKAGES/nvidia" ]]; then
    CUDA_LIB_PATHS="$(find "$PYTHON_SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':' | sed 's/:$//')"
    CUDA_BIN_PATHS="$(find "$PYTHON_SITE_PACKAGES/nvidia" -maxdepth 2 -type d -name bin | tr '\n' ':' | sed 's/:$//')"
    if [[ -n "${CUDA_LIB_PATHS:-}" ]]; then
      export LD_LIBRARY_PATH="$CUDA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    if [[ -n "${CUDA_BIN_PATHS:-}" ]]; then
      export PATH="$CUDA_BIN_PATHS:$PATH"
    fi
  fi
  exec "$SCRIPT_DIR/.venv-wsl/bin/python" run_training.py "$@"
fi

exec python3 run_training.py "$@"
