#!/usr/bin/env bash
set -euo pipefail

# Always build relative to this script's directory so the command works
# from any working directory (e.g. `bash solver/build.sh` from the repo root).
cd "$(dirname "$0")"

cmake -S . -B build
cmake --build build -j"$(nproc)"

mkdir -p bin
cp build/solver bin/solver
chmod +x bin/solver
