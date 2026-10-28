#!/usr/bin/env bash
# Validate the host CUDA device and GPU4PySCF from Codex or another restricted
# session. A sandbox can hide /dev/nvidia* even while the host GPU is healthy.
set -euo pipefail

PYTHON="${GPU4PYSCF_PYTHON:-/home/shabir/software/envs/easyhybrid312/bin/python}"
ACTIVATOR="${GPU4PYSCF_ENV_ACTIVATOR:-/home/shabir/software/pdynamo3.sh}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 2
fi

# Load the shared software runtime. The environment is isolated from ~/.local;
# Python packages used by GPU4PySCF live in this single software location.
if [[ -r "$ACTIVATOR" ]]; then
    # shellcheck disable=SC1090
    source "$ACTIVATOR"
fi
export PYTHONNOUSERSITE=1

"$PYTHON" - <<'PY'
import sys

import cupy

try:
    count = cupy.cuda.runtime.getDeviceCount()
except Exception as exc:
    print(f"CUDA is inaccessible from this execution context: {exc}", file=sys.stderr)
    print("If running inside Codex, rerun this script with host-level execution;", file=sys.stderr)
    print("a sandbox with no /dev/nvidia* does not establish that the host GPU is absent.", file=sys.stderr)
    raise SystemExit(3)

print(f"Python: {sys.executable}")
print(f"CuPy: {cupy.__version__}")
print(f"CUDA runtime: {cupy.cuda.runtime.runtimeGetVersion()}")
print(f"CUDA devices: {count}")
for index in range(count):
    props = cupy.cuda.runtime.getDeviceProperties(index)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(f"GPU {index}: {name}")

import gpu4pyscf
from pyscf import gto
from gpu4pyscf.scf import RHF

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
energy = RHF(mol).kernel()
print(f"GPU4PySCF: {gpu4pyscf.__file__}")
print(f"GPU RHF H2 energy: {energy:.12f} Hartree")
PY
