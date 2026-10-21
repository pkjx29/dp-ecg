#!/usr/bin/env bash
# One command to reproduce milestone 1 (steps 1-3) from a clean checkout:
#   ./run_all.sh            full run (3 seeds per ε)
#   ./run_all.sh --quick    smoke test (1 seed, 3 ε values)
# Re-runs are cheap: verified downloads and cached features are reused.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
PY=".venv/bin/python"
export PYTHONPATH="$PWD/src"

echo "== 0. unit tests";              $PY -m pytest -q
echo "== 1a. download PTB-XL (100 Hz)"; $PY scripts/download_ptbxl.py --out data/raw/ptb-xl
if [ ! -f data/features.npz ]; then
  echo "== 1b. feature extraction";   $PY scripts/build_features.py --root data/raw/ptb-xl --out data/features.npz
fi
echo "== 2. DP summary statistics";   $PY scripts/dp_summary_stats.py --features data/features.npz --out results
echo "== 1c+3. baselines and DP-SGD"; $PY scripts/run_dpsgd.py --features data/features.npz --out results "$@"
echo "Done. See results/ and README.md."
