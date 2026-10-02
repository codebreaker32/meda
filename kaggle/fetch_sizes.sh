#!/usr/bin/env bash
# Download the outputs of the chip-size kernels (kaggle/push_sizes.sh) and compare
# the CNN-PPO baseline with the direction-aware GCN on the same held-out jobs of
# each size. Run from anywhere after the kernels have finished:
#
#   bash kaggle/fetch_sizes.sh
#   SIZES="50 60" bash kaggle/fetch_sizes.sh
#
# Outputs: kaggle/output/<N>x<N>-{cnn,dirgcn}/ and results/kaggle_<N>x<N>/.
set -euo pipefail
cd "$(dirname "$0")/.."
SIZES=${SIZES:-"50 60 100 120"}
USER_NAME=${KAGGLE_USERNAME:-$(python3 - <<'PY'
import json, os
for p in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
    p = os.path.expanduser(p)
    if os.path.exists(p):
        print(json.load(open(p))["username"]); break
PY
)}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi

for N in $SIZES; do
  for m in cnn dirgcn; do
    out="kaggle/output/${N}x${N}-$m"
    kaggle kernels status "$USER_NAME/meda-${N}x${N}-$m" || true
    mkdir -p "$out"
    kaggle kernels output "$USER_NAME/meda-${N}x${N}-$m" -p "$out"
  done
  cnn="kaggle/output/${N}x${N}-cnn/runs/paper_${N}x${N}_healthy"
  dir="kaggle/output/${N}x${N}-dirgcn/runs/gnn_dirgcn_${N}x${N}"
  if [[ -d "$cnn" && -d "$dir" ]]; then
    meda compare-methods --method CNN-PPO "$cnn" --method DirGCN-maxpool-PPO "$dir" \
      --episodes 500 --out "results/kaggle_${N}x${N}"
  else
    echo "!! ${N}x${N}: missing run folder(s), skipping the comparison" >&2
  fi
done
