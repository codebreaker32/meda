#!/usr/bin/env bash
# Download the outputs of the two 30x30 kernels and compare them on the same
# held-out jobs. Run from the repository root after the kernels have finished.
set -euo pipefail
cd "$(dirname "$0")/.."
USER_NAME=${KAGGLE_USERNAME:?set KAGGLE_USERNAME}
for slug in cnn gcn; do
  kaggle kernels status "$USER_NAME/meda-30x30-$slug" || true
  mkdir -p "kaggle/output/$slug"
  kaggle kernels output "$USER_NAME/meda-30x30-$slug" -p "kaggle/output/$slug"
done
meda compare-methods \
  --method CNN-PPO kaggle/output/cnn/runs/paper_30x30_healthy \
  --method GCN-maxpool-PPO kaggle/output/gcn/runs/gnn_maxpool_30x30 \
  --method DirGCN-maxpool-PPO kaggle/output/gcn/runs/gnn_dirgcn_30x30 \
  --episodes 500 --out results/kaggle_30x30
