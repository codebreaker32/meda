#!/usr/bin/env bash
# CNN-PPO baseline vs. GNN + global max pooling + PPO (docs/GNN_METHODOLOGY.md).
#
#   bash scripts/run_gnn_experiment.sh                 # 30x30, paper budget, 5 seeds (GPU recommended)
#   SIZE=16 SEEDS=1 bash scripts/run_gnn_experiment.sh # CPU-sized version (about 1 h on 4 cores)
#   ABLATIONS=1 bash scripts/run_gnn_experiment.sh     # also the dir_gcn and role-readout ablations
#   GPU_SERVER=user@192.168.x.x bash scripts/run_on_gpu_server.sh bash scripts/run_gnn_experiment.sh
#
# Steps: graph checks -> train the frozen CNN baseline -> train the GNN with the
# same settings and seeds -> evaluate both on the same held-out jobs ->
# results/tables, results/logs, results/figures. Runs, checkpoints and training
# plots go to runs/<name>/seed_<s>/. Every network uses a local GPU if present.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${SIZE:-30}     # 30: paper setting; 16: CPU-sized pair of configs
SEEDS=${SEEDS:-5}    # the paper repeats every experiment 5 times [PAPER Sec. V-B]
EPISODES=${EPISODES:-500}
ABLATIONS=${ABLATIONS:-0}   # 1: also train and compare gnn_dirgcn_* and gnn_role_*
EXTRA=("$@")         # further --set overrides for both methods, e.g. --set schedule.epochs=10

if [[ "$SIZE" == "16" ]]; then
  CNN_CFG=configs/training/validation_16x16.yaml; CNN_NAME=validation_16x16
else
  CNN_CFG=configs/training/paper_30x30_healthy.yaml; CNN_NAME=paper_30x30_healthy
fi
S="${SIZE}x${SIZE}"
GNN_CFG=configs/training/gnn_maxpool_$S.yaml; GNN_NAME=gnn_maxpool_$S

meda devices
echo "== graph representation checks"
meda validate-graph -c "$GNN_CFG"
echo "== CNN-PPO baseline ($SEEDS seeds)"
meda train -c "$CNN_CFG" --set repeats="$SEEDS" ${EXTRA[@]+"${EXTRA[@]}"}
echo "== GNN + max pooling + PPO ($SEEDS seeds)"
meda train -c "$GNN_CFG" --set repeats="$SEEDS" ${EXTRA[@]+"${EXTRA[@]}"}
METHODS=(--method "CNN-PPO" "runs/$CNN_NAME" --method "GNN-maxpool-PPO" "runs/$GNN_NAME")
if [[ "$ABLATIONS" == "1" ]]; then
  for ab in dirgcn role; do
    echo "== ablation gnn_${ab}_$S ($SEEDS seeds)"
    meda train -c "configs/training/gnn_${ab}_$S.yaml" --set repeats="$SEEDS" ${EXTRA[@]+"${EXTRA[@]}"}
  done
  METHODS+=(--method "GNN-dirGCN-max-PPO" "runs/gnn_dirgcn_$S" --method "GNN-GCN-role-PPO" "runs/gnn_role_$S")
fi
echo "== comparison on the same held-out jobs"
meda compare-methods "${METHODS[@]}" --episodes "$EPISODES"
