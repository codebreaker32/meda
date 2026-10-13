#!/usr/bin/env bash
# Full-budget reruns on the A100 of the runs that were cut short or under-trained on Kaggle:
#   - direction-aware GCN, 100x100 and 120x120, seed 0: 40 epochs without a time limit
#     (the Kaggle runs stopped after 34 and 21 epochs at the 11-hour limit);
#   - GCN + max pooling, 30x30, seed 0: 40 epochs (it had only 25).
# Other training runs on the server are paused (SIGSTOP) while these train and resumed
# afterwards; then each new model is evaluated on the 500 held-out jobs (seed 20000).
#   cd /workspace/meda && git pull && nohup bash scripts/a100/rerun_full.sh > rerun_full.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-/workspace/meda/out_full}
mkdir -p "$OUT/logs" "$OUT/runs" "$OUT/results"
SETS="--set output_dir=$OUT/runs --set schedule.checkpoint_every=0 --set repeats=1 --set save_plots=false --set ppo.device=cuda --set schedule.epochs=40 --set seed=0"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

others=$(pgrep -f "meda train -c configs" || true)
[[ -n $others ]] && { echo "$(date '+%F %T') pausing other runs: $others"; kill -STOP $others; }
resume() { [[ -n $others ]] && kill -CONT $others 2>/dev/null; echo "$(date '+%F %T') other runs resumed"; }
trap resume EXIT

declare -A cfg=([dirgcn120]=gnn_dirgcn_120x120 [dirgcn100]=gnn_dirgcn_100x100 [maxpool30]=gnn_maxpool_30x30)
declare -A label=([dirgcn120]=DirGCN-maxpool-PPO [dirgcn100]=DirGCN-maxpool-PPO [maxpool30]=GCN-maxpool-PPO)
pids=()
for k in dirgcn120 dirgcn100 maxpool30; do
  ( meda train -c "configs/training/${cfg[$k]}.yaml" $SETS > "$OUT/logs/$k.log" 2>&1 \
      && echo "$(date '+%F %T') trained $k" \
      && meda compare-methods --method "${label[$k]}" "$OUT/runs/${cfg[$k]}" --episodes 500 --device cuda \
           --out "$OUT/results/$k" > "$OUT/logs/eval_$k.log" 2>&1 \
      && echo "$(date '+%F %T') evaluated $k" || echo "$(date '+%F %T') FAILED $k" ) &
  pids+=($!)
done
wait "${pids[@]}"
echo "$(date '+%F %T') all reruns finished"
