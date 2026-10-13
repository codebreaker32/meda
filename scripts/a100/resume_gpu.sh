#!/usr/bin/env bash
# Second pass on the A100 after the CPU queue of overnight.sh proved too slow: runs the
# unfinished jobs on the GPU, reusing the healthy 30x30 models in $OUT/runs.
#   PARALLEL=14 OUT=/workspace/meda/out nohup bash scripts/a100/resume_gpu.sh > resume_gpu.log 2>&1 &
# Jobs (runs of the same name and seed are replaced):
#   - fine-tuning on chips with 10% and 20% faulty MCs, 20 epochs, from the healthy 30x30
#     model of the same seed: direction-aware GCN seeds 0-4, CNN (20% only) seeds 1-4
#   - 50x50 and 60x60, CNN and direction-aware GCN, seeds 1-4, 40 epochs (as seed 0)
set -uo pipefail
cd "$(dirname "$0")/../.."
PARALLEL=${PARALLEL:-14}
OUT=${OUT:-/workspace/meda/out}
THREADS=${THREADS:-4}
SETS="--set output_dir=$OUT/runs --set schedule.checkpoint_every=0 --set repeats=1 --set save_plots=false --set ppo.device=cuda"
jobs=$OUT/jobs_gpu.txt
: > "$jobs"
job() {  # job <name> <config> <seed> [--set ...]
  local name=$1 cfg=$2 seed=$3; shift 3
  local cfgname; cfgname=$(sed -n 's/^name: *\([^ #]*\).*/\1/p' "configs/training/$cfg.yaml")
  echo "cd $PWD && rm -rf $OUT/runs/$cfgname/seed_$seed && ( export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS; meda train -c configs/training/$cfg.yaml --set seed=$seed $SETS $* ) > $OUT/logs/gpu_$name.log 2>&1 && echo \"\$(date '+%F %T') done $name\" || echo \"\$(date '+%F %T') FAILED $name\"" >> "$jobs"
}
for s in 1 2 3 4; do
  job dirgcn60_s$s gnn_dirgcn_60x60 $s --set schedule.epochs=40
  job dirgcn50_s$s gnn_dirgcn_50x50 $s --set schedule.epochs=40
done
for s in 0 1 2 3 4; do
  job dirgcn_f10_s$s faults10_dirgcn_30x30 $s --set init_from=$OUT/runs/gnn_dirgcn_30x30/seed_$s
  job dirgcn_f20_s$s faults20_dirgcn_30x30 $s --set init_from=$OUT/runs/gnn_dirgcn_30x30/seed_$s
done
for s in 1 2 3 4; do
  job cnn_f20_s$s faults20_cnn_30x30 $s --set init_from=$OUT/runs/paper_30x30_healthy/seed_$s
  job cnn60_s$s paper_60x60_healthy $s --set schedule.epochs=40
  job cnn50_s$s paper_50x50_healthy $s --set schedule.epochs=40
done
echo "$(date '+%F %T') $(wc -l < "$jobs") jobs, $PARALLEL in parallel on the GPU"
tr '\n' '\0' < "$jobs" | xargs -0 -P "$PARALLEL" -I{} bash -c '{}'
echo "$(date '+%F %T') all jobs finished"
