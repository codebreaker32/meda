#!/usr/bin/env bash
# Overnight job queue for a single-GPU server (A100): every job is one `meda train` run,
# at most PARALLEL at a time, longest first. Logs in $OUT/logs/<job>.log, runs in $OUT/runs/.
#   PARALLEL=12 OUT=/root/aman/out nohup bash scripts/a100/overnight.sh > overnight.log 2>&1 &
# Jobs:
#   - direction-aware GCN, 100x100 and 120x120, seed 0, full 40 epochs (no time limit)
#   - 50x50 and 60x60, CNN and direction-aware GCN, seeds 1-4 (40 epochs)
#   - 30x30, CNN and direction-aware GCN, seeds 0-4: healthy (40 epochs), then fine-tuned
#     from that model on chips with 10% and with 20% faulty MCs (20 epochs each, Fig. 8)
#   - CNN 30x30, seed 0, 100 epochs: does the baseline converge with more training?
set -uo pipefail
cd "$(dirname "$0")/../.."
PARALLEL=${PARALLEL:-12}
OUT=${OUT:-/root/aman/out}
THREADS=${THREADS:-8}
mkdir -p "$OUT/logs" "$OUT/runs"
SETS="--set output_dir=$OUT/runs --set schedule.checkpoint_every=0 --set repeats=1 --set save_plots=false"

jobs=$OUT/jobs.txt
: > "$jobs"
job() {  # job <name> <command...>
  local name=$1; shift
  echo "cd $PWD && ( export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS; $* ) > $OUT/logs/$name.log 2>&1 && echo \"\$(date '+%F %T') done $name\" || echo \"\$(date '+%F %T') FAILED $name\"" >> "$jobs"
}
train() { echo "meda train -c configs/training/$1.yaml --set seed=$2 $SETS ${*:3}"; }

job dirgcn120_s0 "$(train gnn_dirgcn_120x120 0)"
job dirgcn100_s0 "$(train gnn_dirgcn_100x100 0)"
for s in 1 2 3 4; do
  job dirgcn60_s$s "$(train gnn_dirgcn_60x60 $s)"
  job dirgcn50_s$s "$(train gnn_dirgcn_50x50 $s)"
done
for s in 0 1 2 3 4; do
  for m in "cnn paper_30x30_healthy paper_30x30_faults10 faults20_cnn_30x30" \
           "dirgcn gnn_dirgcn_30x30 faults10_dirgcn_30x30 faults20_dirgcn_30x30"; do
    set -- $m
    base=$OUT/runs/$2/seed_$s
    job ${1}30_faults_s$s "$(train $2 $s) && $(train $3 $s --set init_from=$base) && $(train $4 $s --set init_from=$base)"
  done
done
job cnn30_e100_s0 "$(train paper_30x30_healthy 0 --set name=paper_30x30_healthy_e100 --set schedule.epochs=100)"
for s in 1 2 3 4; do
  job cnn60_s$s "$(train paper_60x60_healthy $s)"
  job cnn50_s$s "$(train paper_50x50_healthy $s)"
done

echo "$(date '+%F %T') $(wc -l < "$jobs") jobs, $PARALLEL in parallel"
tr '\n' '\0' < "$jobs" | xargs -0 -P "$PARALLEL" -I{} bash -c '{}'
echo "$(date '+%F %T') all jobs finished"
