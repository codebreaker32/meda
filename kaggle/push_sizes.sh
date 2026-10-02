#!/usr/bin/env bash
# Chip-size study on Kaggle: CNN-PPO baseline vs direction-aware GCN + max pooling
# on 50x50, 60x60, 100x100 and 120x120 chips (GCN + max pooling is left out: it
# did not learn on 16x16 or 30x30). Each (size, method) is its own kernel,
# meda-<N>x<N>-cnn / meda-<N>x<N>-dirgcn, so no kernel comes near Kaggle's 12 h
# limit. Kaggle runs at most two kernels at a time, so the script pushes one
# size (two kernels), waits until both have finished, then pushes the next.
# Run where the Kaggle CLI is logged in, and leave it running (or use tmux/nohup):
#
#   bash kaggle/push_sizes.sh                 # all four sizes, one after the other
#   SIZES="50 60" bash kaggle/push_sizes.sh   # a subset
#   ONLY=dirgcn SIZES=120 bash kaggle/push_sizes.sh
#   WAIT=0 SIZES=100 bash kaggle/push_sizes.sh   # push and return immediately
#   SEEDS=5 bash kaggle/push_sizes.sh         # paper protocol (5 seeds per kernel)
#   EXTRA="--set schedule.epochs=15" SIZES=120 bash kaggle/push_sizes.sh   # extra overrides
#
# Kernels need "Internet" (phone-verified Kaggle account) to clone the repository,
# so the configs must be pushed to the branch first. Download the results with
# kaggle/fetch_sizes.sh.
set -euo pipefail
cd "$(dirname "$0")"
SIZES=${SIZES:-"50 60 100 120"}
SEEDS=${SEEDS:-1}
ONLY=${ONLY:-}
EXTRA=${EXTRA:-}
WAIT=${WAIT:-1}
POLL=${POLL:-300}   # seconds between status checks
USER_NAME=${KAGGLE_USERNAME:-$(python3 - <<'PY'
import json, os
for p in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
    p = os.path.expanduser(p)
    if os.path.exists(p):
        print(json.load(open(p))["username"]); break
PY
)}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi

push() {  # push <slug> <python list of (config, label)>
  local slug=$1 jobs=$2 dir=build/$1
  mkdir -p "$dir"
  python3 - "$jobs" "$SEEDS" "$EXTRA" > "$dir/run.py" <<'PY'
import sys
src = open("run_kernel.py").read()
src = src.replace("__JOBS__", sys.argv[1]).replace("__SEEDS__", sys.argv[2])
print(src.replace('EXTRA_SETS = ""', f"EXTRA_SETS = {sys.argv[3]!r}", 1), end="")
PY
  cat > "$dir/kernel-metadata.json" <<JSON
{
  "id": "$USER_NAME/$slug",
  "title": "$slug",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
JSON
  echo "== pushing $USER_NAME/$slug ($SEEDS seed(s))"
  kaggle kernels push -p "$dir"
}

wait_for() {  # wait until every given kernel has left the queued/running states
  local slug status busy
  while :; do
    busy=0
    for slug in "$@"; do
      status=$(kaggle kernels status "$USER_NAME/$slug" 2>&1 || true)
      echo "$(date +%H:%M) $slug: ${status##*has status }"
      case "$status" in *[Rr][Uu][Nn][Nn][Ii][Nn][Gg]*|*[Qq][Uu][Ee][Uu][Ee][Dd]*) busy=1 ;; esac
    done
    (( busy )) || return 0
    sleep "$POLL"
  done
}

for N in $SIZES; do
  slugs=()
  if [[ -z "$ONLY" || "$ONLY" == cnn ]]; then
    push "meda-${N}x${N}-cnn" "(\"paper_${N}x${N}_healthy\", \"CNN-PPO\")"
    slugs+=("meda-${N}x${N}-cnn")
  fi
  if [[ -z "$ONLY" || "$ONLY" == dirgcn ]]; then
    push "meda-${N}x${N}-dirgcn" "(\"gnn_dirgcn_${N}x${N}\", \"DirGCN-maxpool-PPO\")"
    slugs+=("meda-${N}x${N}-dirgcn")
  fi
  if (( WAIT )); then
    sleep 60   # let Kaggle register the new versions before polling
    wait_for "${slugs[@]}"
  fi
done
echo "Done. Download with: bash kaggle/fetch_sizes.sh"
