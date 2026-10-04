#!/usr/bin/env bash
# Chip-size study on Kaggle: CNN-PPO baseline vs direction-aware GCN + max pooling
# on 30x30, 50x50, 60x60, 100x100 and 120x120 chips, 40 epochs each (GCN + max
# pooling is left out: it did not learn on 16x16 or 30x30). Each (size, method) is
# its own kernel, meda-size-<N>x<N>-cnn / -dirgcn, and its training stops by 11 h
# (schedule.max_hours, set by run_kernel.py), inside Kaggle's 12 h limit.
#
# Kaggle runs at most two GPU kernels at a time. The script keeps a queue, smallest
# chip first, and pushes the next kernel whenever fewer than MAX kernels are running,
# counting its own and the OTHER_GPU kernels (default: the reference test's GPU
# kernel). A push refused for lack of a free GPU session (Kaggle seems to count
# CPU-only kernels too) is retried every POLL seconds until it goes through; any
# other refused push is retried up to RETRIES times. When both kernels of a size
# have finished, they are downloaded and compared in the background
# (kaggle/fetch_sizes.sh; log in kaggle/build/fetch_<N>x<N>.log).
# Run where the Kaggle CLI is logged in, and leave it running:
#
#   nohup bash kaggle/push_sizes.sh > kaggle/build/push_sizes.log 2>&1 &
#   SIZES="50 60" bash kaggle/push_sizes.sh       # a subset
#   ONLY=dirgcn SIZES=120 bash kaggle/push_sizes.sh
#   EPOCHS=25 bash kaggle/push_sizes.sh            # another budget
#   SEEDS=5 bash kaggle/push_sizes.sh              # paper protocol (5 seeds per kernel)
#   EXTRA="--set eval.episodes=300" bash kaggle/push_sizes.sh   # further overrides
#   OTHER_GPU="" bash kaggle/push_sizes.sh         # no other GPU kernels to count
#   FETCH=0 bash kaggle/push_sizes.sh              # download later with fetch_sizes.sh
#
# Kernels need "Internet" (phone-verified Kaggle account) to clone the repository,
# so configs and code must be pushed to the branch first.
set -euo pipefail
cd "$(dirname "$0")"
SIZES=${SIZES:-"30 50 60 100 120"}
SEEDS=${SEEDS:-1}
EPOCHS=${EPOCHS:-40}
ONLY=${ONLY:-}
EXTRA=${EXTRA:-}
MAX=${MAX:-2}
OTHER_GPU=${OTHER_GPU-meda-ref-ours}
FETCH=${FETCH:-1}
POLL=${POLL:-300}      # seconds between status checks
SETTLE=${SETTLE:-60}   # seconds to let Kaggle register a new version before polling it
RETRIES=${RETRIES:-36}
USER_NAME=${KAGGLE_USERNAME:-$(python3 - <<'PY'
import json, os
for p in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
    p = os.path.expanduser(p)
    if os.path.exists(p):
        print(json.load(open(p))["username"]); break
PY
)}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi
now() { date '+%F %H:%M'; }

build() {  # build <slug> <python tuple (config, label)>
  local slug=$1 dir=build/$1
  mkdir -p "$dir"
  python3 - "$2" "$SEEDS" "--set schedule.epochs=$EPOCHS $EXTRA" > "$dir/run.py" <<'PY'
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
}

queue=()
declare -A size_of left
for N in $SIZES; do
  for m in cnn dirgcn; do
    [[ -z "$ONLY" || "$ONLY" == "$m" ]] || continue
    slug="meda-size-${N}x${N}-$m"
    if [[ $m == cnn ]]; then job="(\"paper_${N}x${N}_healthy\", \"CNN-PPO\")"
    else job="(\"gnn_dirgcn_${N}x${N}\", \"DirGCN-maxpool-PPO\")"; fi
    build "$slug" "$job"
    queue+=("$slug"); size_of[$slug]=$N; left[$N]=$(( ${left[$N]:-0} + 1 ))
  done
done
echo "$(now) queue (${EPOCHS} epochs, ${SEEDS} seed(s)): ${queue[*]}"

declare -A misses
busy() {  # busy <slug>: is our kernel queued or running?
  # Kaggle can deny the status of a newly created private kernel for a while
  # ("Permission 'kernels.get' was denied"), so an unreadable status counts as
  # busy; give up only after 6 unreadable checks in a row.
  local st
  st=$(kaggle kernels status "$USER_NAME/$1" 2>&1 || true)
  case "$st" in
    *[Rr][Uu][Nn][Nn][Ii][Nn][Gg]*|*[Qq][Uu][Ee][Uu][Ee][Dd]*) misses[$1]=0; return 0 ;;
    *"has status"*) misses[$1]=0; echo "$(now) finished: $1 (${st##*has status })"; return 1 ;;
  esac
  misses[$1]=$(( ${misses[$1]:-0} + 1 ))
  echo "$(now) status of $1 not readable yet (${misses[$1]}/6): $st" >&2
  if (( misses[$1] >= 6 )); then
    echo "!! giving up on $1; check https://www.kaggle.com/code/$USER_NAME/$1" >&2
    return 1
  fi
  return 0
}

other_busy() {  # other_busy <slug>: is a kernel outside this study queued or running?
  case "$(kaggle kernels status "$USER_NAME/$1" 2>&1 || true)" in
    *[Rr][Uu][Nn][Nn][Ii][Nn][Gg]*|*[Qq][Uu][Ee][Uu][Ee][Dd]*) return 0 ;;
  esac
  return 1
}

declare -A tries
running=()
while (( ${#queue[@]} )) || (( ${#running[@]} )); do
  still=()
  for k in "${running[@]}"; do
    if busy "$k"; then still+=("$k"); continue; fi
    N=${size_of[$k]}; left[$N]=$(( left[$N] - 1 ))
    if (( left[$N] == 0 && FETCH )); then
      echo "$(now) ${N}x${N} done; downloading and comparing (build/fetch_${N}x${N}.log)"
      SIZES=$N bash fetch_sizes.sh > "build/fetch_${N}x${N}.log" 2>&1 &
    fi
  done
  running=("${still[@]}")

  others=0
  for k in $OTHER_GPU; do other_busy "$k" && others=$(( others + 1 )); done
  while (( ${#queue[@]} )) && (( ${#running[@]} + others < MAX )); do
    k=${queue[0]}
    echo "$(now) pushing $USER_NAME/$k"
    out=$(kaggle kernels push -p "build/$k" 2>&1 || true)
    echo "$out"
    if [[ "$out" == *"session count"* ]]; then   # all GPU sessions busy, some outside this script
      echo "$(now) no free GPU session for $k yet; retrying in $POLL s" >&2
      break
    fi
    if [[ "$out" != *"successfully pushed"* ]]; then   # the CLI exits 0 on failures too
      tries[$k]=$(( ${tries[$k]:-0} + 1 ))
      if (( tries[$k] >= RETRIES )); then
        echo "!! push of $k refused $RETRIES times; stopping (rerun with SIZES/ONLY for the rest)" >&2
        exit 1
      fi
      echo "!! push of $k refused (${tries[$k]}/$RETRIES); retrying in $POLL s" >&2
      break
    fi
    queue=("${queue[@]:1}"); running+=("$k")
    sleep "$SETTLE"
  done
  if (( ${#queue[@]} )) || (( ${#running[@]} )); then
    echo "$(now) running: ${running[*]:-none} | other GPU busy: $others | queued: ${queue[*]:-none}"
    sleep "$POLL"
  fi
done
wait   # for the last downloads
echo "$(now) Done. Results: results/chip_size/ (summary.csv)"
