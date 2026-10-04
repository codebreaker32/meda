#!/usr/bin/env bash
# Reference test on Kaggle: the authors' original code vs. this reimplementation,
# both with the settings of the authors' logged 30x30 run 0825a (Table I CNN,
# 2^14 steps per epoch, 40 epochs, 500 evaluation jobs per epoch).
#   meda-ref-ours        this code, configs/training/reference_0825a_30x30.yaml,
#                        OURS_SEEDS seeds (default 3) on the GPU
#   meda-ref-orig-s<k>   the original code (scripts/reference/run_original_0825a.py),
#                        one kernel per seed in ORIG_SEEDS (default "0 1 2"), CPU only
# At most MAX kernels (default 2) run at once; the script pushes the next one as soon
# as one finishes, so leave it running (tmux/nohup). Download with fetch_reference.sh.
#
#   bash kaggle/push_reference.sh
#   ORIG_SEEDS="1 2" bash kaggle/push_reference.sh   # seed 0 of the original already run elsewhere
#   ONLY=ours bash kaggle/push_reference.sh
set -euo pipefail
cd "$(dirname "$0")"
OURS_SEEDS=${OURS_SEEDS:-3}
ORIG_SEEDS=${ORIG_SEEDS:-"0 1 2"}
ONLY=${ONLY:-}
MAX=${MAX:-2}
POLL=${POLL:-300}
USER_NAME=${KAGGLE_USERNAME:-$(python3 - <<'PY'
import json, os
for p in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
    p = os.path.expanduser(p)
    if os.path.exists(p):
        print(json.load(open(p))["username"]); break
PY
)}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi

build() {  # build <slug> <template> <gpu true|false> <JOBS or ""> <SEEDS>
  local slug=$1 dir=build/$1
  mkdir -p "$dir"
  python3 - "$2" "$4" "$5" > "$dir/run.py" <<'PY'
import sys
src = open(sys.argv[1]).read()
print(src.replace("__JOBS__", sys.argv[2]).replace("__SEEDS__", sys.argv[3]), end="")
PY
  cat > "$dir/kernel-metadata.json" <<JSON
{
  "id": "$USER_NAME/$slug",
  "title": "$slug",
  "code_file": "run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": $3,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
JSON
}

queue=()
if [[ -z "$ONLY" || "$ONLY" == ours ]]; then
  build meda-ref-ours run_kernel.py true '("reference_0825a_30x30", "CNN-PPO-reimplementation")' "$OURS_SEEDS"
  queue+=(meda-ref-ours)
fi
if [[ -z "$ONLY" || "$ONLY" == orig ]]; then
  for s in $ORIG_SEEDS; do
    build "meda-ref-orig-s$s" run_original_kernel.py false "" "[$s]"
    queue+=("meda-ref-orig-s$s")
  done
fi

declare -A misses
busy() {  # busy <slug>: is the kernel queued or running?
  # Kaggle can deny the status of a newly created private kernel for a while
  # ("Permission 'kernels.get' was denied"), so an unreadable status counts as
  # busy; give up only after 6 unreadable checks in a row.
  local st
  st=$(kaggle kernels status "$USER_NAME/$1" 2>&1 || true)
  case "$st" in
    *[Rr][Uu][Nn][Nn][Ii][Nn][Gg]*|*[Qq][Uu][Ee][Uu][Ee][Dd]*) misses[$1]=0; return 0 ;;
    *"has status"*) misses[$1]=0; return 1 ;;
  esac
  misses[$1]=$(( ${misses[$1]:-0} + 1 ))
  echo "$(date +%H:%M) status of $1 not readable yet (${misses[$1]}/6): $st" >&2
  if (( misses[$1] >= 6 )); then
    echo "!! giving up on $1; check https://www.kaggle.com/code/$USER_NAME/$1" >&2
    return 1
  fi
  return 0
}

running=()
while (( ${#queue[@]} )) || (( ${#running[@]} )); do
  still=()
  for k in "${running[@]}"; do busy "$k" && still+=("$k") || echo "$(date +%H:%M) finished: $k"; done
  running=("${still[@]}")
  while (( ${#queue[@]} )) && (( ${#running[@]} < MAX )); do
    k=${queue[0]}; queue=("${queue[@]:1}")
    echo "$(date +%H:%M) pushing $USER_NAME/$k"
    out=$(kaggle kernels push -p "build/$k" 2>&1 || true)
    echo "$out"
    if [[ "$out" != *"successfully pushed"* ]]; then
      echo "!! push of $k failed (see the message above); not continuing" >&2
      exit 1
    fi
    running+=("$k")
    sleep 60   # let Kaggle register the new version before polling it
  done
  (( ${#running[@]} )) && { echo "$(date +%H:%M) running: ${running[*]}"; sleep "$POLL"; }
done
echo "Done. Download with: bash kaggle/fetch_reference.sh"
