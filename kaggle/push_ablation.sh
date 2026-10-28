#!/usr/bin/env bash
# Settings ablation of the 30x30 baseline on Kaggle: the CNN baseline and the direction-aware
# GCN under the paper settings (base) and with one setting changed at a time
# (configs/training/ablate_<factor>_<method>_30x30.yaml): collision marks (marks), k_max of
# the whole chip (kmaxchip), alpha = 2 (alpha2), the five droplet sizes of the authors'
# logged run (sizes5), timeouts treated as terminal (tterm). 40 epochs, SEEDS seeds each;
# one kernel per factor and method, named meda-abl-<factor>-<method>.
# At most MAX GPU kernels per account run at once; a push refused for lack of a free GPU
# session is retried every POLL seconds. ONLY / SKIP are regexes over the kernel names, so the
# kernels can be split between two accounts (run one queue per account, e.g. with HOME set to
# a directory whose .kaggle/ holds the other account's credentials).
#   nohup bash kaggle/push_ablation.sh > kaggle/build/push_ablation.log 2>&1 &
# Download with kaggle/fetch_ablation.sh.
set -uo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-3}
EPOCHS=${EPOCHS:-40}
MAX=${MAX:-2}
POLL=${POLL:-300}
ONLY=${ONLY:-.}
SKIP=${SKIP:-^$}
USER_NAME=${KAGGLE_USERNAME:-$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi
now() { date '+%F %H:%M'; }

build() {  # build <slug> <config> <label>
  local slug=$1 dir=build/$1
  mkdir -p "$dir"
  python3 - "(\"$2\", \"$3\")" "$SEEDS" "--set schedule.epochs=$EPOCHS" > "$dir/run.py" <<'PY'
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
for f in base marks kmaxchip alpha2 sizes5 tterm; do
  for m in cnn dirgcn; do
    slug=meda-abl-$f-$m
    [[ $slug =~ $ONLY && ! $slug =~ $SKIP ]] || continue
    if [[ $f == base ]]; then
      [[ $m == cnn ]] && cfg=paper_30x30_healthy || cfg=gnn_dirgcn_30x30
    else
      cfg=ablate_${f}_${m}_30x30
    fi
    [[ $m == cnn ]] && label=CNN-PPO || label=DirGCN-maxpool-PPO
    build "$slug" "$cfg" "$label"
    queue+=("$slug")
  done
done
echo "$(now) account $USER_NAME, ${#queue[@]} kernels: ${queue[*]}"

busy() {  # busy <slug>: queued, running, or status not readable yet (new private kernel)
  case "$(kaggle kernels status "$USER_NAME/$1" 2>&1 || true)" in
    *[Rr][Uu][Nn][Nn][Ii][Nn][Gg]*|*[Qq][Uu][Ee][Uu][Ee][Dd]*|*denied*|*Permission*) return 0 ;;
  esac
  return 1
}

running=()
while (( ${#queue[@]} )) || (( ${#running[@]} )); do
  still=()
  for k in "${running[@]}"; do
    if busy "$k"; then still+=("$k"); else echo "$(now) finished: $k"; fi
  done
  running=("${still[@]}")
  while (( ${#queue[@]} )) && (( ${#running[@]} < MAX )); do
    k=${queue[0]}
    echo "$(now) pushing $USER_NAME/$k"
    out=$(kaggle kernels push -p "build/$k" 2>&1 || true)
    echo "$out"
    if [[ "$out" == *"successfully pushed"* ]]; then
      queue=("${queue[@]:1}"); running+=("$k"); sleep 60
    else
      echo "$(now) push of $k refused; retrying in $POLL s"; break
    fi
  done
  (( ${#queue[@]} || ${#running[@]} )) && { echo "$(now) running: ${running[*]:-none} | queued: ${queue[*]:-none}"; sleep "$POLL"; }
done
echo "$(now) all kernels finished"
