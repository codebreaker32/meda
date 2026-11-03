#!/usr/bin/env bash
# CNN-PPO baseline with the five droplet sizes 4x4 to 6x6 of the authors' logged run, on 30x30
# (ablate_sizes5_cnn_30x30.yaml) and 60x60 (sizes5_cnn_60x60.yaml) chips: SEEDS seeds, EPOCHS
# epochs, one GPU kernel per size, named meda-s5-<N>x<N>-cnn. The kernels clone the branch, so
# push the configs to GitHub first.
#   bash kaggle/push_sizes5.sh                 # push both kernels
#   FETCH=1 bash kaggle/push_sizes5.sh         # download finished kernels to kaggle/output/sizes5/
set -uo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-5}
EPOCHS=${EPOCHS:-40}
USER_NAME=${KAGGLE_USERNAME:-$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi
declare -A CFG=([30x30]=ablate_sizes5_cnn_30x30 [60x60]=sizes5_cnn_60x60)
for n in 30x30 60x60; do
  slug=meda-s5-$n-cnn
  if [[ -n "${FETCH:-}" ]]; then
    st=$(kaggle kernels status "$USER_NAME/$slug" 2>&1 | tail -1)
    [[ $st == *COMPLETE* ]] || { echo "skip $slug: $st"; continue; }
    mkdir -p "output/sizes5/$slug"
    kaggle kernels output "$USER_NAME/$slug" -p "output/sizes5/$slug" -q \
      --file-pattern '^(?!.*\.zip$).*$' && echo "fetched $slug"
    continue
  fi
  dir=build/$slug
  mkdir -p "$dir"
  python3 - "(\"${CFG[$n]}\", \"CNN-PPO\")" "$SEEDS" "--set schedule.epochs=$EPOCHS" > "$dir/run.py" <<'PY'
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
  out=$(kaggle kernels push -p "$dir" 2>&1 || true)
  echo "$out" | grep -v outdated
  [[ "$out" == *"successfully pushed"* ]] || echo "push of $slug FAILED"
  sleep 30
done
