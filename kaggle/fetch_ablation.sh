#!/usr/bin/env bash
# Download the settings-ablation kernels of push_ablation.sh (logs, progress, held-out
# results; no model files) into kaggle/output/ablation/<kernel>/.
#   bash kaggle/fetch_ablation.sh                 # the signed-in account's kernels
#   ONLY=cnn bash kaggle/fetch_ablation.sh        # regex over kernel names
set -uo pipefail
cd "$(dirname "$0")/.."
ONLY=${ONLY:-.}
USER_NAME=${KAGGLE_USERNAME:-$(kaggle config view 2>/dev/null | sed -n 's/^- username: //p')}
for f in base marks kmaxchip alpha2 sizes5 tterm; do
  for m in cnn dirgcn; do
    k=meda-abl-$f-$m
    [[ $k =~ $ONLY ]] || continue
    st=$(kaggle kernels status "$USER_NAME/$k" 2>&1 | tail -1)
    [[ $st == *COMPLETE* ]] || { echo "skip $k: $st"; continue; }
    mkdir -p "kaggle/output/ablation/$k"
    kaggle kernels output "$USER_NAME/$k" -p "kaggle/output/ablation/$k" -q \
      --file-pattern '^(?!.*\.zip$).*$' && echo "fetched $k"
  done
done
