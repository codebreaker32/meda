#!/usr/bin/env bash
# Push the 30x30 comparison as two Kaggle kernels (Kaggle runs at most two at a
# time): "cnn" trains the CNN-PPO baseline; "gcn" trains GCN + max pooling and
# then direction-aware GCN + max pooling. Run where the Kaggle CLI is logged in:
#
#   bash kaggle/push_30x30.sh            # 1 seed per configuration
#   SEEDS=5 bash kaggle/push_30x30.sh    # paper protocol (may exceed Kaggle's 12 h limit)
#   ONLY=gcn bash kaggle/push_30x30.sh
#
# Kernels need "Internet" (phone-verified Kaggle account) to clone the repository.
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-1}
ONLY=${ONLY:-}
USER_NAME=${KAGGLE_USERNAME:-$(python3 - <<'PY'
import json, os
for p in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
    p = os.path.expanduser(p)
    if os.path.exists(p):
        print(json.load(open(p))["username"]); break
PY
)}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi

# slug | python list of (config, label) trained in order
JOBS="cnn|(\"paper_30x30_healthy\", \"CNN-PPO\")
gcn|(\"gnn_maxpool_30x30\", \"GCN-maxpool-PPO\"), (\"gnn_dirgcn_30x30\", \"DirGCN-maxpool-PPO\")"

while IFS='|' read -r slug jobs; do
  [[ -n "$ONLY" && "$ONLY" != "$slug" ]] && continue
  dir=build/meda-30x30-$slug
  mkdir -p "$dir"
  python3 - "$jobs" "$SEEDS" > "$dir/run.py" <<'PY'
import sys
src = open("run_kernel.py").read()
print(src.replace("__JOBS__", sys.argv[1]).replace("__SEEDS__", sys.argv[2]), end="")
PY
  cat > "$dir/kernel-metadata.json" <<JSON
{
  "id": "$USER_NAME/meda-30x30-$slug",
  "title": "meda-30x30-$slug",
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
  echo "== pushing $USER_NAME/meda-30x30-$slug ($SEEDS seed(s))"
  kaggle kernels push -p "$dir"
done <<< "$JOBS"
echo "Check progress: kaggle kernels status $USER_NAME/meda-30x30-<cnn|gcn>"
