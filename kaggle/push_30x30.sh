#!/usr/bin/env bash
# Push one Kaggle kernel per 30x30 configuration (CNN-PPO baseline, GCN + max
# pooling, direction-aware GCN + max pooling). Run on a machine where the Kaggle
# CLI is logged in:
#
#   bash kaggle/push_30x30.sh            # 1 seed per configuration
#   SEEDS=5 bash kaggle/push_30x30.sh    # paper protocol (may exceed Kaggle's 12 h limit)
#   ONLY=dirgcn bash kaggle/push_30x30.sh
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

# slug | config | label
JOBS="cnn|paper_30x30_healthy|CNN-PPO
gcn|gnn_maxpool_30x30|GCN-maxpool-PPO
dirgcn|gnn_dirgcn_30x30|DirGCN-maxpool-PPO"

while IFS='|' read -r slug cfg label; do
  [[ -n "$ONLY" && "$ONLY" != "$slug" ]] && continue
  dir=build/meda-30x30-$slug
  mkdir -p "$dir"
  sed -e "s/__CONFIG__/$cfg/" -e "s/__LABEL__/$label/" -e "s/__SEEDS__/$SEEDS/" run_kernel.py > "$dir/run.py"
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
  echo "== pushing $USER_NAME/meda-30x30-$slug ($cfg, $SEEDS seed(s))"
  kaggle kernels push -p "$dir"
done <<< "$JOBS"
echo "Check progress: kaggle kernels status $USER_NAME/meda-30x30-<cnn|gcn|dirgcn>"
