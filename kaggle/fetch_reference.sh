#!/usr/bin/env bash
# Download the reference-test kernels (kaggle/push_reference.sh) and compare the
# original code with this reimplementation (scripts/reference/compare_with_original.py).
#   bash kaggle/fetch_reference.sh            # ORIG_SEEDS as for the push
#   AUTHORS_LOG=<MEDA clone>/policy/0825a_030x030_E100_NPS64.pickle bash kaggle/fetch_reference.sh  # add their log
set -euo pipefail
cd "$(dirname "$0")/.."
ORIG_SEEDS=${ORIG_SEEDS:-"0 1 2"}
USER_NAME=${KAGGLE_USERNAME:-$(python3 - <<'PY'
import json, os
for p in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
    p = os.path.expanduser(p)
    if os.path.exists(p):
        print(json.load(open(p))["username"]); break
PY
)}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi
for k in meda-ref-ours $(for s in $ORIG_SEEDS; do echo "meda-ref-orig-s$s"; done); do
  kaggle kernels status "$USER_NAME/$k" || true
  mkdir -p "kaggle/output/reference/$k"
  kaggle kernels output "$USER_NAME/$k" -p "kaggle/output/reference/$k"
done
python3 scripts/reference/compare_with_original.py ${AUTHORS_LOG:+--authors-log "$AUTHORS_LOG"} \
  --original kaggle/output/reference/meda-ref-orig-s*/original/seed_* \
  --ours kaggle/output/reference/meda-ref-ours/runs/reference_0825a_30x30/seed_* \
  --out results/reference_test
