#!/usr/bin/env bash
# Download the outputs of the chip-size kernels (kaggle/push_sizes.sh) and compare
# the CNN-PPO baseline with the direction-aware GCN on the same held-out jobs of
# each size. push_sizes.sh runs it for each size as soon as both kernels have
# finished; run it by hand from anywhere to redo a size:
#
#   bash kaggle/fetch_sizes.sh
#   SIZES="50 60" bash kaggle/fetch_sizes.sh
#
# Outputs: kaggle/output/chip_size/<N>x<N>-{cnn,dirgcn}/ (kernel log, runs/ and the
# kernel's own evaluation in results/), results/chip_size/<N>x<N>/ (meda
# compare-methods) and results/chip_size/summary.csv (one row per size and method,
# rebuilt from every size compared so far).
set -euo pipefail
cd "$(dirname "$0")/.."
SIZES=${SIZES:-"30 50 60 100 120"}
[[ -x .venv/bin/meda ]] && PATH="$PWD/.venv/bin:$PATH"
USER_NAME=${KAGGLE_USERNAME:-$(python3 - <<'PY'
import json, os
for p in ("~/.kaggle/kaggle.json", "~/.config/kaggle/kaggle.json"):
    p = os.path.expanduser(p)
    if os.path.exists(p):
        print(json.load(open(p))["username"]); break
PY
)}
if [[ -z "$USER_NAME" ]]; then echo "Set KAGGLE_USERNAME" >&2; exit 1; fi

for N in $SIZES; do
  for m in cnn dirgcn; do
    slug="meda-size-${N}x${N}-$m"
    out="kaggle/output/chip_size/${N}x${N}-$m"
    kaggle kernels status "$USER_NAME/$slug" || true
    mkdir -p "$out"
    kaggle kernels output "$USER_NAME/$slug" -p "$out" || echo "!! could not download $slug" >&2
  done
  cnn="kaggle/output/chip_size/${N}x${N}-cnn/runs/paper_${N}x${N}_healthy"
  dir="kaggle/output/chip_size/${N}x${N}-dirgcn/runs/gnn_dirgcn_${N}x${N}"
  if [[ -d "$cnn" && -d "$dir" ]]; then
    meda compare-methods --method CNN-PPO "$cnn" --method DirGCN-maxpool-PPO "$dir" \
      --episodes 500 --out "results/chip_size/${N}x${N}"
  else
    echo "!! ${N}x${N}: missing run folder(s), skipping the comparison" >&2
  fi
done

python3 - <<'PY'
# results/chip_size/summary.csv: the comparison of every size, plus how many epochs
# each run trained and whether the time limit (schedule.max_hours) stopped it.
import csv, glob, json, os, re
rows = []
for table in sorted(glob.glob("results/chip_size/*x*/tables/main_comparison.csv"),
                    key=lambda p: int(re.search(r"/(\d+)x\d+/", p).group(1))):
    size = table.split("/")[2]
    folder = {"CNN-PPO": f"{size}-cnn", "DirGCN-maxpool-PPO": f"{size}-dirgcn"}
    for row in csv.DictReader(open(table)):
        epochs, stopped = [], []
        for s in glob.glob(f"kaggle/output/chip_size/{folder.get(row['method'], '?')}/runs/*/seed_*/summary.json"):
            info = json.load(open(s))
            epochs.append(str(info.get("epochs")))
            stopped.append(str(info.get("stopped_by_time_limit", False)))
        rows.append({"chip": size, **row, "epochs_trained": ";".join(epochs),
                     "stopped_by_time_limit": ";".join(stopped)})
if rows:
    os.makedirs("results/chip_size", exist_ok=True)
    with open("results/chip_size/summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"results/chip_size/summary.csv: {len(rows)} rows")
PY
