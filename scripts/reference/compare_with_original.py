"""Compare training runs of the authors' original code with runs of this reimplementation.

    python scripts/reference/compare_with_original.py \
        --original <dir with progress.csv> [...] --ours <dir with progress.csv> [...] \
        [--authors-log <MEDA clone>/policy/0825a_030x030_E100_NPS64.pickle] --out results/reference_test

Both sides evaluate the deterministic policy on 500 jobs after every 2^14-step epoch.
Outputs in --out:
  per_run.csv       one row per run: epochs to 90% / 95% success, convergence epoch
                    (first of three consecutive evaluations >= 95%), final success
                    and cycles (mean of the last 5 epochs)
  summary.csv       the same, mean and standard deviation per implementation
  curves.csv        success rate and cycles per epoch of every run
  learning_curves.png  success rate and cycles vs. epoch, mean and min-max band
"""
from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_run(path: Path) -> pd.DataFrame:
    h = pd.read_csv(path / "progress.csv")
    return h[["epoch", "success_rate", "mean_cycles"]].sort_values("epoch").reset_index(drop=True)


def load_authors_log(path: Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    goals = np.asarray(d["a_goals"][0], dtype=float) / 100
    cycles = np.asarray(d["a_cycles"][0], dtype=float)
    return pd.DataFrame({"epoch": np.arange(1, len(goals) + 1), "success_rate": goals, "mean_cycles": cycles})


def first_epoch(success: np.ndarray, epochs: np.ndarray, level: float, run: int = 1):
    """First epoch that starts `run` consecutive evaluations at or above `level`."""
    ok = success >= level - 1e-9
    for i in range(len(ok) - run + 1):
        if ok[i:i + run].all():
            return int(epochs[i])
    return np.nan


def metrics(h: pd.DataFrame, last: int = 5) -> dict:
    s, e = h["success_rate"].to_numpy(), h["epoch"].to_numpy()
    return {
        "epochs": int(e[-1]),
        "epoch_90": first_epoch(s, e, 0.90),
        "epoch_95": first_epoch(s, e, 0.95),
        "converged_epoch": first_epoch(s, e, 0.95, run=3),
        "final_success": float(s[-last:].mean()),
        "final_cycles": float(h["mean_cycles"].to_numpy()[-last:].mean()),
        "best_success": float(s.max()),
    }


def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p)) or [p]
        out += [Path(x) for x in hits if (Path(x) / "progress.csv").exists()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", nargs="+", required=True)
    ap.add_argument("--ours", nargs="+", required=True)
    ap.add_argument("--authors-log", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = {"original code": expand(a.original), "reimplementation": expand(a.ours)}
    rows, curves = [], []
    for impl, paths in groups.items():
        if not paths:
            raise SystemExit(f"no runs with progress.csv for {impl}: {a.original if impl == 'original code' else a.ours}")
        for p in paths:
            h = load_run(p)
            rows.append({"implementation": impl, "run": str(p), **metrics(h)})
            curves.append(h.assign(implementation=impl, run=str(p)))
    authors = load_authors_log(Path(a.authors_log)) if a.authors_log else None
    if authors is not None:
        rows.append({"implementation": "authors' published log", "run": a.authors_log, **metrics(authors.iloc[:40])})

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    pd.concat(curves).to_csv(out / "curves.csv", index=False)
    cols = ["epoch_90", "epoch_95", "converged_epoch", "final_success", "final_cycles", "best_success"]
    agg = per_run.groupby("implementation", sort=False)[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.to_csv(out / "summary.csv")
    pd.set_option("display.width", 200)
    print(per_run.drop(columns="run").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"original code": "#A9521A", "reimplementation": "#2563AD"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
    for impl, paths in groups.items():
        c = pd.concat(x for x in curves if x["implementation"].iat[0] == impl)
        g = c.groupby("epoch")
        for ax, col, scale in ((axes[0], "success_rate", 100), (axes[1], "mean_cycles", 1)):
            m = g[col].mean() * scale
            ax.fill_between(m.index, g[col].min() * scale, g[col].max() * scale, color=colours[impl], alpha=0.18, lw=0)
            ax.plot(m.index, m.values, color=colours[impl], lw=2, label=f"{impl} (n={len(paths)})")
    if authors is not None:
        x = authors.iloc[:40]
        axes[0].plot(x["epoch"], x["success_rate"] * 100, color="#5B6170", lw=1.3, ls=(0, (4, 3)), label="authors' published log")
        axes[1].plot(x["epoch"], x["mean_cycles"], color="#5B6170", lw=1.3, ls=(0, (4, 3)))
    axes[0].axhline(95, color="#8A8F98", lw=0.8, ls=":")
    axes[0].set_ylabel("Evaluation success rate (%)")
    axes[0].set_ylim(0, 102)
    axes[1].set_ylabel("Mean cycles per job")
    for ax in axes:
        ax.set_xlabel("Epoch ($2^{14}$ environment steps)")
        ax.grid(axis="y", color="#E3E0D8", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
