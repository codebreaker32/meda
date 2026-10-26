"""Figures of the IEEE paper (run from the repository root):

    python3 report/ieee/make_ieee_figures.py

* figures/curves_30x30.png: per-epoch evaluation success on 30x30 chips against environment
  steps: five seeds of the CNN baseline and of the direction-aware GCN (25 epochs; line =
  mean, band = min-max) and the 100-epoch run of the baseline.
* figures/chip_size.png: (a) held-out success against chip size (mean and min-max over the
  available seeds; 40 epochs), (b) mean seconds per training epoch on a Tesla T4.
* figures/reference_test.png: copied from report/figures/ (comparison with the authors'
  original implementation).
"""

from __future__ import annotations

import csv
import shutil
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent / "figures"
A100 = ROOT / "kaggle" / "output" / "a100"
KAGGLE = ROOT / "kaggle" / "output" / "chip_size"
CNN_C, GCN_C = "#A9521A", "#2563AD"

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8})


def _tidy(ax) -> None:
    ax.tick_params(labelsize=7.5, length=2.5, color="#9A968C")
    ax.grid(axis="y", color="#E3E0D8", lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9A968C")


def _progress(run: Path):
    rows = list(csv.DictReader(open(run / "progress.csv")))
    return [int(r["timesteps"]) / 1000 for r in rows], [100 * float(r["success_rate"]) for r in rows], rows


def curves_30x30() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.3), dpi=300)
    for name, colour, label in (("paper_30x30_healthy", CNN_C, "CNN baseline (5 seeds)"),
                                ("gnn_dirgcn_30x30", GCN_C, "Direction-aware GCN (5 seeds)")):
        runs = sorted((A100 / "out" / "runs" / name).glob("seed_*"))
        series = [_progress(r) for r in runs]
        x = series[0][0]
        ys = list(zip(*[s[1] for s in series]))
        ax.fill_between(x, [min(v) for v in ys], [max(v) for v in ys], color=colour, alpha=0.18, lw=0)
        ax.plot(x, [st.mean(v) for v in ys], color=colour, lw=1.4, label=label)
    x, y, _ = _progress(A100 / "out" / "runs" / "paper_30x30_healthy_e100" / "seed_0")
    ax.plot(x, y, color=CNN_C, lw=1.0, ls=(0, (3, 2)), label="CNN baseline, 100 epochs (1 seed)")
    ax.axhline(95, color="#8A8F98", lw=0.6, ls=":")
    ax.set_xlim(0, 1650)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Evaluation success rate (%)")
    _tidy(ax)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "curves_30x30.png")
    plt.close(fig)


def _held_out(n: int, method: str) -> list[float]:
    """Held-out success (%) of every 40-epoch run of one method at one chip size."""
    vals = []
    slug = "cnn" if method == "CNN-PPO" else "dirgcn"
    if n in (100, 120) and method != "CNN-PPO":  # full 40-epoch reruns on the A100
        f = A100 / "out_full" / "results" / f"dirgcn{n}" / "tables" / "per_seed_results.csv"
    else:
        f = next((KAGGLE / f"{n}x{n}-{slug}").glob("results/*/tables/per_seed_results.csv"))
    vals += [100 * float(r["success_rate"]) for r in csv.DictReader(open(f))]
    extra = A100 / "out" / "results" / f"healthy{n}" / "tables" / "per_seed_results.csv"
    if n in (50, 60):  # seeds 1-4 at 50x50 and 60x60 (healthy30 holds 25-epoch runs, not used here)
        vals += [100 * float(r["success_rate"]) for r in csv.DictReader(open(extra)) if r["method"] == method]
    return vals


def chip_size() -> None:
    sizes = [30, 50, 60, 100, 120]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), dpi=300)
    for method, slug, colour, marker, label in (("CNN-PPO", "cnn", CNN_C, "s", "CNN baseline"),
                                               ("DirGCN-maxpool-PPO", "dirgcn", GCN_C, "o", "Direction-aware GCN")):
        vals = [_held_out(n, method) for n in sizes]
        mean = [st.mean(v) for v in vals]
        err = [[m - min(v) for m, v in zip(mean, vals)], [max(v) - m for m, v in zip(mean, vals)]]
        a.errorbar(sizes, mean, yerr=err, color=colour, marker=marker, ms=3.5, lw=1.3, capsize=2, label=label)
        secs = [st.mean(float(r["epoch_seconds"]) for r in _progress(next((KAGGLE / f"{n}x{n}-{slug}").glob("runs/*/seed_0")))[2])
                for n in sizes]
        b.plot(sizes, secs, color=colour, marker=marker, ms=3.5, lw=1.3, label=label)
    a.set_ylim(0, 104)
    a.set_ylabel("Held-out success rate (%)")
    b.set_yscale("log")
    b.set_ylabel("Seconds per training epoch")
    for ax, tag in ((a, "(a)"), (b, "(b)")):
        ax.set_xlabel(r"Chip size $N$ ($N\times N$ MCs)")
        ax.set_xticks(sizes)
        ax.set_title(tag, fontsize=8, loc="left")
        _tidy(ax)
    a.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(pad=0.3, w_pad=2.0)
    fig.savefig(FIG / "chip_size.png")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    curves_30x30()
    chip_size()
    shutil.copy(ROOT / "report" / "figures" / "reference_test.png", FIG / "reference_test.png")
    print("written to", FIG)
