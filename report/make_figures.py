"""Regenerate the data-driven figures of the report (run from the repository root).

    python report/make_figures.py

* figures/routing_example.png: a 16x16 routing job in the simulator, drawn
  at its start (droplet blue, goal green, fully degraded MCs red, routing
  zone light grey), with the path of the trained direction-aware GCN agent
  (runs/gnn_dirgcn_16x16/seed_0) in orange. The chip has 10% faulty MCs, although
  the agent was trained on healthy chips;
* figures/routing_observation.png: the same job without the path (the
  observation panel of the pipeline figure);
* figures/learning_curves.png: per-epoch evaluation success of the three
  compared configurations (16x16 study);
* figures/learning_curves_30x30.png: the same on 30x30 chips, the CNN and the
  direction-aware GCN from their 40-epoch runs;
* figures/chip_size.png: held-out success and seconds per epoch against chip
  size (results/chip_size/, kaggle/output/chip_size/);
* figures/reference_test.png: the reference test against the authors' original
  implementation (results/reference_test/curves.csv).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from meda_routing.envs import MEDARoutingEnv
from meda_routing.routers.drl import DRLRouter
from meda_routing.viz.animation import record_episode

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"
MODEL = ROOT / "runs" / "gnn_dirgcn_16x16" / "seed_0"
SEED = 1  # a job whose direct path is blocked by degraded MCs


def routing_example() -> None:
    router = DRLRouter.load(MODEL)
    env = MEDARoutingEnv({"width": 16, "height": 16, "obs_size": None, "fault_fraction": 0.1})

    def policy(obs, env):
        action, _ = router.model.predict(obs, deterministic=True)
        return int(action)

    env.reset(seed=SEED)
    start = env.render_frame(scale=1)
    info = record_episode(env, policy, FIG / "_tmp.gif", seed=SEED, record_path=True, end_pause=0)
    (FIG / "_tmp.gif").unlink()
    path = np.array([((d.xa + d.xb) / 2, (d.ya + d.yb) / 2) for d in info["path"]])
    goal = info["path"][-1]

    from matplotlib.patches import Rectangle

    for name, with_path in (("routing_example.png", True), ("routing_observation.png", False)):
        fig = Figure(figsize=(3.2, 3.2), dpi=200)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(start, extent=(-0.5, 15.5, -0.5, 15.5), origin="upper", interpolation="nearest")
        if with_path:
            ax.plot(path[:, 0], path[:, 1], color="#eb6834", lw=2.2, marker="o", ms=3)
        ax.add_patch(Rectangle((goal.xa - 0.5, goal.ya - 0.5), goal.width, goal.height,
                               fill=False, ec="#0b0b0b", lw=1.2, ls="--"))
        ax.set_xlim(-0.5, 15.5)
        ax.set_ylim(-0.5, 15.5)
        ax.set_axis_off()
        fig.savefig(FIG / name)
    print(f"routing_example.png: {info['cycles']} cycles, success={info['success']}")


def _curves(out: str, series, xlim: float, offsets: dict) -> None:
    """Per-epoch evaluation success of the compared configurations, one line each."""
    import matplotlib
    import pandas as pd

    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                                "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
    fig = Figure(figsize=(6.0, 3.2), dpi=300)
    ax = fig.add_axes((0.10, 0.16, 0.62, 0.79))
    for run, label, colour, lw in series:
        h = pd.read_csv(ROOT / run / "seed_0" / "progress.csv")
        x, y = h["timesteps"] / 1000, h["success_rate"] * 100
        ax.plot(x, y, color=colour, lw=lw, solid_capstyle="round")
        ax.annotate(label, (x.iloc[-1], y.iloc[-1]), xytext=(8, offsets.get(run, 0)),
                    textcoords="offset points", va="center", fontsize=9, color="#1F2933")
    ax.axhline(95, color="#8A8F98", lw=0.9, ls=(0, (4, 3)))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Environment steps (thousands)", fontsize=9.5)
    ax.set_ylabel("Evaluation success rate (%)", fontsize=9.5)
    ax.tick_params(labelsize=9, length=2.5, color="#9A968C")
    ax.grid(axis="y", color="#E3E0D8", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9A968C")
    fig.savefig(FIG / out)
    print(f"{out} written")


def learning_curves() -> None:
    """figures/learning_curves.png: 16x16 study (runs/ of the CPU experiments)."""
    _curves("learning_curves.png", [  # colours validated for CVD separation
        ("runs/gnn_maxpool_16x16", "GCN + max pooling", "#7D838C", 1.6),
        ("runs/cnn_16x16", r"CNN$\endash$PPO (baseline)", "#A9521A", 2.0),
        ("runs/gnn_dirgcn_16x16", "Direction-aware GCN\n+ max pooling", "#2563AD", 2.0),
    ], 340, {"runs/cnn_16x16": -11, "runs/gnn_dirgcn_16x16": 4})


def learning_curves_30x30() -> None:
    """figures/learning_curves_30x30.png: 30x30 study (Kaggle runs). The CNN and the
    direction-aware GCN come from the 40-epoch runs (kaggle/output/chip_size/30x30-*), GCN +
    max pooling from its 25-epoch run (kaggle/output/gcn/); it was not run for 40 epochs."""
    _curves("learning_curves_30x30.png", [
        ("kaggle/output/gcn/runs/gnn_maxpool_30x30", "GCN + max pooling\n(25 epochs)", "#7D838C", 1.6),
        ("kaggle/output/chip_size/30x30-cnn/runs/paper_30x30_healthy", r"CNN$\endash$PPO (baseline)", "#A9521A", 2.0),
        ("kaggle/output/chip_size/30x30-dirgcn/runs/gnn_dirgcn_30x30", "Direction-aware GCN\n+ max pooling", "#2563AD", 2.0),
    ], 680, {"kaggle/output/chip_size/30x30-cnn/runs/paper_30x30_healthy": -14,
             "kaggle/output/chip_size/30x30-dirgcn/runs/gnn_dirgcn_30x30": -2,
             "kaggle/output/gcn/runs/gnn_maxpool_30x30": 6})


def _style():
    import matplotlib

    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
                                "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})


def _tidy(ax) -> None:
    ax.tick_params(labelsize=8.5, length=2.5, color="#9A968C")
    ax.grid(axis="y", color="#E3E0D8", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9A968C")


CHIP_METHODS = (("CNN-PPO", "cnn", r"CNN$\endash$PPO (baseline)", "#A9521A", "s"),
                ("DirGCN-maxpool-PPO", "dirgcn", "Direction-aware GCN + max pooling", "#2563AD", "o"))


def chip_size() -> None:
    """figures/chip_size.png (from kaggle/output/chip_size/): (a) held-out success and (b) mean seconds per training epoch
    (2^14 steps plus the per-epoch evaluation, Kaggle T4) against chip size. Hollow
    markers: runs stopped by the 11 h time limit before their 40 epochs."""
    import pandas as pd

    _style()
    import json

    out = ROOT / "kaggle" / "output" / "chip_size"
    sizes = sorted({int(d.name.split("x")[0]) for d in out.iterdir() if d.is_dir()})
    fig = Figure(figsize=(6.4, 2.75), dpi=300)
    ax1 = fig.add_axes((0.085, 0.17, 0.37, 0.70))
    ax2 = fig.add_axes((0.595, 0.17, 0.37, 0.70))
    for method, slug, label, colour, marker in CHIP_METHODS:
        # each kernel's own held-out evaluation (the same 500 jobs as `meda compare-methods`)
        success, secs, stopped = [], [], []
        for n in sizes:
            d = out / f"{n}x{n}-{slug}"
            success.append(float(pd.read_csv(next(d.glob("results/*/tables/main_comparison.csv")))["success_rate_pct"].iloc[0]))
            secs.append(pd.read_csv(next(d.glob("runs/*/seed_0/progress.csv")))["epoch_seconds"].mean())
            stopped.append(bool(json.load(open(next(d.glob("runs/*/seed_0/summary.json")))).get("stopped_by_time_limit")))
        stopped = np.array(stopped)
        for ax, y in ((ax1, np.array(success)), (ax2, np.array(secs))):
            x = np.array(sizes)
            ax.plot(x, y, color=colour, lw=1.6, label=label)
            ax.scatter(x[~stopped], y[~stopped], color=colour, marker=marker, s=18, zorder=3)
            ax.scatter(x[stopped], y[stopped], facecolor="white", edgecolor=colour, marker=marker, s=22, lw=1.2, zorder=3)
    ax1.set_ylim(0, 104)
    ax1.set_ylabel("Held-out success rate (%)", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_ylabel("Seconds per epoch", fontsize=9)
    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        ax.set_xlabel("Chip size $N$ ($N\\times N$ MCs)", fontsize=9)
        ax.set_xticks(sizes)
        ax.set_title(tag, fontsize=9, loc="left")
        _tidy(ax)
    ax1.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.savefig(FIG / "chip_size.png")
    print("chip_size.png written")


def reference_test() -> None:
    """figures/reference_test.png: per-epoch success and cycles of the authors' original
    implementation and of ours (3 seeds each): mean and min-max band."""
    import pandas as pd

    _style()
    c = pd.read_csv(ROOT / "results" / "reference_test" / "curves.csv")
    series = (("original code", "Authors' implementation", "#A9521A"),
              ("reimplementation", "Our implementation (Table I CNN)", "#2563AD"),
              ("reimplementation, network and decay of 1667016", "Our implementation (authors' CNN and decay)", "#2E8B57"))
    fig = Figure(figsize=(6.4, 3.0), dpi=300)
    axes = (fig.add_axes((0.085, 0.29, 0.37, 0.66)), fig.add_axes((0.595, 0.29, 0.37, 0.66)))
    for impl, label, colour in series:
        g = c[c["implementation"] == impl].groupby("epoch")
        for ax, col, scale in ((axes[0], "success_rate", 100), (axes[1], "mean_cycles", 1)):
            m = g[col].mean() * scale
            ax.fill_between(m.index, g[col].min() * scale, g[col].max() * scale, color=colour, alpha=0.15, lw=0)
            ax.plot(m.index, m.values, color=colour, lw=1.6, label=label)
    axes[0].axhline(95, color="#8A8F98", lw=0.8, ls=(0, (4, 3)))
    axes[0].set_ylim(0, 102)
    axes[0].set_ylabel("Evaluation success rate (%)", fontsize=9)
    axes[1].set_ylabel("Mean cycles per job", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Epoch ($2^{14}$ environment steps)", fontsize=9)
        _tidy(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, loc="lower center", ncol=3)
    fig.savefig(FIG / "reference_test.png")
    print("reference_test.png written")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    routing_example()
    learning_curves()
    learning_curves_30x30()
    chip_size()
    reference_test()
