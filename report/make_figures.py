"""Regenerate the data-driven figures of the report (run from the repository root).

    python report/make_figures.py

* figures/routing_example.png: a 16x16 routing job in the simulator, drawn
  at its start (droplet blue, goal green, fully degraded MCs red, routing
  zone light grey), with the path of the trained direction-aware GNN agent
  (runs/gnn_dirgcn_16x16/seed_0) in orange;
* figures/routing_observation.png: the same job without the path (the
  observation panel of the pipeline figure);
* figures/learning_curves.png: per-epoch evaluation success of the three
  compared configurations.
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
    """figures/learning_curves_30x30.png: 30x30 study (Kaggle runs, kaggle/output/)."""
    _curves("learning_curves_30x30.png", [
        ("kaggle/output/gcn/runs/gnn_maxpool_30x30", "GCN + max pooling", "#7D838C", 1.6),
        ("kaggle/output/cnn/runs/paper_30x30_healthy", r"CNN$\endash$PPO (baseline)", "#A9521A", 2.0),
        ("kaggle/output/gcn/runs/gnn_dirgcn_30x30", "Direction-aware GCN\n+ max pooling", "#2563AD", 2.0),
    ], 420, {"kaggle/output/gcn/runs/gnn_dirgcn_30x30": -8})


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    routing_example()
    learning_curves()
    learning_curves_30x30()
