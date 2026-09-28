"""Regenerate the data-driven figures of the report (run from the repository root).

    python report/make_figures.py

* figures/routing_example.png: a 16x16 routing job in the simulator, drawn
  at its start (droplet blue, goal green, fully degraded MCs red, routing
  zone light grey), with the path of the trained direction-aware GNN agent
  (runs/gnn_dirgcn_16x16/seed_0) in orange;
* figures/success_rate_vs_env_steps.png: copied from results/figures/.
"""

from __future__ import annotations

import shutil
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

    fig = Figure(figsize=(3.2, 3.2), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(start, extent=(-0.5, 15.5, -0.5, 15.5), origin="upper", interpolation="nearest")
    ax.plot(path[:, 0], path[:, 1], color="#eb6834", lw=2.2, marker="o", ms=3)
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (goal.xa - 0.5, goal.ya - 0.5), goal.width, goal.height, fill=False, ec="#0b0b0b", lw=1.2, ls="--"))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 15.5)
    ax.set_axis_off()
    fig.savefig(FIG / "routing_example.png")
    print(f"routing_example.png: {info['cycles']} cycles, success={info['success']}")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    routing_example()
    shutil.copy(ROOT / "results" / "figures" / "success_rate_vs_env_steps.png", FIG)
