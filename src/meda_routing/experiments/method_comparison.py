"""CNN-PPO vs. GNN-PPO comparison: the tables, logs and figures of
docs/GNN_METHODOLOGY.md, section 8.

``meda compare-methods`` takes trained runs (one run folder per method,
holding one ``seed_*`` folder per seed), evaluates every seed's model on the
same held-out jobs, and writes into ``results/`` of the project folder::

    results/tables/experiment_config.csv   what each method/seed was trained with
    results/tables/main_comparison.csv     one row per method (mean +- std over seeds)
    results/tables/graph_validation.csv    checks of the graph representation
    results/logs/training_metrics.csv      one row per method/seed/epoch
    results/logs/evaluation_jobs.csv       one row per method/seed/job/repeat
    results/figures/*.png                  success rate and cycles vs. env steps

Definitions (the same for every method):

* held-out jobs: evaluation environments seeded with ``--seed`` (default
  20000 [ASSUMED]), different from the per-epoch evaluation seed used to pick
  ``best_model.zip`` (10000) and from the training seeds; every method and
  seed sees exactly the same jobs and chips;
* cycle statistics count a timed-out job at ``k_max`` cycles; the
  "successful jobs" variants use successful jobs only;
* convergence point: the first evaluation after which the per-epoch success
  rate stays >= ``--converge-at`` (default 0.95 [ASSUMED]) for
  ``--converge-window`` (default 3 [ASSUMED]) consecutive evaluations,
  reported in environment steps and optimizer steps (empty if never reached);
* inference time: mean wall-clock time of one ``model.predict`` call on a
  single observation (one routing decision) on the stated device.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from ..agents import get_extractor
from ..devices import resolve_device
from ..envs.meda_env import MEDARoutingEnv
from ..paths import find_existing, project_root
from ..representations.graph_builder import validate_graph
from ..training.config import TrainConfig
from ..training.evaluation import evaluate_model
from ..training.trainer import json_safe, make_vec

GRAPH_EXTRACTORS = {"gnn", "gnn_maxpool"}


@dataclass
class MethodRuns:
    label: str
    seeds: Dict[int, Path]  # seed -> seed_<s> run folder


def find_runs(label: str, run: str) -> MethodRuns:
    """The seed folders of a run (``runs/<name>`` or one ``runs/<name>/seed_<s>``)."""
    root = find_existing(run)
    folders = [root] if (root / "config.yaml").exists() and root.name.startswith("seed_") else \
        sorted(root.glob("seed_*"), key=lambda p: int(p.name.split("_", 1)[1]))
    folders = [f for f in folders if (f / "model.zip").exists()]
    if not folders:
        raise FileNotFoundError(f"no trained seed_* folders with model.zip under {root}")
    return MethodRuns(label, {int(f.name.split("_", 1)[1]): f for f in folders})


def _config(folder: Path) -> TrainConfig:
    with open(folder / "config.yaml", "r", encoding="utf-8") as fh:
        return TrainConfig.from_dict(yaml.safe_load(fh) or {})


def convergence_point(history: pd.DataFrame, threshold: float, window: int) -> Dict[str, float]:
    ok = (history["success_rate"] >= threshold).to_numpy()
    for i in range(len(ok) - window + 1):
        if ok[i:i + window].all():
            row = history.iloc[i]
            return {"converged_epoch": int(row["epoch"]), "env_steps_to_convergence": int(row["timesteps"]),
                    "optimizer_steps_to_convergence": int(row["optimizer_steps"])
                    if "optimizer_steps" in row and pd.notna(row["optimizer_steps"]) else np.nan}
    return {"converged_epoch": np.nan, "env_steps_to_convergence": np.nan, "optimizer_steps_to_convergence": np.nan}


def experiment_config_row(label: str, seed: int, cfg: TrainConfig, history: pd.DataFrame,
                          episodes: int, eval_seed: int) -> Dict[str, object]:
    graph = cfg.agent.extractor in GRAPH_EXTRACTORS
    # the extractor's actual arguments: its defaults, overridden by the config
    params = inspect.signature(get_extractor(cfg.agent.extractor).__init__).parameters
    kw = {k: v.default for k, v in params.items() if v.default is not inspect.Parameter.empty}
    kw.update(cfg.agent.extractor_kwargs)
    return {
        "method": label,
        "seed": seed,
        "encoder": cfg.agent.extractor,
        "graph_node_features": "health, droplet, goal" if graph else "",
        "graph_connectivity": "8-neighbour grid, both directions" if graph else "",
        "edge_attributes": "none" if graph else "",
        "gnn_type": kw.get("gnn_type", "") if graph else "",
        "gnn_hidden_dim": kw.get("hidden_dim", "") if graph else "",
        "gnn_num_layers": kw.get("num_layers", "") if graph else "",
        "pooling": kw.get("pooling", "") if graph else "flatten (CNN)",
        "cnn_channels": "" if graph else str(kw.get("channels", "")),
        "cnn_hidden_dim": "" if graph else kw.get("hidden_dim", ""),
        "chip": f"{cfg.env.width}x{cfg.env.height}",
        "obs_size": "native" if cfg.env.obs_size is None else "x".join(map(str, cfg.env.obs_size)),
        "fault_fraction": cfg.env.fault_fraction,
        "mark_collisions": cfg.env.mark_collisions,
        "ppo_n_envs": cfg.ppo.n_envs, "ppo_n_steps": cfg.ppo.n_steps, "ppo_batch_size": cfg.ppo.batch_size,
        "ppo_n_epochs": cfg.ppo.n_epochs, "ppo_gamma": cfg.ppo.gamma, "ppo_gae_lambda": cfg.ppo.gae_lambda,
        "ppo_ent_coef": cfg.ppo.ent_coef, "ppo_vf_coef": cfg.ppo.vf_coef, "ppo_clip_range": cfg.ppo.clip_range,
        "lr0": cfg.schedule.lr0, "epochs": int(len(history)), "steps_per_epoch": cfg.schedule.steps_per_epoch,
        "total_env_steps": int(history["timesteps"].iloc[-1]) if len(history) else 0,
        "evaluation_protocol": f"{episodes} held-out jobs, eval seed {eval_seed}, deterministic policy, "
                               f"timeouts counted at k_max",
    }


def inference_ms(model, env: MEDARoutingEnv, n: int = 200) -> float:
    obs, _ = env.reset(seed=0)
    model.predict(obs, deterministic=True)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(obs, deterministic=True)
    return 1e3 * (time.perf_counter() - t0) / n


def _mean_std(values: Sequence[float]) -> str:
    a = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if not len(a):
        return ""
    return f"{a.mean():.4g}" if len(a) == 1 else f"{a.mean():.4g} +- {a.std(ddof=1):.2g}"


def compare_methods(
    methods: List[MethodRuns],
    out_dir: Optional[Path] = None,
    episodes: int = 500,
    seed: int = 20_000,
    repeats: int = 1,
    checkpoint: str = "model.zip",
    device: str = "auto",
    n_envs: int = 8,
    converge_at: float = 0.95,
    converge_window: int = 3,
    log=print,
) -> Dict[str, Path]:
    """Evaluate every method/seed on the same jobs and write the tables, logs and figures."""
    from stable_baselines3 import PPO

    out = Path(out_dir) if out_dir else project_root() / "results"
    tables, logs, figures = out / "tables", out / "logs", out / "figures"
    for d in (tables, logs, figures):
        d.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)

    configs, training_rows, job_rows, per_seed = [], [], [], []
    histories: Dict[str, List[pd.DataFrame]] = {}
    env_ref = None
    for m in methods:
        for s, folder in m.seeds.items():
            cfg = _config(folder)
            env_cfg = {k: v for k, v in cfg.env.to_dict().items() if k != "obs_size"}
            if env_ref is None:
                env_ref = env_cfg
            elif env_cfg != env_ref:
                log(f"warning: {m.label} seed {s} was trained in a different environment than the first run")
            history = pd.read_csv(folder / "progress.csv")
            histories.setdefault(m.label, []).append(history)
            for _, row in history.iterrows():
                training_rows.append({"method": m.label, "seed": s, **row.to_dict()})
            model = PPO.load(folder / checkpoint, device=dev)
            results = []
            for r in range(repeats):
                vec = make_vec(cfg.env, n_envs, seed + 1000 * r)
                res = evaluate_model(model, vec, episodes, deterministic=True, return_jobs=True)
                vec.close()
                results.append(res)
                job_rows += [{"method": m.label, "seed": s, "repeat": r, **j} for j in res.pop("jobs")]
            ms = inference_ms(model, MEDARoutingEnv(cfg.env))
            conv = convergence_point(history, converge_at, converge_window)
            per_seed.append({
                "method": m.label, "seed": s,
                **{k: float(np.mean([res[k] for res in results])) for k in results[0] if k != "episodes"},
                **conv,
                "train_time_s": float(history["elapsed_seconds"].iloc[-1])
                if "elapsed_seconds" in history else float(history["epoch_seconds"].sum()),
                "inference_ms_per_decision": ms,
            })
            configs.append(experiment_config_row(m.label, s, cfg, history, episodes, seed))
            log(f"[{m.label} seed {s}] success {per_seed[-1]['success_rate']:.1%}  "
                f"cycles {per_seed[-1]['mean_cycles']:.2f}  inference {ms:.2f} ms ({dev})")

    seeds_df = pd.DataFrame(per_seed)
    main_rows = []
    for label, grp in seeds_df.groupby("method", sort=False):
        main_rows.append({
            "method": label,
            "n_seeds": len(grp),
            "success_rate_pct": _mean_std(100 * grp["success_rate"]),
            "mean_cycles": _mean_std(grp["mean_cycles"]),
            "median_cycles": _mean_std(grp["median_cycles"]),
            "std_cycles": _mean_std(grp["std_cycles"]),
            "mean_cycles_successful_jobs": _mean_std(grp["mean_cycles_success"]),
            "failure_rate_pct": _mean_std(100 * grp["failure_rate"]),
            "invalid_action_rate": _mean_std(grp["invalid_action_rate"]),
            "env_steps_to_convergence": _mean_std(grp["env_steps_to_convergence"]),
            "optimizer_steps_to_convergence": _mean_std(grp["optimizer_steps_to_convergence"]),
            "seeds_converged": int(grp["env_steps_to_convergence"].notna().sum()),
            "train_time_s": _mean_std(grp["train_time_s"]),
            "inference_ms_per_decision": _mean_std(grp["inference_ms_per_decision"]),
            "device": dev,
        })

    paths = {
        "experiment_config": tables / "experiment_config.csv",
        "main_comparison": tables / "main_comparison.csv",
        "per_seed": tables / "per_seed_results.csv",
        "training_metrics": logs / "training_metrics.csv",
        "evaluation_jobs": logs / "evaluation_jobs.csv",
        "graph_validation": tables / "graph_validation.csv",
    }
    pd.DataFrame(configs).to_csv(paths["experiment_config"], index=False)
    pd.DataFrame(main_rows).to_csv(paths["main_comparison"], index=False)
    seeds_df.to_csv(paths["per_seed"], index=False)
    pd.DataFrame(training_rows).to_csv(paths["training_metrics"], index=False)
    pd.DataFrame(job_rows).to_csv(paths["evaluation_jobs"], index=False)

    # graph checks on an observation of the (first) graph method's environment
    graph_cfg = next((_config(f) for m in methods for f in m.seeds.values()
                      if _config(f).agent.extractor in GRAPH_EXTRACTORS), None)
    env = MEDARoutingEnv((graph_cfg or _config(next(iter(methods[0].seeds.values())))).env)
    obs, _ = env.reset(seed=seed)
    pd.DataFrame(validate_graph(obs)).to_csv(paths["graph_validation"], index=False)

    from ..viz.plots import plot_method_curves

    for column, ylabel, name in (
        ("success_rate", "evaluation success rate (%)", "success_rate_vs_env_steps"),
        ("train_success_rate", "training success rate (%)", "train_success_rate_vs_env_steps"),
        ("mean_cycles", "mean routing cycles (timeouts at k_max)", "mean_cycles_vs_env_steps"),
    ):
        if all(column in h for hs in histories.values() for h in hs):
            paths[name] = plot_method_curves(histories, column, ylabel, figures / f"{name}.png",
                                             percent=column.endswith("success_rate"))
    job_df = pd.DataFrame(job_rows)
    if len(histories) >= 2 and len(job_df):
        from ..viz.plots import plot_per_job_cycles

        paths["per_job_cycles"] = plot_per_job_cycles(job_df, figures / "per_job_cycles.png")
    return paths
