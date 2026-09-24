"""GNN-PPO end to end: policy shapes, save/load, training logs, comparison artifacts,
and consistency with the frozen CNN baseline (docs/GNN_METHODOLOGY.md, sections 6-8, 12)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO

from meda_routing.agents import policy_kwargs_for
from meda_routing.envs import MEDARoutingEnv
from meda_routing.training.config import load_config
from meda_routing.training.trainer import PROGRESS_COLUMNS, make_vec, train

TINY = [
    "env.width=8",
    "env.height=8",
    "env.jobs.droplet_sizes=[[2,2]]",
    "ppo.n_envs=2",
    "ppo.n_steps=32",
    "schedule.epochs=2",
    "schedule.steps_per_epoch=64",
    "schedule.checkpoint_every=0",
    "eval.episodes=4",
    "eval.n_envs=2",
    "verbose=0",
    "save_plots=false",
]
GNN = ["agent.extractor=gnn", "agent.extractor_kwargs={hidden_dim: 8, num_layers: 2}", "env.obs_size=null"]
CNN = ["agent.extractor_kwargs={channels: [4], hidden_dim: 8}", "env.obs_size=[8,8]"]


def test_ppo_accepts_the_graph_embedding(tmp_path):
    env = make_vec(load_config(None, TINY + GNN).env, 2, 0)
    model = PPO("MlpPolicy", env, n_steps=16, batch_size=16, device="cpu",
                policy_kwargs=policy_kwargs_for("gnn", {"hidden_dim": 8, "num_layers": 2}))
    obs = env.reset()
    t = torch.as_tensor(obs)
    dist = model.policy.get_distribution(t)
    assert dist.distribution.logits.shape == (2, 8)  # the same 8 actions as the baseline
    assert torch.allclose(dist.distribution.probs.sum(-1), torch.ones(2))
    value = model.policy.predict_values(t)
    assert value.shape == (2, 1) and torch.isfinite(value).all()
    model.learn(32)
    actions, _ = model.predict(obs, deterministic=True)
    assert actions.shape == (2,) and set(actions) <= set(range(8))
    model.save(tmp_path / "gnn.zip")
    loaded = PPO.load(tmp_path / "gnn.zip", device="cpu")
    assert np.array_equal(loaded.predict(obs, deterministic=True)[0], actions)
    env.step(actions)  # environment stepping works with the policy's actions


def test_gnn_config_changes_only_the_encoder():
    base = dataclasses.asdict(load_config("configs/training/paper_30x30_healthy.yaml"))
    gnn = dataclasses.asdict(load_config("configs/training/gnn_maxpool_30x30.yaml"))
    assert gnn["agent"]["extractor"] == "gnn" and gnn["agent"]["extractor_kwargs"]["pooling"] == "max"
    assert "channels" not in gnn["agent"]["extractor_kwargs"]  # CNN arguments are not inherited
    for section in ("ppo", "schedule", "eval"):
        assert gnn[section] == base[section], section
    env_b, env_g = dict(base["env"]), dict(gnn["env"])
    assert env_g.pop("obs_size") is None and env_b.pop("obs_size") == (30, 30)
    assert env_b == env_g  # same simulator, reward, actions and jobs
    small = dataclasses.asdict(load_config("configs/training/gnn_maxpool_16x16.yaml"))
    ref = dataclasses.asdict(load_config("configs/training/validation_16x16.yaml"))
    assert small["ppo"] == ref["ppo"] and small["schedule"] == ref["schedule"]


def test_native_graph_observation_equals_the_baseline_image_on_30x30():
    cnn = MEDARoutingEnv(load_config("configs/training/paper_30x30_healthy.yaml").env)
    gnn = MEDARoutingEnv(load_config("configs/training/gnn_maxpool_30x30.yaml").env)
    for seed in range(3):
        a, _ = cnn.reset(seed=seed)
        b, _ = gnn.reset(seed=seed)
        assert np.array_equal(a, b)


def test_training_logs_the_comparison_metrics(tmp_path):
    (run,) = train(load_config(None, TINY + GNN + ["name=g"]), tmp_path)
    progress = pd.read_csv(run / "progress.csv")
    assert list(progress.columns) == PROGRESS_COLUMNS and len(progress) == 2
    last = progress.iloc[-1]
    assert last["timesteps"] == 128 and last["ppo_updates"] == 2
    assert last["optimizer_steps"] == 2 * 4 * 2  # 2 rollouts x 4 PPO epochs x (64 samples / 32 per minibatch)
    assert 0 <= last["failure_rate"] <= 1 and last["elapsed_seconds"] >= last["epoch_seconds"]


def test_compare_methods_writes_all_artifacts(tmp_path):
    from meda_routing.experiments.method_comparison import compare_methods, find_runs

    runs = tmp_path / "runs"
    train(load_config(None, TINY + CNN + ["name=cnn", "repeats=2"]), runs)
    train(load_config(None, TINY + GNN + ["name=gnn"]), runs)
    out = tmp_path / "results"
    paths = compare_methods([find_runs("CNN-PPO", str(runs / "cnn")), find_runs("GNN-max-PPO", str(runs / "gnn"))],
                            out, episodes=6, n_envs=2, log=lambda *_: None)
    for name in ("experiment_config", "main_comparison", "training_metrics", "evaluation_jobs",
                 "graph_validation", "success_rate_vs_env_steps", "mean_cycles_vs_env_steps"):
        assert paths[name].exists(), name
    main = pd.read_csv(paths["main_comparison"])
    assert list(main["method"]) == ["CNN-PPO", "GNN-max-PPO"] and list(main["n_seeds"]) == [2, 1]
    jobs = pd.read_csv(paths["evaluation_jobs"])
    ids = jobs.groupby(["method", "seed"])["job_id"].apply(tuple)
    assert len(set(ids)) == 1  # every method and seed saw the same held-out jobs
    config = pd.read_csv(paths["experiment_config"])
    assert set(config["pooling"]) == {"flatten (CNN)", "max"}
    assert pd.read_csv(paths["graph_validation"])["passed"].all()
    assert set(pd.read_csv(paths["training_metrics"])["method"]) == {"CNN-PPO", "GNN-max-PPO"}


def test_validate_graph_command(tmp_path):
    from meda_routing.cli import main

    out = tmp_path / "gv.csv"
    main(["validate-graph", "-c", "configs/training/gnn_maxpool_16x16.yaml", "--out", str(out)])
    table = pd.read_csv(out)
    assert table["passed"].all() and (table["check"] == "node count = W x H").any()
