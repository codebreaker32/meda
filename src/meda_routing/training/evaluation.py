"""Per-epoch evaluation (Sec. V-B).

"The metrics are collected after each training epoch by testing the agent for
500 random routing jobs": mean score (episode return), success rate and the
average number of cycles.  Episodes are split evenly across the vectorized
environments so that short episodes are not over-represented.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv


def evaluate_model(
    model: BaseAlgorithm,
    vec_env: VecEnv,
    n_episodes: int = 500,
    deterministic: bool = True,
    return_jobs: bool = False,
) -> Dict[str, Any]:
    """Run ``n_episodes`` evaluation episodes and summarize them.

    Cycle statistics: ``mean_cycles`` / ``median_cycles`` / ``std_cycles``
    count every episode, a timed-out one at ``k_max`` cycles (as the paper's
    reference code does); ``mean_cycles_success`` averages successful
    episodes only (NaN when none succeeds; never reported as 0).
    ``invalid_action_rate`` is invalid actions per decision.  With
    ``return_jobs`` the result also holds ``jobs``: one record per episode
    (job index, success, cycles, termination reason, invalid actions),
    episodes numbered in the fixed order of the seeded evaluation envs.
    """
    n_envs = vec_env.num_envs
    targets = np.array([(n_episodes + i) // n_envs for i in range(n_envs)], dtype=int)
    counts = np.zeros(n_envs, dtype=int)
    returns = np.zeros(n_envs)
    invalid = np.zeros(n_envs, dtype=int)
    jobs: List[Dict[str, Any]] = []
    obs = vec_env.reset()
    while (counts < targets).any():
        actions, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(actions)
        returns += rewards
        for i in range(n_envs):
            invalid[i] += bool(infos[i].get("invalid_action", False))
            if dones[i]:
                if counts[i] < targets[i]:
                    success = bool(infos[i]["is_success"])
                    jobs.append({
                        # env i evaluates jobs i, i + n_envs, ... of the fixed sequence
                        "job_id": int(i + n_envs * counts[i]),
                        "success": success,
                        "cycles": int(infos[i]["num_cycles"]),
                        "termination": "goal" if success else "timeout",
                        "invalid_actions": int(invalid[i]),
                        "score": float(returns[i]),
                    })
                    counts[i] += 1
                returns[i] = 0.0
                invalid[i] = 0
    jobs.sort(key=lambda j: j["job_id"])
    scores_a = np.array([j["score"] for j in jobs])
    cycles_a = np.array([j["cycles"] for j in jobs], dtype=float)
    success_a = np.array([j["success"] for j in jobs])
    decisions = cycles_a.sum()
    out: Dict[str, Any] = {
        "episodes": int(len(jobs)),
        "mean_score": float(scores_a.mean()),
        "std_score": float(scores_a.std()),
        "success_rate": float(success_a.mean()),
        "failure_rate": float(1.0 - success_a.mean()),
        "mean_cycles": float(cycles_a.mean()),
        "median_cycles": float(np.median(cycles_a)),
        "std_cycles": float(cycles_a.std()),
        "mean_cycles_success": float(cycles_a[success_a].mean()) if success_a.any() else float("nan"),
        "median_cycles_success": float(np.median(cycles_a[success_a])) if success_a.any() else float("nan"),
        "invalid_action_rate": float(sum(j["invalid_actions"] for j in jobs) / decisions) if decisions else float("nan"),
    }
    if return_jobs:
        out["jobs"] = jobs
    return out
