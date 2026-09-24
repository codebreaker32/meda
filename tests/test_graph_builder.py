"""Graph representation of observations (docs/GNN_METHODOLOGY.md, sections 4 and 12)."""

from __future__ import annotations

import numpy as np
import pytest

from meda_routing.envs import MEDARoutingEnv
from meda_routing.representations import (
    NEIGHBOUR_OFFSETS,
    batch_to_node_features,
    expected_edge_count,
    grid_edges,
    node_coords,
    node_index,
    observation_to_graph,
    validate_graph,
)


@pytest.mark.parametrize("width,height", [(1, 1), (1, 5), (2, 2), (3, 4), (16, 16), (60, 30)])
def test_edges_are_exactly_the_8_neighbours(width, height):
    edges = grid_edges(width, height)
    assert edges.shape == (2, expected_edge_count(width, height))
    got = set(zip(edges[0].tolist(), edges[1].tolist()))
    want = set()
    for y in range(height):
        for x in range(width):
            for dx, dy in NEIGHBOUR_OFFSETS:
                if 0 <= x + dx < width and 0 <= y + dy < height:
                    want.add((node_index(x, y, width), node_index(x + dx, y + dy, width)))
    assert got == want and len(got) == edges.shape[1]  # no duplicates, both directions
    assert all((j, i) in got for i, j in got)


def test_eight_directions_match_the_action_space():
    from meda_routing.core.actions import DIRECTIONS

    assert set(NEIGHBOUR_OFFSETS) == set(DIRECTIONS.values()) and len(NEIGHBOUR_OFFSETS) == 8


def test_degrees():
    edges = grid_edges(5, 4)
    degree = np.bincount(edges[0], minlength=20)
    assert degree[node_index(0, 0, 5)] == 3 and degree[node_index(2, 0, 5)] == 5
    assert degree[node_index(2, 2, 5)] == 8


def test_node_index_round_trip():
    for x, y in [(0, 0), (4, 0), (0, 3), (4, 3), (2, 1)]:
        assert node_coords(node_index(x, y, 5), 5) == (x, y)


@pytest.mark.parametrize("width,height", [(16, 16), (30, 30), (60, 30), (7, 11)])
def test_node_features_are_the_baseline_observation(width, height):
    env = MEDARoutingEnv({"width": width, "height": height, "obs_size": None,
                          "jobs": {"droplet_sizes": [[2, 2]]}, "fault_fraction": 0.1})
    obs, _ = env.reset(seed=3)
    x, edges = observation_to_graph(obs)
    assert x.shape == (width * height, 3) and x.dtype == obs.dtype
    for cy in range(height):
        for cx in range(width):
            assert np.array_equal(x[node_index(cx, cy, width)], obs[:, cy, cx])
    # droplet and goal nodes are exactly the MCs the env reports
    d = env.droplet
    droplet_nodes = {node_index(cx, cy, width) for cx in range(d.xa, d.xb + 1) for cy in range(d.ya, d.yb + 1)}
    assert set(np.flatnonzero(x[:, 1] > 0)) == droplet_nodes
    rows = validate_graph(obs)
    assert all(r["passed"] for r in rows), [r for r in rows if not r["passed"]]


def test_batched_node_features_numpy_and_torch():
    import torch

    obs = np.random.default_rng(0).random((4, 3, 5, 6)).astype(np.float32)
    xs = batch_to_node_features(obs)
    xt = batch_to_node_features(torch.as_tensor(obs))
    assert xs.shape == (4, 30, 3) and tuple(xt.shape) == (4, 30, 3)
    assert np.array_equal(xs[1], observation_to_graph(obs[1])[0]) and np.allclose(xt.numpy(), xs)


def test_rejects_non_observations():
    with pytest.raises(ValueError):
        observation_to_graph(np.zeros((2, 4, 4)))
