"""Graph readouts and the GNN encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from meda_routing.agents import MedaGNN, get_extractor
from meda_routing.agents.gnn import GraphEncoder
from meda_routing.agents.graph_readout import graph_readout, readout_dim
from meda_routing.envs import MEDARoutingEnv


def test_global_max_pooling_is_the_elementwise_maximum():
    z = torch.randn(3, 10, 4)
    g = graph_readout(z, "max")
    assert g.shape == (3, 4) and torch.equal(g, z.max(dim=1).values)
    perm = torch.randperm(10)
    assert torch.equal(graph_readout(z[:, perm], "max"), g)  # node order does not matter


def test_role_readout_and_sizes():
    z = torch.randn(2, 6, 4)
    x = torch.zeros(2, 6, 3)
    x[:, 0, 1] = 1  # droplet node
    x[:, [4, 5], 2] = 1  # goal nodes
    g = graph_readout(z, "role", x)
    assert g.shape == (2, 12) and readout_dim("role", 4) == 12
    assert torch.allclose(g[:, 4:8], z[:, 0]) and torch.allclose(g[:, 8:], z[:, 4:6].mean(1))
    with pytest.raises(ValueError):
        graph_readout(z, "attention")


def test_sparse_propagation_equals_edge_by_edge_scatter():
    for gnn_type in ("gcn", "dir_gcn"):
        enc = GraphEncoder(7, 5, gnn_type, hidden_dim=8, num_layers=1)
        m = enc.num_nodes * (1 if gnn_type == "gcn" else 8)
        h = torch.randn(3, m, 4)
        ref = torch.zeros(3, enc.num_nodes, 4).index_add_(
            1, enc.dst, h[:, enc._columns()] * enc.weight.view(1, -1, 1))
        assert torch.allclose(enc.propagate(h), ref, atol=1e-6)


def test_gcn_weights_are_symmetric_normalized():
    enc = GraphEncoder(4, 3, "gcn", hidden_dim=2, num_layers=1)
    a = torch.zeros(12, 12)
    a[enc.dst, enc.src] = enc.weight
    assert torch.allclose(a, a.T)  # D^-1/2 (A + I) D^-1/2 is symmetric
    deg = (a > 0).sum(1).float()
    assert torch.allclose(a.diagonal(), 1 / deg)  # self-loop weight 1 / deg


def _mirrored_pair(size: int = 15):
    """A job and its left-right mirror image on a chip that is symmetric about its centre."""
    env = MEDARoutingEnv({"width": size, "height": size, "obs_size": None,
                          "jobs": {"droplet_sizes": [[2, 2]], "hazard_margin": None}})
    env.reset(seed=0)
    obs = np.zeros((3, size, size), dtype=np.float32)
    obs[0] = 0.75  # uniform health everywhere
    obs[1, 6:8, 2:4] = 1  # droplet on the west side
    obs[2, 6:8, 11:13] = 1  # goal on the east side
    return env.observation_space, torch.as_tensor(np.stack([obs, obs[:, :, ::-1].copy()]))


def test_isotropic_gcn_with_max_pooling_cannot_tell_a_job_from_its_mirror_image():
    """Documents the limitation discussed in agents/gnn.py: same embedding, so same policy."""
    space, pair = _mirrored_pair()
    torch.manual_seed(0)
    g = MedaGNN(space, gnn_type="gcn", pooling="max")(pair)
    assert torch.allclose(g[0], g[1], atol=1e-5)


def test_directional_layers_distinguish_the_mirror_image():
    space, pair = _mirrored_pair()
    torch.manual_seed(0)
    g = MedaGNN(space, gnn_type="dir_gcn", pooling="max")(pair)
    assert not torch.allclose(g[0], g[1], atol=1e-3)


@pytest.mark.parametrize("kwargs,dim", [({}, 64), ({"pooling": "role", "hidden_dim": 16}, 48),
                                         ({"gnn_type": "dir_gcn", "num_layers": 2}, 64)])
def test_extractor_output_shapes(kwargs, dim):
    env = MEDARoutingEnv({"width": 9, "height": 6, "obs_size": None})
    obs = torch.as_tensor(np.stack([env.reset(seed=s)[0] for s in range(5)]))
    ext = get_extractor("gnn")(env.observation_space, **kwargs)
    out = ext(obs)
    assert out.shape == (5, dim) and ext.features_dim == dim and torch.isfinite(out).all()
    assert get_extractor("gnn_maxpool") is MedaGNN
    x, z = ext.node_embeddings(obs)
    assert x.shape == (5, 54, 3) and z.shape[:2] == (5, 54)


def test_bad_arguments():
    env = MEDARoutingEnv({"width": 9, "height": 9, "obs_size": None})
    with pytest.raises(ValueError):
        MedaGNN(env.observation_space, gnn_type="gat")
    with pytest.raises(ValueError):
        MedaGNN(env.observation_space, num_layers=0)
