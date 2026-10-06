"""The two theory claims of docs/PAPER_NOTES.md (F1 and F3), checked in float64.

F1: every symmetry of the chip (the 8 elements of D4 on a square chip; the mirrors and
the 180-degree rotation on a rectangular one) maps the 8-neighbour grid onto itself, and
the isotropic GCN weights all neighbours alike, so the GCN's node embeddings are permuted
along with the chip and any permutation-invariant readout gives the same graph embedding.
The policy, a linear head on that embedding, then cannot tell a job from its mirror image
or rotation.

F3: a direction-aware layer, h'_i = W_0 h_i + b + sum_r W_r h_(i - o_r), is a 3x3 stride-1
convolution with zero padding: W_0 is the centre tap and W_r the tap at (row 1 - dy,
column 1 - dx) for the offset o_r = (dx, dy); a neighbour missing at the chip edge
contributes nothing, as zero padding does. The direction-aware GCN with global max
pooling is therefore a 3-layer 3x3 CNN with global max pooling.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from meda_routing.agents import MedaGNN
from meda_routing.agents.gnn import GraphEncoder
from meda_routing.representations.graph_builder import NEIGHBOUR_OFFSETS

# Maps of a (..., H, W) observation; the first three exist on any rectangular chip.
CHIP_SYMMETRIES = {
    "mirror left-right": lambda x: x.flip(-1),
    "mirror up-down": lambda x: x.flip(-2),
    "rotate 180": lambda x: torch.rot90(x, 2, (-2, -1)),
    "rotate 90": lambda x: torch.rot90(x, 1, (-2, -1)),
    "rotate 270": lambda x: torch.rot90(x, 3, (-2, -1)),
    "transpose": lambda x: x.transpose(-2, -1),
    "anti-transpose": lambda x: torch.rot90(x, 2, (-2, -1)).transpose(-2, -1),
}
CHIPS = [(9, 9), (11, 7)]  # (width, height): square (all of D4) and rectangular (first three)


def _symmetries(width: int, height: int):
    names = list(CHIP_SYMMETRIES) if width == height else list(CHIP_SYMMETRIES)[:3]
    return [(n, CHIP_SYMMETRIES[n]) for n in names]


def _extractor(width: int, height: int, gnn_type: str, pooling: str = "max") -> MedaGNN:
    """Random weights and nonzero biases, in float64."""
    torch.manual_seed(0)
    space = gym.spaces.Box(0.0, 1.0, (3, height, width), np.float32)
    ext = MedaGNN(space, gnn_type=gnn_type, pooling=pooling).double()
    with torch.no_grad():
        for p in ext.parameters():
            p.normal_(0.0, 0.5)
    return ext


def _observations(width: int, height: int) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.rand(4, 3, height, width, dtype=torch.float64)


@pytest.mark.parametrize("pooling", ["max", "mean", "sum"])
@pytest.mark.parametrize("width,height", CHIPS)
def test_isotropic_gcn_embedding_is_invariant_to_every_chip_symmetry(width, height, pooling):
    ext = _extractor(width, height, "gcn", pooling)
    obs = _observations(width, height)
    g = ext(obs)
    for name, f in _symmetries(width, height):
        assert torch.allclose(ext(f(obs)), g, rtol=1e-10, atol=1e-12), name


@pytest.mark.parametrize("width,height", CHIPS)
def test_direction_aware_gcn_embedding_changes_under_every_chip_symmetry(width, height):
    """Control for the test above: the maps do change the input the encoder sees."""
    ext = _extractor(width, height, "dir_gcn")
    obs = _observations(width, height)
    g = ext(obs)
    for name, f in _symmetries(width, height):
        assert not torch.allclose(ext(f(obs)), g, rtol=1e-3, atol=1e-3), name


def _convolutional_twin(encoder: GraphEncoder) -> nn.Sequential:
    """The 3x3 zero-padded CNN with the weights of a direction-aware encoder."""
    layers = []
    for layer in encoder.layers:
        out_dim, in_dim = layer.self_linear.weight.shape
        conv = nn.Conv2d(in_dim, out_dim, 3, padding=1).to(layer.self_linear.weight.dtype)
        w_rel = layer.rel_linear.weight.view(len(NEIGHBOUR_OFFSETS), out_dim, in_dim)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, :, 1, 1] = layer.self_linear.weight
            conv.bias.copy_(layer.self_linear.bias)
            for r, (dx, dy) in enumerate(NEIGHBOUR_OFFSETS):
                conv.weight[:, :, 1 - dy, 1 - dx] = w_rel[r]
        layers += [conv, nn.ReLU()]
    return nn.Sequential(*layers)


@pytest.mark.parametrize("width,height", CHIPS + [(16, 16)])
def test_direction_aware_gcn_is_a_zero_padded_3x3_cnn_with_global_max_pooling(width, height):
    ext = _extractor(width, height, "dir_gcn")
    cnn = _convolutional_twin(ext.encoder)
    obs = _observations(width, height)
    _, z = ext.node_embeddings(obs)  # (B, N, d), node (x, y) = y*W + x
    maps = cnn(obs)  # (B, d, H, W)
    assert torch.allclose(z, maps.flatten(2).transpose(1, 2), rtol=0, atol=1e-12)
    assert torch.allclose(ext(obs), maps.amax(dim=(-2, -1)), rtol=0, atol=1e-12)


def test_direction_aware_gcn_has_the_parameters_of_the_3x3_cnn():
    ext = _extractor(9, 9, "dir_gcn")
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    # Conv2d(3, 64, 3) + 2 x Conv2d(64, 64, 3): 1,792 + 36,928 + 36,928
    assert count(ext) == count(_convolutional_twin(ext.encoder)) == 75_648
