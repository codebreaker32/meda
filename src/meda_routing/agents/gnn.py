"""GNN state encoder for PPO: graph message passing + graph readout.

Replaces the CNN feature extractor of the paper (Table I) while keeping
everything else fixed (environment, rewards, action space, PPO settings);
see ``docs/GNN_METHODOLOGY.md``.  Pipeline::

    observation (3, H, W)
      -> graph: one node per MC, 3 features, 8-neighbour edges  (representations/graph_builder.py)
      -> num_layers x message passing                            -> node embeddings Z (B, N, d)
      -> global max pooling                                      -> graph embedding g (B, d)
      -> PPO actor (8 action logits) and critic (V(s)), linear heads as for the CNN

Message-passing layers (``gnn_type``):

* ``gcn`` (default) [METHOD-DOC]: graph convolution of Kipf & Welling,
  ``H' = ReLU(D^-1/2 (A + I) D^-1/2 H W + b)`` on the 8-neighbour graph.
  It is *isotropic*: every neighbour is treated the same, whatever its
  direction.  With max pooling this makes the graph embedding invariant to
  mirroring the job, so the policy cannot tell "goal to the east" from
  "goal to the west" except through the chip boundary
  (``tests/test_graph_readout.py`` demonstrates it).  This is the first
  configuration to evaluate, as the methodology asks.
* ``dir_gcn`` [ASSUMED, ablation]: relational message passing in which each
  of the 8 neighbour directions has its own weight matrix (R-GCN with the
  direction as the relation type), ``H'_i = ReLU(W_0 H_i + sum_r W_r H_{j_r(i)} + b)``.
  The direction comes from the graph's own geometry, not from stored edge
  attributes.  Use it if ``gcn`` cannot learn direction.

All tensors (the edge list included) are module buffers, so the encoder runs
on whatever device PPO is placed on (``devices.py``: a GPU when available).
"""

from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from ..representations.graph_builder import NEIGHBOUR_OFFSETS, NUM_NODE_FEATURES, grid_edges
from .graph_readout import graph_readout, readout_dim
from .registry import register_extractor

GNN_TYPES = ("gcn", "dir_gcn")


def _edge_relations(width: int, edges: np.ndarray) -> np.ndarray:
    """Index into NEIGHBOUR_OFFSETS of each edge's direction (source -> target)."""
    dx = edges[1] % width - edges[0] % width
    dy = edges[1] // width - edges[0] // width
    lookup = {off: r for r, off in enumerate(NEIGHBOUR_OFFSETS)}
    return np.array([lookup[(int(a), int(b))] for a, b in zip(dx, dy)], dtype=np.int64)


class GCNLayer(nn.Module):
    """``out = A_hat (h W) + b`` with ``A_hat = D^-1/2 (A + I) D^-1/2``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, h: torch.Tensor, propagate) -> torch.Tensor:
        return propagate(self.linear(h), None) + self.bias


class DirectionalLayer(nn.Module):
    """Relational layer: own weights for the node itself and for each of the 8 directions."""

    def __init__(self, in_dim: int, out_dim: int, n_relations: int = len(NEIGHBOUR_OFFSETS)) -> None:
        super().__init__()
        self.n_relations = n_relations
        self.self_linear = nn.Linear(in_dim, out_dim)
        # W_r for all directions at once: (in) -> (R * out)
        self.rel_linear = nn.Linear(in_dim, n_relations * out_dim, bias=False)

    def forward(self, h: torch.Tensor, propagate) -> torch.Tensor:
        b, n, _ = h.shape
        messages = self.rel_linear(h).reshape(b, n * self.n_relations, -1)  # row j*R + r = W_r h_j
        return self.self_linear(h) + propagate(messages, "relational")


class GraphEncoder(nn.Module):
    """``num_layers`` message-passing layers with ReLU: ``(B, N, 3) -> (B, N, d)``.

    Messages flow along the explicit edge list of the 8-neighbour graph.  The
    edge list is turned into sparse adjacency matrices once per device, and
    aggregation is a sparse-dense product (about 8x faster than scattering
    edge by edge, same result); on Apple MPS, which lacks sparse support, the
    edge-by-edge scatter is used.
    """

    def __init__(self, width: int, height: int, gnn_type: str = "gcn",
                 hidden_dim: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        if gnn_type not in GNN_TYPES:
            raise ValueError(f"unknown gnn_type {gnn_type!r}; choose from {GNN_TYPES}")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_nodes = n = width * height
        self.gnn_type = gnn_type
        edges = np.array(grid_edges(width, height))
        if gnn_type == "gcn":  # weights D^-1/2 (A + I) D^-1/2, self-loops included
            loops = np.arange(n, dtype=np.int64)
            src = np.concatenate([edges[0], loops])
            dst = np.concatenate([edges[1], loops])
            deg = np.bincount(dst, minlength=n).astype(np.float64)
            weight = 1.0 / np.sqrt(deg[src] * deg[dst])
            rel = np.full(len(src), -1, dtype=np.int64)
        else:  # one relation per direction; the node itself has its own weights
            src, dst = edges
            weight = np.ones(edges.shape[1])
            rel = _edge_relations(width, edges)
        # derived from the grid size, so not saved in checkpoints
        self.register_buffer("src", torch.tensor(src), persistent=False)
        self.register_buffer("dst", torch.tensor(dst), persistent=False)
        self.register_buffer("weight", torch.tensor(weight, dtype=torch.float32), persistent=False)
        self.register_buffer("rel", torch.tensor(rel), persistent=False)
        self._sparse = {}
        layer = GCNLayer if gnn_type == "gcn" else DirectionalLayer
        dims = [NUM_NODE_FEATURES] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(layer(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def _columns(self) -> torch.Tensor:
        """Column of each edge's message: node j, or row j*R + r for relational messages."""
        if self.gnn_type == "gcn":
            return self.src
        return self.src * len(NEIGHBOUR_OFFSETS) + self.rel

    def _adjacency(self, n_cols: int) -> torch.Tensor:
        key = (self.src.device, n_cols)
        if key not in self._sparse:
            index = torch.stack([self.dst, self._columns()])  # row = receiving node
            self._sparse[key] = torch.sparse_coo_tensor(
                index, self.weight, (self.num_nodes, n_cols), check_invariants=True).coalesce()
        return self._sparse[key]

    def propagate(self, h: torch.Tensor, kind=None) -> torch.Tensor:
        """Sum of the weighted messages each node receives along its edges.

        ``h`` is ``(B, N, d)`` (one message per node, GCN) or ``(B, N*R, d)``
        (one message per node and direction, relational layers).
        """
        b, m, d = h.shape
        if h.device.type == "mps":
            msgs = h[:, self._columns()] * self.weight.view(1, -1, 1)
            out = torch.zeros(b, self.num_nodes, d, dtype=h.dtype, device=h.device)
            return out.index_add_(1, self.dst, msgs)
        x = h.transpose(0, 1).reshape(m, b * d)
        return torch.sparse.mm(self._adjacency(m), x).reshape(self.num_nodes, b, d).transpose(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h, self.propagate))
        return h


@register_extractor("gnn")
class MedaGNN(BaseFeaturesExtractor):
    """PPO feature extractor: graph encoder + readout on the MEDA observation.

    Args (all recorded in the run's ``config.yaml`` under ``agent.extractor_kwargs``):
        gnn_type: ``"gcn"`` (default) or ``"dir_gcn"``.  [METHOD-DOC] GCN / [ASSUMED] ablation
        hidden_dim: node-embedding size ``d``.  [ASSUMED] 64
        num_layers: message-passing layers (receptive field in hops).  [ASSUMED] 3
        pooling: ``"max"`` (default), ``"mean"``, ``"sum"`` or ``"role"``.  [METHOD-DOC] max
    """

    def __init__(self, observation_space: gym.spaces.Box, gnn_type: str = "gcn",
                 hidden_dim: int = 64, num_layers: int = 3, pooling: str = "max") -> None:
        channels, height, width = observation_space.shape
        if channels != NUM_NODE_FEATURES:
            raise ValueError(f"expected {NUM_NODE_FEATURES} observation channels, got {channels}")
        super().__init__(observation_space, features_dim=readout_dim(pooling, hidden_dim))
        self.pooling = pooling
        self.encoder = GraphEncoder(width, height, gnn_type, hidden_dim, num_layers)

    def node_embeddings(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(X, Z)``: node features ``(B, N, 3)`` and embeddings ``(B, N, d)``."""
        b, c, h, w = obs.shape
        x = obs.reshape(b, c, h * w).transpose(1, 2)  # node(x, y) = y*W + x
        return x, self.encoder(x)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x, z = self.node_embeddings(obs)
        return graph_readout(z, self.pooling, x)


# The methodology's first configuration by name: GCN + global max pooling.
register_extractor("gnn_maxpool")(MedaGNN)
