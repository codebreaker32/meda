"""Graph readouts: node embeddings ``Z (B, N, d)`` -> one graph embedding per sample.

* ``max`` (default) [METHOD-DOC]: global max pooling, ``g[k] = max_i Z[i, k]``,
  giving ``g`` of size ``d``.  It keeps the strongest activation per
  dimension and ignores node order.  It may lose *where* the droplet and
  goal are, which the methodology treats as a hypothesis to test.
* ``mean`` / ``sum`` [ASSUMED]: alternatives for readout ablations.
* ``role`` [METHOD-DOC, "role-aware readout" ablation]: concatenates the
  global max pool, the mean over the droplet's nodes and the mean over the
  goal's nodes (roles read from node features 1 and 2), giving ``3d``.
"""

from __future__ import annotations

import torch

#: Readouts and the size of the graph embedding they produce, as a multiple of d.
READOUTS = {"max": 1, "mean": 1, "sum": 1, "role": 3}


def readout_dim(pooling: str, hidden_dim: int) -> int:
    if pooling not in READOUTS:
        raise ValueError(f"unknown pooling {pooling!r}; choose from {sorted(READOUTS)}")
    return READOUTS[pooling] * hidden_dim


def _masked_mean(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1)
    return (z * weights).sum(1) / weights.sum(1).clamp_min(1e-6)


def graph_readout(z: torch.Tensor, pooling: str = "max", node_features: torch.Tensor = None) -> torch.Tensor:
    """Graph embedding ``(B, readout_dim)`` from node embeddings ``z (B, N, d)``.

    ``node_features (B, N, 3)`` is needed only for ``role`` (droplet = feature
    1 > 0, goal = feature 2 > 0).
    """
    if pooling == "max":
        return z.max(dim=1).values
    if pooling == "mean":
        return z.mean(dim=1)
    if pooling == "sum":
        return z.sum(dim=1)
    if pooling == "role":
        if node_features is None:
            raise ValueError("role readout needs the node features")
        droplet = (node_features[..., 1] > 0).to(z.dtype)
        goal = (node_features[..., 2] > 0).to(z.dtype)
        return torch.cat([z.max(dim=1).values, _masked_mean(z, droplet), _masked_mean(z, goal)], dim=-1)
    raise ValueError(f"unknown pooling {pooling!r}; choose from {sorted(READOUTS)}")
