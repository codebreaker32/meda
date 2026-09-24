"""Policy networks (state encoders for PPO).

* ``cnn`` - the paper's CNN of Table I (:class:`MedaCNN`), the baseline;
* ``gnn`` / ``gnn_maxpool`` - the proposed graph encoder with global max
  pooling (:class:`MedaGNN`, ``docs/GNN_METHODOLOGY.md``).
"""

from .cnn import MedaCNN
from .registry import available_extractors, get_extractor, policy_kwargs_for, register_extractor
from .gnn import MedaGNN  # noqa: E402  (registers "gnn" and "gnn_maxpool")

__all__ = [
    "MedaCNN",
    "MedaGNN",
    "available_extractors",
    "get_extractor",
    "policy_kwargs_for",
    "register_extractor",
]
