"""State representations of a MEDA observation for the policy networks.

* grid: the paper's 3-channel image (``envs/observation.py``), used by the CNN;
* graph: one node per microelectrode with 8-neighbour edges
  (:mod:`.graph_builder`), used by the GNN encoder (``agents/gnn.py``).
"""

from .graph_builder import (
    NEIGHBOUR_OFFSETS,
    batch_to_node_features,
    expected_edge_count,
    grid_edges,
    node_coords,
    node_index,
    observation_to_graph,
    validate_graph,
)

__all__ = [
    "NEIGHBOUR_OFFSETS",
    "batch_to_node_features",
    "expected_edge_count",
    "grid_edges",
    "node_coords",
    "node_index",
    "observation_to_graph",
    "validate_graph",
]
