"""Graph representation of a MEDA observation: one node per microelectrode (MC).

``G = (V, E, X)`` for a chip observed as a ``W x H`` grid
(docs/GNN_METHODOLOGY.md, section 4):

* **V**: one node per MC, ``N = W * H`` nodes.  The mapping from grid
  coordinate to node index is fixed and row-major::

      node(x, y) = y * W + x          x = column (east), y = row (north)

  which is exactly the order of ``obs.reshape(3, -1)`` for the environment's
  channels-first observation ``(3, rows = y, cols = x)``.
* **E**: spatial adjacency between MCs in the **8 directions** (left, right,
  up, down and the 4 diagonals), the same 8 directions as the action space
  (N, S, E, W, NE, NW, SE, SW).  The methodology document first asked for 4
  directions; the supervisor's correction changed it to 8 for consistency
  with the actions.  Edges are stored in both directions (``i -> j`` and
  ``j -> i``) so that messages flow both ways, and carry **no attributes**.
  No self-loops are stored; a GCN layer adds its own.
* **X**: ``N x 3`` node features, the same three values as the baseline's
  observation (masked health, droplet, goal) with the same normalization;
  nothing is recomputed or rescaled here.

Degraded MCs stay in the graph as nodes whose health feature is low; the
stochastic movement model stays in the simulator.

Source tags: the graph representation is [METHOD-DOC] (the project's
methodology document, not the paper); the paper only defines the three
observation features [PAPER Fig. 2].
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

#: Neighbour offsets ``(dx, dy)``: 4 cardinal + 4 diagonal directions.
#: [METHOD-DOC, corrected to 8] same directions as the 8 actions [PAPER Sec. III-B].
NEIGHBOUR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (1, 0), (-1, 0), (0, 1), (0, -1),     # E, W, N, S
    (1, 1), (-1, 1), (1, -1), (-1, -1),   # NE, NW, SE, SW
)
#: Node features per MC. [PAPER Fig. 2] health, droplet, goal.
NUM_NODE_FEATURES = 3


def node_index(x: int, y: int, width: int) -> int:
    """Node of the MC in column ``x``, row ``y`` (row-major)."""
    return y * width + x


def node_coords(index: int, width: int) -> Tuple[int, int]:
    """Inverse of :func:`node_index`: ``(x, y)`` of a node."""
    return index % width, index // width


@lru_cache(maxsize=32)
def grid_edges(width: int, height: int) -> np.ndarray:
    """Directed edge list ``(2, E)`` of the 8-neighbour grid, both directions.

    ``E = 2 * [(W-1)H + W(H-1) + 2(W-1)(H-1)]``.  Cached per grid size; the
    returned array is read-only.
    """
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = xs.ravel(), ys.ravel()
    src: List[np.ndarray] = []
    dst: List[np.ndarray] = []
    for dx, dy in NEIGHBOUR_OFFSETS:
        nx, ny = xs + dx, ys + dy
        ok = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
        src.append(ys[ok] * width + xs[ok])
        dst.append(ny[ok] * width + nx[ok])
    edges = np.stack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64)
    edges.setflags(write=False)
    return edges


def expected_edge_count(width: int, height: int) -> int:
    """Number of directed edges of the 8-neighbour ``W x H`` grid."""
    undirected = (width - 1) * height + width * (height - 1) + 2 * (width - 1) * (height - 1)
    return 2 * undirected


def observation_to_graph(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(X, edge_index)`` of one observation ``(3, H, W)``.

    ``X`` is ``(N, 3)`` with row ``node(x, y)`` holding ``obs[:, y, x]``.
    """
    obs = np.asarray(obs)
    if obs.ndim != 3 or obs.shape[0] != NUM_NODE_FEATURES:
        raise ValueError(f"expected an observation of shape (3, H, W), got {obs.shape}")
    _, height, width = obs.shape
    features = obs.reshape(NUM_NODE_FEATURES, height * width).T.copy()
    return features, grid_edges(width, height)


def batch_to_node_features(obs):
    """``(B, 3, H, W)`` tensor or array -> ``(B, N, 3)`` node features (same mapping)."""
    b, c, h, w = obs.shape
    flat = obs.reshape(b, c, h * w)
    return flat.transpose(0, 2, 1) if isinstance(flat, np.ndarray) else flat.transpose(1, 2)


def validate_graph(obs: np.ndarray) -> List[Dict[str, object]]:
    """Checks of section 12 of the methodology for one observation.

    Returns one row per check (``check``, ``expected``, ``actual``, ``passed``),
    as written to ``results/tables/graph_validation.csv``.
    """
    obs = np.asarray(obs, dtype=np.float32)
    _, height, width = obs.shape
    x, edges = observation_to_graph(obs)
    n = width * height
    rows: List[Dict[str, object]] = []

    def check(name: str, expected: object, actual: object) -> None:
        rows.append({"check": name, "expected": expected, "actual": actual, "passed": expected == actual})

    check("node count = W x H", n, x.shape[0])
    check("node feature dimension", NUM_NODE_FEATURES, x.shape[1])
    check("directed edges (8-neighbour, both directions)", expected_edge_count(width, height), edges.shape[1])
    dxs = np.abs(edges[1] % width - edges[0] % width)
    dys = np.abs(edges[1] // width - edges[0] // width)
    check("every edge joins 8-neighbours (|dx|,|dy| <= 1, not self)", True,
          bool(np.all((dxs <= 1) & (dys <= 1) & ((dxs + dys) > 0))))
    pairs = set(zip(edges[0].tolist(), edges[1].tolist()))
    check("edges stored in both directions", True, all((j, i) in pairs for i, j in pairs))
    check("no duplicate edges", edges.shape[1], len(pairs))
    degree = np.bincount(edges[0], minlength=n)
    corner = 3 if width > 1 and height > 1 else 1
    check("corner node degree", corner, int(degree[node_index(0, 0, width)]))
    if width > 2 and height > 2:
        check("interior node degree", 8, int(degree[node_index(1, 1, width)]))
    check("no isolated nodes", True, bool(n == 1 or degree.min() > 0))
    ok = all(np.array_equal(x[node_index(cx, cy, width)], obs[:, cy, cx])
             for cx, cy in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1),
                            (width // 2, height // 2)))
    check("coordinate mapping node(x, y) = y*W + x", True, ok)
    for c, name in enumerate(("health", "droplet", "goal")):
        check(f"{name} feature parity with grid (exact)", True,
              bool(np.array_equal(np.sort(x[:, c]), np.sort(obs[c].ravel()))
                   and np.array_equal(x[:, c], obs[c].ravel())))
    batch = np.stack([obs, obs])
    check("batched node features shape (B, N, 3)", (2, n, NUM_NODE_FEATURES),
          tuple(batch_to_node_features(batch).shape))
    return rows
