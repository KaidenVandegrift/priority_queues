"""
graph_generators.py
--------------------
Factory functions for the three graph families used in the experiments:

  1. Random graphs  – Erdős–Rényi G(n, p) with uniform random weights
  2. Grid graphs    – k×k lattice with uniform random weights
  3. Synthetic worst-case graphs (see below)

All graphs are undirected and connected (generators retry / augment as
needed to guarantee connectivity).

Worst-case graph families
-------------------------
For Dijkstra: a layered funnel graph that maximises decrease-key calls.
  - Source → V1 → V2 → … → V_{L-1} → sink, each layer fully connected
    to the next.  Edges set so that every vertex in layer i+1 is relaxed
    once per vertex in layer i (O(E) decrease-key operations).

For Prim: a graph where the minimum-spanning tree structure changes
  frequently as vertices are added, forcing many key updates.
"""

from __future__ import annotations
import random
import math
from .graph import Graph


# -----------------------------------------------------------------------
# 1. Random (Erdős–Rényi)
# -----------------------------------------------------------------------

def random_graph(
    n: int,
    p: float,
    weight_lo: float = 1.0,
    weight_hi: float = 100.0,
    seed: int = 42,
) -> Graph:
    """
    G(n, p) undirected graph with uniform random weights in [weight_lo, weight_hi].
    Adds a spanning tree first to guarantee connectivity.

    Parameters
    ----------
    n         : number of vertices
    p         : edge probability in (0, 1]
    weight_lo : minimum edge weight
    weight_hi : maximum edge weight
    seed      : RNG seed for reproducibility
    """
    rng = random.Random(seed)
    g = Graph(n)

    # Guarantee connectivity via a random spanning tree (Prüfer-style shuffle)
    vertices = list(range(n))
    rng.shuffle(vertices)
    for i in range(1, n):
        u = vertices[i - 1]
        v = vertices[i]
        w = rng.uniform(weight_lo, weight_hi)
        g.add_edge(u, v, w)

    # Add random edges (skip duplicates; simple approach)
    added = set()
    for i in range(1, n):
        added.add((vertices[i - 1], vertices[i]))
        added.add((vertices[i], vertices[i - 1]))

    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in added and rng.random() < p:
                w = rng.uniform(weight_lo, weight_hi)
                g.add_edge(u, v, w)
                added.add((u, v))
                added.add((v, u))

    return g


# -----------------------------------------------------------------------
# 2. Grid graphs
# -----------------------------------------------------------------------

def grid_graph(
    rows: int,
    cols: int,
    weight_lo: float = 1.0,
    weight_hi: float = 10.0,
    seed: int = 42,
) -> Graph:
    """
    k×k lattice (4-connected) with random weights.

    Vertex id for (r, c) = r * cols + c.
    """
    rng = random.Random(seed)
    n = rows * cols
    g = Graph(n)

    def vid(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            u = vid(r, c)
            if c + 1 < cols:
                w = rng.uniform(weight_lo, weight_hi)
                g.add_edge(u, vid(r, c + 1), w)
            if r + 1 < rows:
                w = rng.uniform(weight_lo, weight_hi)
                g.add_edge(u, vid(r + 1, c), w)

    return g


# -----------------------------------------------------------------------
# 3. Worst-case (decrease-key stress) graphs
# -----------------------------------------------------------------------

def worst_case_dijkstra(n: int, seed: int = 42) -> Graph:
    """
    Layered funnel graph designed to maximise decrease-key operations.

    Structure: L layers of roughly sqrt(n) vertices each.
    Every vertex in layer i is connected to every vertex in layer i+1.
    Weights are set so that each layer i+1 vertex is relaxed via all
    layer-i vertices, producing O(E) decrease-key calls.

    n is rounded down to layer_size^2 to get a clean layered structure.
    """
    rng = random.Random(seed)
    layer_size = max(2, int(math.isqrt(n)))
    num_layers = max(2, n // layer_size)
    actual_n = layer_size * num_layers
    g = Graph(actual_n)

    for layer in range(num_layers - 1):
        base_u = layer * layer_size
        base_v = (layer + 1) * layer_size
        for i in range(layer_size):
            u = base_u + i
            for j in range(layer_size):
                v = base_v + j
                # Weight decreases slightly for inner pairs so relaxations cascade
                w = 10.0 + rng.uniform(-0.5, 0.5)
                g.add_edge(u, v, w)

    return g


def worst_case_prim(n: int, seed: int = 42) -> Graph:
    """
    Star + complete graph: a hub connected to all vertices, plus dense
    random edges among vertices.  Forces Prim to keep updating keys as
    cheaper edges through the hub are continually beaten by direct edges.
    """
    rng = random.Random(seed)
    g = Graph(n)

    # Hub vertex = 0; connect hub to all others with moderate weight
    hub_weight = 50.0
    for v in range(1, n):
        g.add_edge(0, v, hub_weight)

    # Dense random edges with widely varying weights
    for u in range(1, n):
        for v in range(u + 1, n):
            if rng.random() < 0.4:   # moderately dense
                w = rng.uniform(1.0, 100.0)
                g.add_edge(u, v, w)

    return g


# -----------------------------------------------------------------------
# Helper: sparse vs dense parameterisation
# -----------------------------------------------------------------------

def sparse_random(n: int, seed: int = 42) -> Graph:
    """Sparse random graph: p ≈ 2 ln(n) / n (just above connectivity threshold)."""
    p = min(0.05, 2.5 * math.log(max(n, 2)) / max(n, 2))
    return random_graph(n, p, seed=seed)


def dense_random(n: int, seed: int = 42) -> Graph:
    """Dense random graph: p = 0.5 (roughly half all possible edges)."""
    return random_graph(n, 0.5, seed=seed)
