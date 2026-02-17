"""
graph.py
--------
Weighted undirected / directed graph stored as adjacency lists.

Used by both Dijkstra's and Prim's algorithms.
"""

from __future__ import annotations
from typing import Iterator, NamedTuple


class Edge(NamedTuple):
    dst: int      # destination vertex
    weight: float # non-negative edge weight


class Graph:
    """
    Weighted graph (undirected by default).

    Parameters
    ----------
    n         : number of vertices  (0 … n-1)
    directed  : if True, add_edge only adds one direction
    """

    def __init__(self, n: int, directed: bool = False):
        self.n = n
        self.directed = directed
        self._adj: list[list[Edge]] = [[] for _ in range(n)]
        self._m: int = 0   # edge count (each undirected edge counted once)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_edge(self, u: int, v: int, weight: float) -> None:
        """Add edge u–v (both directions if undirected)."""
        self._adj[u].append(Edge(v, weight))
        self._m += 1
        if not self.directed:
            self._adj[v].append(Edge(u, weight))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def neighbors(self, u: int) -> list[Edge]:
        return self._adj[u]

    def degree(self, u: int) -> int:
        return len(self._adj[u])

    @property
    def num_vertices(self) -> int:
        return self.n

    @property
    def num_edges(self) -> int:
        return self._m

    def __repr__(self) -> str:
        return f"Graph(V={self.n}, E={self._m}, directed={self.directed})"
