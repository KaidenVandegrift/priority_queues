"""
prim.py
-------
Prim's minimum spanning tree algorithm.

The implementation is fully heap-agnostic: it accepts any object
satisfying the AbstractHeap interface.

Complexity (using heap H)
--------------------------
  O((V + E) · T_extract + E · T_decrease_key)

Same asymptotic structure as Dijkstra; the Fibonacci heap gives the
classic O(E + V log V) MST time.

Algorithm
---------
1. key[v] = ∞ for all v; key[source] = 0; parent[v] = -1.
2. Insert all vertices into the priority queue with key as priority.
3. Repeat until the queue is empty:
   a. u = extract_min() with key k_u
   b. Mark u as in-MST.
   c. For each neighbor v of u with edge weight w:
      if v not in MST and w < key[v]:
          key[v]    = w
          parent[v] = u
          decrease_key(handle[v], w)
4. Return (mst_edges, total_weight).

Note: parent[source] = -1 (root of spanning tree).
"""

from __future__ import annotations
import math
from typing import Type

from .heap_interface import AbstractHeap
from .graph import Graph


def prim(
    graph: Graph,
    source: int,
    heap_cls: Type[AbstractHeap],
) -> tuple[list[tuple[int, int, float]], float]:
    """
    Run Prim's MST from *source* on *graph* using *heap_cls* as the PQ.

    Parameters
    ----------
    graph    : Graph  (undirected, non-negative weights)
    source   : starting vertex
    heap_cls : a class (not instance) implementing AbstractHeap

    Returns
    -------
    mst_edges    : list of (u, v, weight) triples forming the MST
    total_weight : sum of MST edge weights (0.0 if graph has ≤ 1 vertex)
    """
    V = graph.num_vertices
    INF = math.inf

    key     = [INF] * V
    parent  = [-1]  * V
    in_mst  = [False] * V
    key[source] = 0.0

    heap    = heap_cls()
    handles = [None] * V

    for v in range(V):
        handles[v] = heap.insert(key[v], v)

    while not heap.is_empty():
        node = heap.extract_min()
        u = node.value

        # Stale extraction guard
        if in_mst[u]:
            continue
        in_mst[u] = True

        for edge in graph.neighbors(u):
            v, w = edge.dst, edge.weight
            if not in_mst[v] and w < key[v]:
                key[v]    = w
                parent[v] = u
                heap.decrease_key(handles[v], w)

    # Collect MST edges (skip root whose parent == -1)
    mst_edges: list[tuple[int, int, float]] = []
    total_weight = 0.0
    for v in range(V):
        if parent[v] != -1:
            mst_edges.append((parent[v], v, key[v]))
            total_weight += key[v]

    return mst_edges, total_weight
