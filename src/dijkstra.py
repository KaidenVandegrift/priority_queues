"""
dijkstra.py
-----------
Dijkstra's single-source shortest-path algorithm.

The implementation is fully heap-agnostic: it accepts any object that
satisfies the AbstractHeap interface (BinaryHeap, FibonacciHeap,
PairingHeap, …).

Complexity (using heap H)
--------------------------
  O((V + E) · T_extract + E · T_decrease_key)

where T_extract and T_decrease_key are the costs for the chosen heap:

  Heap           extract-min     decrease-key
  ───────────────────────────────────────────
  Binary         O(log V)        O(log V)
  Fibonacci      O(log V) amort  O(1) amort
  Pairing        O(log V) amort  O(log V) amort

Algorithm
---------
1. Initialise dist[source] = 0, dist[v] = ∞ for all others.
2. Insert all vertices into the priority queue.
3. Repeat until the queue is empty:
   a. u = extract_min()
   b. For each neighbor v of u with edge weight w:
      if dist[u] + w < dist[v]:
          dist[v] = dist[u] + w
          decrease_key(handle[v], dist[v])
          parent[v] = u
4. Return (dist, parent).
"""

from __future__ import annotations
import math
from typing import Type

from .heap_interface import AbstractHeap
from .graph import Graph


def dijkstra(
    graph: Graph,
    source: int,
    heap_cls: Type[AbstractHeap],
) -> tuple[list[float], list[int]]:
    """
    Run Dijkstra from *source* on *graph* using *heap_cls* as the PQ.

    Parameters
    ----------
    graph    : Graph  (undirected or directed, non-negative weights)
    source   : source vertex index
    heap_cls : a class (not instance) implementing AbstractHeap

    Returns
    -------
    dist   : list of length V, dist[v] = shortest distance source → v
             (math.inf if unreachable)
    parent : list of length V, parent[v] = predecessor on shortest path
             (-1 for source and unreachable vertices)
    """
    V = graph.num_vertices
    INF = math.inf

    dist   = [INF] * V
    parent = [-1]  * V
    dist[source] = 0.0

    heap = heap_cls()
    handles = [None] * V          # handles[v] = node returned by insert()

    # Insert all vertices
    for v in range(V):
        handles[v] = heap.insert(dist[v], v)

    while not heap.is_empty():
        node = heap.extract_min()
        u = node.value
        d_u = node.key            # this is dist[u] at the time of extraction

        # Skip stale entries (shouldn't occur with decrease_key, but guard anyway)
        if d_u > dist[u]:
            continue

        for edge in graph.neighbors(u):
            v, w = edge.dst, edge.weight
            alt = dist[u] + w
            if alt < dist[v]:
                dist[v]   = alt
                parent[v] = u
                heap.decrease_key(handles[v], alt)

    return dist, parent
