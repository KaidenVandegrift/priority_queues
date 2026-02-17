"""
instrumented_algorithms.py
---------------------------
Drop-in replacements for dijkstra() and prim() that record:

  • Total wall-clock time (whole algorithm)
  • Time attributed to extract_min calls only
  • Time attributed to decrease_key calls only
  • Time attributed to insert calls only
  • Counts of each operation
  • Peak heap size during execution
  • Number of edge relaxations attempted vs successful (decrease_key triggered)

Both functions return the algorithm result PLUS a MetricsRecord dict.

Usage
-----
  from instrumented_algorithms import dijkstra_metrics, prim_metrics
  from src.binary_heap import BinaryHeap
  from src.graph_generators import sparse_random

  g = sparse_random(1000)
  result, m = dijkstra_metrics(g, 0, BinaryHeap)
  print(m)
"""

from __future__ import annotations
import time
import math
import tracemalloc
import sys
from typing import Type

from src.heap_interface import AbstractHeap
from src.graph import Graph
from instrumented_heap import InstrumentedHeap


# ── cached instrumented classes (one per base class) ────────────────
_cache: dict = {}

def _get_instrumented(heap_cls: Type[AbstractHeap]) -> Type[AbstractHeap]:
    if heap_cls not in _cache:
        _cache[heap_cls] = InstrumentedHeap(heap_cls)
    return _cache[heap_cls]


# ─────────────────────────────────────────────────────────────────────
# Dijkstra with full metrics
# ─────────────────────────────────────────────────────────────────────

def dijkstra_metrics(
    graph: Graph,
    source: int,
    heap_cls: Type[AbstractHeap],
) -> tuple[tuple, dict]:
    """
    Run Dijkstra and return (algo_result, metrics_dict).

    algo_result = (dist, parent) as per dijkstra()
    metrics_dict keys:
      n_insert, n_extract_min, n_decrease_key,
      t_insert, t_extract_min, t_decrease_key,
      t_total_algo,   ← whole algorithm wall-clock
      t_heap_total,   ← sum of heap-op times
      t_non_heap,     ← t_total_algo − t_heap_total
      peak_size,
      n_relaxations_attempted,
      n_relaxations_successful,   ← == n_decrease_key
      mem_peak_bytes, mem_delta_bytes
    """
    IHeap = _get_instrumented(heap_cls)
    V = graph.num_vertices
    INF = math.inf

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    t_algo_start = time.perf_counter()

    dist   = [INF] * V
    parent = [-1]  * V
    dist[source] = 0.0

    heap = IHeap()
    handles = [None] * V
    n_relax_attempted = 0

    for v in range(V):
        handles[v] = heap.insert(dist[v], v)

    while not heap.is_empty():
        node = heap.extract_min()
        u = node.value
        d_u = node.key

        if d_u > dist[u]:
            continue

        for edge in graph.neighbors(u):
            v, w = edge.dst, edge.weight
            alt = dist[u] + w
            n_relax_attempted += 1
            if alt < dist[v]:
                dist[v]   = alt
                parent[v] = u
                heap.decrease_key(handles[v], alt)

    t_algo_total = time.perf_counter() - t_algo_start

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    mem_peak, _ = tracemalloc.get_traced_memory() if False else (0, 0)
    stats = snap_after.compare_to(snap_before, 'lineno')
    mem_delta = sum(s.size_diff for s in stats)
    mem_peak_b = sum(s.size for s in snap_after.statistics('lineno'))

    m = heap.metrics()
    t_heap = m["t_total"]
    m.update({
        "t_total_algo":             t_algo_total,
        "t_heap_total":             t_heap,
        "t_non_heap":               max(0.0, t_algo_total - t_heap),
        "n_relaxations_attempted":  n_relax_attempted,
        "n_relaxations_successful": m["n_decrease_key"],
        "mem_peak_bytes":           mem_peak_b,
        "mem_delta_bytes":          mem_delta,
    })
    return (dist, parent), m


# ─────────────────────────────────────────────────────────────────────
# Prim with full metrics
# ─────────────────────────────────────────────────────────────────────

def prim_metrics(
    graph: Graph,
    source: int,
    heap_cls: Type[AbstractHeap],
) -> tuple[tuple, dict]:
    """
    Run Prim's MST and return (algo_result, metrics_dict).

    algo_result = (mst_edges, total_weight) as per prim()
    metrics_dict: same keys as dijkstra_metrics.
    """
    IHeap = _get_instrumented(heap_cls)
    V = graph.num_vertices
    INF = math.inf

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    t_algo_start = time.perf_counter()

    key    = [INF] * V
    parent = [-1]  * V
    in_mst = [False] * V
    key[source] = 0.0

    heap    = IHeap()
    handles = [None] * V
    n_relax_attempted = 0

    for v in range(V):
        handles[v] = heap.insert(key[v], v)

    while not heap.is_empty():
        node = heap.extract_min()
        u = node.value

        if in_mst[u]:
            continue
        in_mst[u] = True

        for edge in graph.neighbors(u):
            v, w = edge.dst, edge.weight
            n_relax_attempted += 1
            if not in_mst[v] and w < key[v]:
                key[v]    = w
                parent[v] = u
                heap.decrease_key(handles[v], w)

    t_algo_total = time.perf_counter() - t_algo_start

    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap_after.compare_to(snap_before, 'lineno')
    mem_delta = sum(s.size_diff for s in stats)
    mem_peak_b = sum(s.size for s in snap_after.statistics('lineno'))

    mst_edges = []
    total_weight = 0.0
    for v in range(V):
        if parent[v] != -1:
            mst_edges.append((parent[v], v, key[v]))
            total_weight += key[v]

    m = heap.metrics()
    t_heap = m["t_total"]
    m.update({
        "t_total_algo":             t_algo_total,
        "t_heap_total":             t_heap,
        "t_non_heap":               max(0.0, t_algo_total - t_heap),
        "n_relaxations_attempted":  n_relax_attempted,
        "n_relaxations_successful": m["n_decrease_key"],
        "mem_peak_bytes":           mem_peak_b,
        "mem_delta_bytes":          mem_delta,
    })
    return (mst_edges, total_weight), m
