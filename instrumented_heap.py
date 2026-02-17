"""
instrumented_heap.py
--------------------
A transparent wrapper around any AbstractHeap that records:

  Per-operation counts
  ─────────────────────
  • n_insert          : total insert() calls
  • n_extract_min     : total extract_min() calls
  • n_decrease_key    : total decrease_key() calls
  • n_find_min        : total find_min() calls

  Per-operation cumulative wall-clock time (seconds)
  ────────────────────────────────────────────────────
  • t_insert
  • t_extract_min
  • t_decrease_key
  • t_find_min
  • t_total           : sum of all four

  Internal step counters (heap-type specific, via callbacks)
  ──────────────────────────────────────────────────────────
  • n_comparisons     : key comparisons (sift, consolidate, link, …)
  • n_swaps           : swap / pointer-reassignment operations
  • n_cuts            : Fibonacci/Pairing cut operations
  • n_cascading_cuts  : Fibonacci cascading-cut operations
  • n_links           : Fibonacci/Pairing link (meld) operations
  • n_consolidations  : Fibonacci consolidate passes
  • n_sift_up_steps   : Binary heap sift-up steps
  • n_sift_down_steps : Binary heap sift-down steps

  Peak size
  ──────────
  • peak_size         : maximum number of elements observed at any point

Usage
-----
  from instrumented_heap import InstrumentedHeap
  from src.fibonacci_heap import FibonacciHeap

  IH = InstrumentedHeap.make(FibonacciHeap)   # returns a NEW class
  heap = IH()                                  # instantiate
  node = heap.insert(3.0, "x")
  heap.decrease_key(node, 1.0)
  heap.extract_min()
  print(heap.metrics())                        # dict of all counters
"""

from __future__ import annotations
import time
from typing import Any, Type, Optional
from src.heap_interface import AbstractHeap, HeapNode


class MetricsStore:
    """Plain container for all counters; also acts as a callback sink."""
    __slots__ = (
        "n_insert", "n_extract_min", "n_decrease_key", "n_find_min",
        "t_insert", "t_extract_min", "t_decrease_key", "t_find_min",
        "n_comparisons", "n_swaps", "n_cuts", "n_cascading_cuts",
        "n_links", "n_consolidations",
        "n_sift_up_steps", "n_sift_down_steps",
        "peak_size",
    )

    def __init__(self):
        for slot in self.__slots__:
            setattr(self, slot, 0)

    def to_dict(self) -> dict:
        d = {s: getattr(self, s) for s in self.__slots__}
        d["t_total"] = d["t_insert"] + d["t_extract_min"] + d["t_decrease_key"] + d["t_find_min"]
        return d

    # ── callback hooks ────────────────────────────────────────────────
    def on_compare(self, n: int = 1):    self.n_comparisons      += n
    def on_swap(self, n: int = 1):       self.n_swaps            += n
    def on_cut(self, n: int = 1):        self.n_cuts             += n
    def on_cascading_cut(self, n=1):     self.n_cascading_cuts   += n
    def on_link(self, n: int = 1):       self.n_links            += n
    def on_consolidate(self, n: int=1):  self.n_consolidations   += n
    def on_sift_up(self, n: int = 1):    self.n_sift_up_steps    += n
    def on_sift_down(self, n: int = 1):  self.n_sift_down_steps  += n


def InstrumentedHeap(base_cls: Type[AbstractHeap]) -> Type[AbstractHeap]:
    """
    Factory: given a heap CLASS, return a NEW class that wraps it with
    full metrics instrumentation.

    The returned class has an extra .metrics() → dict method and a
    .store attribute (MetricsStore).
    """

    class _Instrumented(base_cls):
        def __init__(self):
            super().__init__()
            self.store = MetricsStore()
            # Inject the metrics store into the underlying heap's callbacks
            # so internal step counters work without modifying heap source.
            # We patch _on_* hooks if the base class declared them.
            if hasattr(self, '_metrics'):
                self._metrics = self.store

        # ── count + time each operation ──────────────────────────────

        def insert(self, key: float, value: Any) -> HeapNode:
            self.store.n_insert += 1
            t0 = time.perf_counter()
            node = super().insert(key, value)
            self.store.t_insert += time.perf_counter() - t0
            sz = len(self)
            if sz > self.store.peak_size:
                self.store.peak_size = sz
            return node

        def find_min(self) -> Optional[HeapNode]:
            self.store.n_find_min += 1
            t0 = time.perf_counter()
            result = super().find_min()
            self.store.t_find_min += time.perf_counter() - t0
            return result

        def extract_min(self) -> Optional[HeapNode]:
            self.store.n_extract_min += 1
            t0 = time.perf_counter()
            result = super().extract_min()
            self.store.t_extract_min += time.perf_counter() - t0
            return result

        def decrease_key(self, node: HeapNode, new_key: float) -> None:
            self.store.n_decrease_key += 1
            t0 = time.perf_counter()
            super().decrease_key(node, new_key)
            self.store.t_decrease_key += time.perf_counter() - t0

        def metrics(self) -> dict:
            return self.store.to_dict()

        def reset_metrics(self):
            self.store = MetricsStore()

    _Instrumented.__name__ = f"Instrumented{base_cls.__name__}"
    _Instrumented.__qualname__ = _Instrumented.__name__
    return _Instrumented
