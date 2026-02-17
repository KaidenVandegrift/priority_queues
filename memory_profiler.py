"""
memory_profiler.py
------------------
Measures actual memory footprint of each heap type by:

  1. Object-level analysis via sys.getsizeof + recursive traversal
  2. tracemalloc before/after construction of a heap with N nodes
  3. Per-node overhead comparison across heap types

Outputs a MemoryReport dict per heap with:
  node_size_bytes       : sys.getsizeof of one node object
  node_fields           : number of pointer/attribute fields per node
  heap_object_bytes     : sys.getsizeof of the heap container itself
  total_allocated_n100  : tracemalloc delta for a 100-node heap (bytes)
  bytes_per_node_n100   : total_allocated_n100 / 100
  qualitative_summary   : one-sentence description
"""

from __future__ import annotations
import sys
import gc
import tracemalloc
from typing import Type

from src.heap_interface import AbstractHeap
from src.binary_heap    import BinaryHeap,    BinaryHeapNode
from src.fibonacci_heap import FibonacciHeap, FibNode
from src.pairing_heap   import PairingHeap,   PairingNode


# ── per-node slot analysis ───────────────────────────────────────────

NODE_METADATA = {
    "Binary": {
        "class":        BinaryHeapNode,
        "slots":        ["key", "value", "_id"],
        "description":  "Flat object: key + value + integer ID. No pointer fields.",
        "extra_struct": "dict[int→int] position map (1 entry per node)",
    },
    "Fibonacci": {
        "class":        FibNode,
        "slots":        ["key", "value", "degree", "mark", "parent", "child", "left", "right"],
        "description":  "8-field node: 4 pointer fields (parent, child, left, right) + degree + mark.",
        "extra_struct": "None beyond the node graph itself",
    },
    "Pairing": {
        "class":        PairingNode,
        "slots":        ["key", "value", "child", "sibling", "prev"],
        "description":  "5-field node: 3 pointer fields (child, sibling, prev) + key + value.",
        "extra_struct": "None beyond the node tree itself",
    },
}


def _sizeof_node(cls, key=1.0, value=0) -> int:
    """Return sys.getsizeof of one instantiated node object."""
    if cls is BinaryHeapNode:
        return sys.getsizeof(cls(key, value, 0))
    else:
        return sys.getsizeof(cls(key, value))


def _measure_heap_allocation(heap_cls: Type[AbstractHeap], n: int) -> tuple[int, int]:
    """
    Return (total_allocated_bytes, heap_obj_bytes) for a fresh heap
    containing n nodes, measured via tracemalloc.
    """
    gc.collect()
    tracemalloc.start()
    snap0 = tracemalloc.take_snapshot()

    h = heap_cls()
    for i in range(n):
        h.insert(float(i), i)

    snap1 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap1.compare_to(snap0, 'lineno')
    total = sum(max(0, s.size_diff) for s in stats)
    heap_obj = sys.getsizeof(h)
    return total, heap_obj


def build_memory_report(n: int = 100) -> dict[str, dict]:
    """
    Build the full memory report for all three heap types at size n.
    """
    heap_classes = {
        "Binary":    BinaryHeap,
        "Fibonacci": FibonacciHeap,
        "Pairing":   PairingHeap,
    }

    report = {}
    for name, heap_cls in heap_classes.items():
        meta = NODE_METADATA[name]
        node_sz  = _sizeof_node(meta["class"])
        total_b, heap_obj_b = _measure_heap_allocation(heap_cls, n)

        report[name] = {
            "node_size_bytes":      node_sz,
            "node_fields":          len(meta["slots"]),
            "node_pointer_fields":  sum(1 for s in meta["slots"]
                                        if s in ("parent","child","left","right",
                                                  "sibling","prev")),
            "heap_object_bytes":    heap_obj_b,
            "total_allocated_bytes":total_b,
            "bytes_per_node":       total_b / n if n > 0 else 0,
            "extra_struct":         meta["extra_struct"],
            "description":          meta["description"],
            "slots":                meta["slots"],
        }
    return report


def print_memory_report(report: dict[str, dict]) -> None:
    print("\n" + "="*70)
    print("  MEMORY FOOTPRINT ANALYSIS")
    print("="*70)

    headers = ["Metric", "Binary", "Fibonacci", "Pairing"]
    rows = [
        ("Node size (sys.getsizeof, bytes)",
            [report[h]["node_size_bytes"] for h in ["Binary","Fibonacci","Pairing"]]),
        ("Fields per node",
            [report[h]["node_fields"] for h in ["Binary","Fibonacci","Pairing"]]),
        ("Pointer fields per node",
            [report[h]["node_pointer_fields"] for h in ["Binary","Fibonacci","Pairing"]]),
        ("Heap object bytes",
            [report[h]["heap_object_bytes"] for h in ["Binary","Fibonacci","Pairing"]]),
        ("Total allocated (n=100, bytes)",
            [report[h]["total_allocated_bytes"] for h in ["Binary","Fibonacci","Pairing"]]),
        ("Bytes per node (n=100)",
            [f"{report[h]['bytes_per_node']:.1f}" for h in ["Binary","Fibonacci","Pairing"]]),
    ]

    col_w = 20
    print(f"\n  {'Metric':<35}" + "".join(f"{h:>{col_w}}" for h in ["Binary","Fibonacci","Pairing"]))
    print("  " + "-"*95)
    for label, vals in rows:
        print(f"  {label:<35}" + "".join(f"{str(v):>{col_w}}" for v in vals))

    print()
    for name in ["Binary","Fibonacci","Pairing"]:
        print(f"  [{name}] {report[name]['description']}")
        print(f"          Extra structure: {report[name]['extra_struct']}")
        print(f"          Slots: {report[name]['slots']}")
        print()


if __name__ == "__main__":
    report = build_memory_report(n=100)
    print_memory_report(report)
