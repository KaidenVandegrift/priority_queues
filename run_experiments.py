"""
run_experiments.py
------------------
Experimental performance analysis comparing:

  Algorithms : Dijkstra, Prim
  Heaps      : BinaryHeap (baseline), FibonacciHeap, PairingHeap
  Graph types: sparse random, dense random, grid, worst-case

Output
------
  results/raw_results.csv  – one row per (algo, heap, graph_type, size, trial)
  results/summary.csv      – mean ± std grouped by (algo, heap, graph_type, size)

Usage
-----
  python run_experiments.py              # full suite (slow)
  python run_experiments.py --quick     # small sizes only (fast smoke-test)
"""

import sys
import os
import time
import csv
import math
import statistics
import argparse
from typing import Callable, Type

# Make src importable
sys.path.insert(0, os.path.dirname(__file__))

from src.binary_heap      import BinaryHeap
from src.fibonacci_heap   import FibonacciHeap
from src.pairing_heap     import PairingHeap
from src.heap_interface   import AbstractHeap
from src.graph            import Graph
from src.graph_generators import (
    sparse_random, dense_random, grid_graph,
    worst_case_dijkstra, worst_case_prim,
)
from src.dijkstra import dijkstra
from src.prim     import prim

# -----------------------------------------------------------------------
# Experiment parameters
# -----------------------------------------------------------------------

HEAP_CLASSES: dict[str, Type[AbstractHeap]] = {
    "Binary":    BinaryHeap,
    "Fibonacci": FibonacciHeap,
    "Pairing":   PairingHeap,
}

SMALL_SIZES  = [100, 250, 500]
MEDIUM_SIZES = [1_000, 2_500, 5_000]
LARGE_SIZES  = [10_000, 25_000]
QUICK_SIZES  = [100, 300, 600, 1_000]

TRIALS = 5          # independent repetitions per cell


# -----------------------------------------------------------------------
# Graph factory registry
# -----------------------------------------------------------------------

def _make_sparse(n: int, seed: int) -> Graph:
    return sparse_random(n, seed=seed)

def _make_dense(n: int, seed: int) -> Graph:
    return dense_random(n, seed=seed)

def _make_grid(n: int, seed: int) -> Graph:
    k = max(2, int(math.isqrt(n)))
    return grid_graph(k, k, seed=seed)

def _make_worst_dijkstra(n: int, seed: int) -> Graph:
    return worst_case_dijkstra(n, seed=seed)

def _make_worst_prim(n: int, seed: int) -> Graph:
    return worst_case_prim(n, seed=seed)


GRAPH_FACTORIES: dict[str, Callable[[int, int], Graph]] = {
    "sparse_random":        _make_sparse,
    "dense_random":         _make_dense,
    "grid":                 _make_grid,
    "worst_case_dijkstra":  _make_worst_dijkstra,
    "worst_case_prim":      _make_worst_prim,
}

# Which graph types to use per algorithm
ALGO_GRAPHS = {
    "Dijkstra": ["sparse_random", "dense_random", "grid", "worst_case_dijkstra"],
    "Prim":     ["sparse_random", "dense_random", "grid", "worst_case_prim"],
}


# -----------------------------------------------------------------------
# Timing helper
# -----------------------------------------------------------------------

def time_run(algo_fn, graph: Graph, heap_cls: Type[AbstractHeap]) -> float:
    """Run algo_fn(graph, 0, heap_cls) and return wall-clock seconds."""
    t0 = time.perf_counter()
    algo_fn(graph, 0, heap_cls)
    t1 = time.perf_counter()
    return t1 - t0


# -----------------------------------------------------------------------
# Main experiment loop
# -----------------------------------------------------------------------

def run(sizes: list[int], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_results.csv")
    sum_path = os.path.join(output_dir, "summary.csv")

    raw_fields = ["algorithm", "heap", "graph_type", "n", "actual_n",
                  "num_edges", "trial", "time_s"]

    summary_rows: list[dict] = []

    with open(raw_path, "w", newline="") as raw_f:
        writer = csv.DictWriter(raw_f, fieldnames=raw_fields)
        writer.writeheader()

        total_cells = (
            len(sizes) * len(HEAP_CLASSES) *
            sum(len(v) for v in ALGO_GRAPHS.values())
        )
        done = 0

        for algo_name, algo_fn in [("Dijkstra", dijkstra), ("Prim", prim)]:
            graph_types = ALGO_GRAPHS[algo_name]

            for graph_type in graph_types:
                factory = GRAPH_FACTORIES[graph_type]

                for n in sizes:
                    for heap_name, heap_cls in HEAP_CLASSES.items():
                        times = []
                        actual_n = n
                        num_edges = 0

                        for trial in range(TRIALS):
                            g = factory(n, seed=trial * 1000 + n)
                            actual_n = g.num_vertices
                            num_edges = g.num_edges

                            try:
                                elapsed = time_run(algo_fn, g, heap_cls)
                            except Exception as e:
                                print(f"  ERROR {algo_name}/{heap_name}/{graph_type}/n={n}/t={trial}: {e}")
                                elapsed = float("nan")

                            times.append(elapsed)
                            writer.writerow({
                                "algorithm": algo_name,
                                "heap":       heap_name,
                                "graph_type": graph_type,
                                "n":          n,
                                "actual_n":   actual_n,
                                "num_edges":  num_edges,
                                "trial":      trial,
                                "time_s":     elapsed,
                            })

                        valid = [t for t in times if not math.isnan(t)]
                        mean = statistics.mean(valid) if valid else float("nan")
                        std  = statistics.stdev(valid) if len(valid) > 1 else 0.0

                        summary_rows.append({
                            "algorithm":  algo_name,
                            "heap":       heap_name,
                            "graph_type": graph_type,
                            "n":          n,
                            "actual_n":   actual_n,
                            "num_edges":  num_edges,
                            "mean_s":     mean,
                            "std_s":      std,
                            "trials":     len(valid),
                        })

                        done += 1
                        pct = 100 * done / total_cells
                        print(
                            f"[{pct:5.1f}%] {algo_name:8s} | {heap_name:10s} | "
                            f"{graph_type:22s} | n={n:6d} | "
                            f"mean={mean*1000:8.2f} ms"
                        )
                        raw_f.flush()

    # Write summary
    sum_fields = ["algorithm", "heap", "graph_type", "n", "actual_n",
                  "num_edges", "mean_s", "std_s", "trials"]
    with open(sum_path, "w", newline="") as sum_f:
        writer = csv.DictWriter(sum_f, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nResults written to {output_dir}/")
    print(f"  {raw_path}")
    print(f"  {sum_path}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Heap performance benchmark: Dijkstra vs Prim × Binary/Fibonacci/Pairing"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only small sizes for a fast smoke-test"
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory (default: results/)"
    )
    args = parser.parse_args()

    if args.quick:
        sizes = QUICK_SIZES
        print("=== QUICK MODE: sizes", sizes, "===\n")
    else:
        sizes = SMALL_SIZES + MEDIUM_SIZES + LARGE_SIZES
        print("=== FULL MODE: sizes", sizes, "===\n")

    run(sizes, args.output)
