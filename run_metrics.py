"""
run_metrics.py
--------------
Master metrics collection script for Part 4.

Collects and saves all four metric categories:
  1. Total runtime (wall-clock, full algorithm)
  2. Time in extract-min and decrease-key (per-operation breakdown)
  3. Number of heap operations (insert / extract / decrease-key counts)
  4. Memory usage (per-node bytes, total allocation)

Outputs
-------
  metrics_results/
    op_metrics.csv        per (algo, heap, graph_type, n, trial)
    op_summary.csv        mean ± std grouped by (algo, heap, graph_type, n)
    memory_report.csv

Run:  python run_metrics.py [--quick]
"""

import sys, os, csv, math, statistics, argparse, tracemalloc, time
sys.path.insert(0, os.path.dirname(__file__))

from src.binary_heap      import BinaryHeap
from src.fibonacci_heap   import FibonacciHeap
from src.pairing_heap     import PairingHeap
from src.graph_generators import sparse_random, dense_random, grid_graph, worst_case_dijkstra, worst_case_prim
from src.graph            import Graph
from instrumented_algorithms import dijkstra_metrics, prim_metrics
from memory_profiler         import build_memory_report

import math as _math

HEAP_CLASSES = {
    "Binary":    BinaryHeap,
    "Fibonacci": FibonacciHeap,
    "Pairing":   PairingHeap,
}

GRAPH_FACTORIES = {
    "sparse_random":       lambda n, s: sparse_random(n, seed=s),
    "dense_random":        lambda n, s: dense_random(n, seed=s),
    "grid":                lambda n, s: grid_graph(max(2,int(_math.isqrt(n))), max(2,int(_math.isqrt(n))), seed=s),
    "worst_case_dijkstra": lambda n, s: worst_case_dijkstra(n, seed=s),
    "worst_case_prim":     lambda n, s: worst_case_prim(n, seed=s),
}

ALGO_GRAPHS = {
    "Dijkstra": ["sparse_random", "dense_random", "grid", "worst_case_dijkstra"],
    "Prim":     ["sparse_random", "dense_random", "grid", "worst_case_prim"],
}

ALGO_FNS = {"Dijkstra": dijkstra_metrics, "Prim": prim_metrics}

QUICK_SIZES = [100, 300, 600, 1000]
FULL_SIZES  = [100, 250, 500, 1000, 2500, 5000]
TRIALS      = 5
OUT_DIR     = "metrics_results"


# ─────────────────────────────────────────────────────────────────────
# 1 + 2 + 3: Operation metrics (runtime, per-op time, counts)
# ─────────────────────────────────────────────────────────────────────

RAW_FIELDS = [
    "algorithm", "heap", "graph_type", "n", "actual_n", "num_edges", "trial",
    # Timing
    "t_total_algo", "t_insert", "t_extract_min", "t_decrease_key",
    "t_heap_total", "t_non_heap",
    # Fractions
    "frac_extract", "frac_decrease_key", "frac_insert",
    # Counts
    "n_insert", "n_extract_min", "n_decrease_key",
    "n_relaxations_attempted", "n_relaxations_successful",
    "peak_size",
    # Memory
    "mem_peak_bytes", "mem_delta_bytes",
]

SUM_FIELDS = [
    "algorithm", "heap", "graph_type", "n", "actual_n", "num_edges",
    # mean/std runtime
    "mean_t_total", "std_t_total",
    "mean_t_extract", "std_t_extract",
    "mean_t_decrease", "std_t_decrease",
    "mean_t_insert", "std_t_insert",
    # mean fractions
    "mean_frac_extract", "mean_frac_decrease", "mean_frac_insert",
    # mean counts
    "mean_n_insert", "mean_n_extract", "mean_n_decrease",
    "mean_n_relax_attempted", "mean_n_relax_successful",
    "mean_relax_rate",    # successful / attempted
    "mean_peak_size",
    # memory
    "mean_mem_bytes", "std_mem_bytes",
    "mean_bytes_per_node",
    "trials",
]


def collect_op_metrics(sizes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "op_metrics.csv")
    sum_path = os.path.join(out_dir, "op_summary.csv")

    summary_rows = []

    with open(raw_path, "w", newline="") as rf:
        writer = csv.DictWriter(rf, fieldnames=RAW_FIELDS)
        writer.writeheader()

        total = len(sizes) * len(HEAP_CLASSES) * sum(len(v) for v in ALGO_GRAPHS.values())
        done = 0

        for algo_name, graphs in ALGO_GRAPHS.items():
            fn = ALGO_FNS[algo_name]
            for gt in graphs:
                factory = GRAPH_FACTORIES[gt]
                for n in sizes:
                    for heap_name, heap_cls in HEAP_CLASSES.items():
                        rows_this = []
                        for trial in range(TRIALS):
                            g = factory(n, trial * 1000 + n)
                            actual_n = g.num_vertices
                            num_edges = g.num_edges
                            try:
                                _, m = fn(g, 0, heap_cls)
                            except Exception as e:
                                print(f"  ERROR {algo_name}/{heap_name}/{gt}/n={n}/t={trial}: {e}")
                                continue

                            t_heap = m["t_heap_total"]
                            frac_e = m["t_extract_min"] / t_heap if t_heap > 0 else 0
                            frac_d = m["t_decrease_key"] / t_heap if t_heap > 0 else 0
                            frac_i = m["t_insert"]       / t_heap if t_heap > 0 else 0

                            row = {
                                "algorithm": algo_name,
                                "heap":       heap_name,
                                "graph_type": gt,
                                "n":          n,
                                "actual_n":   actual_n,
                                "num_edges":  num_edges,
                                "trial":      trial,
                                "t_total_algo":   m["t_total_algo"],
                                "t_insert":       m["t_insert"],
                                "t_extract_min":  m["t_extract_min"],
                                "t_decrease_key": m["t_decrease_key"],
                                "t_heap_total":   t_heap,
                                "t_non_heap":     m["t_non_heap"],
                                "frac_extract":   frac_e,
                                "frac_decrease_key": frac_d,
                                "frac_insert":    frac_i,
                                "n_insert":       m["n_insert"],
                                "n_extract_min":  m["n_extract_min"],
                                "n_decrease_key": m["n_decrease_key"],
                                "n_relaxations_attempted":  m["n_relaxations_attempted"],
                                "n_relaxations_successful": m["n_relaxations_successful"],
                                "peak_size":      m["peak_size"],
                                "mem_peak_bytes": m["mem_peak_bytes"],
                                "mem_delta_bytes":m["mem_delta_bytes"],
                            }
                            writer.writerow(row)
                            rows_this.append(row)

                        if rows_this:
                            def _mean(k): return statistics.mean(r[k] for r in rows_this)
                            def _std(k):
                                v = [r[k] for r in rows_this]
                                return statistics.stdev(v) if len(v)>1 else 0.0
                            ra = _mean("n_relaxations_attempted")
                            rs = _mean("n_relaxations_successful")
                            summary_rows.append({
                                "algorithm": algo_name,
                                "heap":       heap_name,
                                "graph_type": gt,
                                "n":          n,
                                "actual_n":   rows_this[0]["actual_n"],
                                "num_edges":  rows_this[0]["num_edges"],
                                "mean_t_total":   _mean("t_total_algo"),
                                "std_t_total":    _std("t_total_algo"),
                                "mean_t_extract": _mean("t_extract_min"),
                                "std_t_extract":  _std("t_extract_min"),
                                "mean_t_decrease":_mean("t_decrease_key"),
                                "std_t_decrease": _std("t_decrease_key"),
                                "mean_t_insert":  _mean("t_insert"),
                                "std_t_insert":   _std("t_insert"),
                                "mean_frac_extract":  _mean("frac_extract"),
                                "mean_frac_decrease": _mean("frac_decrease_key"),
                                "mean_frac_insert":   _mean("frac_insert"),
                                "mean_n_insert":   _mean("n_insert"),
                                "mean_n_extract":  _mean("n_extract_min"),
                                "mean_n_decrease": _mean("n_decrease_key"),
                                "mean_n_relax_attempted":  ra,
                                "mean_n_relax_successful": rs,
                                "mean_relax_rate": rs/ra if ra>0 else 0,
                                "mean_peak_size":  _mean("peak_size"),
                                "mean_mem_bytes":  _mean("mem_delta_bytes"),
                                "std_mem_bytes":   _std("mem_delta_bytes"),
                                "mean_bytes_per_node": _mean("mem_delta_bytes") / max(rows_this[0]["actual_n"],1),
                                "trials":         len(rows_this),
                            })

                        done += 1
                        pct = 100*done/total
                        last = summary_rows[-1] if summary_rows else {}
                        t_ms = last.get("mean_t_total",0)*1000
                        dk_ms = last.get("mean_t_decrease",0)*1000
                        ext_ms = last.get("mean_t_extract",0)*1000
                        print(f"[{pct:5.1f}%] {algo_name:8s}|{heap_name:10s}|{gt:22s}|n={n:5d} "
                              f"total={t_ms:7.2f}ms  ext={ext_ms:6.2f}ms  dk={dk_ms:6.2f}ms")
                        rf.flush()

    with open(sum_path, "w", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=SUM_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


# ─────────────────────────────────────────────────────────────────────
# 4: Memory report
# ─────────────────────────────────────────────────────────────────────

def collect_memory_metrics(out_dir):
    print("\n[Memory] Measuring heap allocation footprints…")
    sizes = [50, 100, 500, 1000]
    rows = []
    for n in sizes:
        report = build_memory_report(n=n)
        for heap_name, data in report.items():
            rows.append({
                "heap": heap_name,
                "n":    n,
                "node_size_bytes":      data["node_size_bytes"],
                "node_fields":          data["node_fields"],
                "node_pointer_fields":  data["node_pointer_fields"],
                "total_allocated_bytes":data["total_allocated_bytes"],
                "bytes_per_node":       f"{data['bytes_per_node']:.1f}",
                "heap_object_bytes":    data["heap_object_bytes"],
            })

    path = os.path.join(out_dir, "memory_report.csv")
    fields = ["heap","n","node_size_bytes","node_fields","node_pointer_fields",
              "total_allocated_bytes","bytes_per_node","heap_object_bytes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"  Memory report → {path}")
    return rows



# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=OUT_DIR)
    args = parser.parse_args()

    sizes = QUICK_SIZES if args.quick else FULL_SIZES
    print(f"\n{'='*60}")
    print(f"  PART 4 METRICS COLLECTION  (sizes={sizes})")
    print(f"{'='*60}\n")

    summary = collect_op_metrics(sizes, args.output)
    mem_rows = collect_memory_metrics(args.output)

    print(f"\n{'='*60}")
    print(f"  All metrics saved to {args.output}/")
    print(f"{'='*60}")
