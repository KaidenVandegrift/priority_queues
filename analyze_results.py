"""
analyze_results.py
------------------
Reads results/summary.csv and produces:
  - results/plots/  (PNG charts)
  - Prints a formatted analysis table to stdout

Run: python analyze_results.py
"""

import sys, os, csv, math, json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[warn] matplotlib not found – skipping plots")


PLOT_DIR = "results/plots"
SUMMARY  = "results/summary.csv"

COLORS = {
    "Binary":    "#4C72B0",
    "Fibonacci": "#DD8452",
    "Pairing":   "#55A868",
}
MARKERS = {"Binary": "o", "Fibonacci": "s", "Pairing": "^"}

GRAPH_LABELS = {
    "sparse_random":       "Sparse Random",
    "dense_random":        "Dense Random",
    "grid":                "Grid",
    "worst_case_dijkstra": "Worst-Case (Dijkstra)",
    "worst_case_prim":     "Worst-Case (Prim)",
}


# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------

def load_summary():
    rows = []
    with open(SUMMARY) as f:
        for row in csv.DictReader(f):
            row["n"]      = int(row["n"])
            row["mean_s"] = float(row["mean_s"])
            row["std_s"]  = float(row["std_s"])
            rows.append(row)
    return rows


def group_by(rows, keys):
    """Return nested dict keyed by *keys* (tuple), value = list of rows."""
    d = defaultdict(list)
    for r in rows:
        k = tuple(r[k] for k in keys)
        d[k].append(r)
    return d


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_scaling(rows, algo, graph_type, out_dir):
    """Line plot: time vs n, one line per heap type."""
    subset = [r for r in rows
              if r["algorithm"] == algo and r["graph_type"] == graph_type]
    if not subset:
        return

    heaps = sorted(set(r["heap"] for r in subset))
    sizes = sorted(set(r["n"] for r in subset))

    fig, ax = plt.subplots(figsize=(6, 4))
    for heap in heaps:
        pts = sorted(
            [r for r in subset if r["heap"] == heap],
            key=lambda r: r["n"]
        )
        xs = [p["n"] for p in pts]
        ys = [p["mean_s"] * 1000 for p in pts]  # ms
        es = [p["std_s"]  * 1000 for p in pts]
        ax.errorbar(xs, ys, yerr=es,
                    label=heap,
                    color=COLORS[heap], marker=MARKERS[heap],
                    linewidth=2, capsize=4)

    ax.set_xlabel("Graph size n (vertices)", fontsize=12)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=12)
    ax.set_title(f"{algo} – {GRAPH_LABELS.get(graph_type, graph_type)}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    fname = f"{algo}_{graph_type}.png".replace(" ", "_")
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


def plot_heap_comparison(rows, algo, n_target, out_dir):
    """Bar chart: heap vs graph_type at a fixed n."""
    # Pick the closest available n
    available = sorted(set(r["n"] for r in rows if r["algorithm"] == algo))
    n = min(available, key=lambda x: abs(x - n_target))

    subset = [r for r in rows if r["algorithm"] == algo and r["n"] == n]
    if not subset:
        return

    graph_types = sorted(set(r["graph_type"] for r in subset))
    heaps       = ["Binary", "Fibonacci", "Pairing"]

    x     = range(len(graph_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, heap in enumerate(heaps):
        heights = []
        for gt in graph_types:
            match = [r for r in subset if r["heap"] == heap and r["graph_type"] == gt]
            heights.append(match[0]["mean_s"] * 1000 if match else 0)
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], heights, width,
                      label=heap, color=COLORS[heap], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels([GRAPH_LABELS.get(g, g) for g in graph_types],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=12)
    ax.set_title(f"{algo} heap comparison (n={n})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    fname = f"{algo}_heap_comparison_n{n}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


def plot_algo_comparison(rows, heap, graph_type, out_dir):
    """Compare Dijkstra vs Prim on the same graph type with a given heap."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for algo, color in [("Dijkstra", "#2196F3"), ("Prim", "#F44336")]:
        subset = [r for r in rows
                  if r["algorithm"] == algo
                  and r["graph_type"] == graph_type
                  and r["heap"] == heap]
        if not subset:
            continue
        subset.sort(key=lambda r: r["n"])
        xs = [r["n"] for r in subset]
        ys = [r["mean_s"] * 1000 for r in subset]
        es = [r["std_s"]  * 1000 for r in subset]
        ax.errorbar(xs, ys, yerr=es, label=algo, color=color,
                    linewidth=2, marker="o", capsize=4)

    ax.set_xlabel("n (vertices)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(f"Dijkstra vs Prim — {heap} heap — {GRAPH_LABELS.get(graph_type, graph_type)}", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    fname = f"algo_comparison_{heap}_{graph_type}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------
# Text summary
# -----------------------------------------------------------------------

def print_table(rows, algo, n_target):
    available = sorted(set(r["n"] for r in rows if r["algorithm"] == algo))
    if not available:
        return
    n = min(available, key=lambda x: abs(x - n_target))

    subset = [r for r in rows if r["algorithm"] == algo and r["n"] == n]
    graph_types = sorted(set(r["graph_type"] for r in subset))
    heaps       = ["Binary", "Fibonacci", "Pairing"]

    col_w = 14
    print(f"\n{'='*70}")
    print(f"  {algo}  (n ≈ {n})")
    print(f"{'='*70}")
    header = f"{'Graph type':<25}" + "".join(f"{h:>{col_w}}" for h in heaps)
    print(header)
    print("-" * 70)
    for gt in graph_types:
        row_str = f"{GRAPH_LABELS.get(gt, gt):<25}"
        for heap in heaps:
            match = [r for r in subset if r["heap"] == heap and r["graph_type"] == gt]
            if match:
                ms  = match[0]["mean_s"] * 1000
                std = match[0]["std_s"]  * 1000
                row_str += f"{ms:>9.2f}±{std:>3.1f}".rjust(col_w)
            else:
                row_str += f"{'N/A':>{col_w}}"
        print(row_str)
    print("-" * 70)
    print("  Values in ms (mean ± std over 5 trials)")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_summary()
    sizes = sorted(set(r["n"] for r in rows))
    n_large = max(sizes)

    # Text tables
    for algo in ["Dijkstra", "Prim"]:
        print_table(rows, algo, n_large)

    if not HAS_MATPLOTLIB:
        print("\nInstall matplotlib to generate plots.")
        sys.exit(0)

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Scaling plots (time vs n per graph type)
    for algo in ["Dijkstra", "Prim"]:
        for gt in set(r["graph_type"] for r in rows if r["algorithm"] == algo):
            plot_scaling(rows, algo, gt, PLOT_DIR)

    # Bar chart comparisons at largest n
    for algo in ["Dijkstra", "Prim"]:
        plot_heap_comparison(rows, algo, n_large, PLOT_DIR)

    # Dijkstra vs Prim comparison on shared graph types
    for heap in ["Binary", "Fibonacci", "Pairing"]:
        for gt in ["sparse_random", "dense_random", "grid"]:
            plot_algo_comparison(rows, heap, gt, PLOT_DIR)

    print(f"\nPlots saved to {PLOT_DIR}/")
    print(f"Files: {sorted(os.listdir(PLOT_DIR))}")
