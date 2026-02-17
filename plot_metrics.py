"""
plot_metrics.py  —  Part 4 visualisations
Reads metrics_results/ CSVs and produces charts in metrics_results/plots/
"""
import os, csv, math, statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PLOT_DIR = "metrics_results/plots"
COLORS   = {"Binary": "#4C72B0", "Fibonacci": "#DD8452", "Pairing": "#55A868"}
MARKERS  = {"Binary": "o", "Fibonacci": "s", "Pairing": "^"}
HEAPS    = ["Binary", "Fibonacci", "Pairing"]

GRAPH_SHORT = {
    "sparse_random":       "Sparse",
    "dense_random":        "Dense",
    "grid":                "Grid",
    "worst_case_dijkstra": "Worst-Case",
    "worst_case_prim":     "Worst-Case",
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def flt(x):
    try: return float(x)
    except: return 0.0


os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
summary = load_csv("metrics_results/op_summary.csv")
memory  = load_csv("metrics_results/memory_report.csv")

# ─────────────────────────────────────────────────────────────────────
# FIG 1: Per-operation time breakdown — stacked bar, n=1000
# ─────────────────────────────────────────────────────────────────────
def fig_time_breakdown():
    for algo in ["Dijkstra", "Prim"]:
        rows = [r for r in summary if r["algorithm"]==algo and int(r["n"])==1000]
        graph_types = sorted(set(r["graph_type"] for r in rows))
        x = np.arange(len(graph_types))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))
        for i, heap in enumerate(HEAPS):
            t_ext = [flt(next((r["mean_t_extract"] for r in rows
                               if r["heap"]==heap and r["graph_type"]==gt), 0))*1000 for gt in graph_types]
            t_dk  = [flt(next((r["mean_t_decrease"] for r in rows
                               if r["heap"]==heap and r["graph_type"]==gt), 0))*1000 for gt in graph_types]
            t_ins = [flt(next((r["mean_t_insert"] for r in rows
                               if r["heap"]==heap and r["graph_type"]==gt), 0))*1000 for gt in graph_types]

            offset = (i-1)*width
            base = ax.bar(x+offset, t_ins, width, label=f"{heap} insert",
                          color=COLORS[heap], alpha=0.35, hatch="//")
            ax.bar(x+offset, t_ext, width, bottom=t_ins, label=f"{heap} extract-min",
                   color=COLORS[heap], alpha=0.70)
            t_bot = [a+b for a,b in zip(t_ins,t_ext)]
            ax.bar(x+offset, t_dk, width, bottom=t_bot, label=f"{heap} decrease-key",
                   color=COLORS[heap], alpha=1.0, hatch="xx")

        ax.set_xticks(x)
        ax.set_xticklabels([GRAPH_SHORT.get(g,g) for g in graph_types], fontsize=10)
        ax.set_ylabel("Heap operation time (ms)", fontsize=12)
        ax.set_title(f"{algo} — Per-operation time breakdown (n=1,000)", fontsize=13)

        # Custom legend
        handles = [mpatches.Patch(color=COLORS[h], label=h) for h in HEAPS]
        style_handles = [
            mpatches.Patch(color="gray", alpha=0.35, hatch="//", label="insert"),
            mpatches.Patch(color="gray", alpha=0.70, label="extract-min"),
            mpatches.Patch(color="gray", alpha=1.00, hatch="xx", label="decrease-key"),
        ]
        ax.legend(handles=handles+style_handles, fontsize=8, ncol=2, loc="upper left")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/{algo}_time_breakdown.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {algo}_time_breakdown.png")


# ─────────────────────────────────────────────────────────────────────
# FIG 2: Extract-min vs decrease-key time ratio (frac over n)
# ─────────────────────────────────────────────────────────────────────
def fig_op_fractions():
    for algo in ["Dijkstra", "Prim"]:
        # Pick sparse_random as representative
        rows = [r for r in summary
                if r["algorithm"]==algo and r["graph_type"]=="sparse_random"]
        rows.sort(key=lambda r: int(r["n"]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        for heap in HEAPS:
            sub = [r for r in rows if r["heap"]==heap]
            ns    = [int(r["n"]) for r in sub]
            f_ext = [flt(r["mean_frac_extract"])*100 for r in sub]
            f_dk  = [flt(r["mean_frac_decrease"])*100 for r in sub]
            ax1.plot(ns, f_ext, label=heap, color=COLORS[heap], marker=MARKERS[heap], linewidth=2)
            ax2.plot(ns, f_dk,  label=heap, color=COLORS[heap], marker=MARKERS[heap], linewidth=2)

        for ax, title in [(ax1, "extract-min"), (ax2, "decrease-key")]:
            ax.set_xlabel("n", fontsize=11)
            ax.set_ylabel("% of heap time", fontsize=11)
            ax.set_title(f"{algo} — {title} share of heap time\n(sparse random)", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_ylim(0, 100)

        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/{algo}_op_fractions.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {algo}_op_fractions.png")


# ─────────────────────────────────────────────────────────────────────
# FIG 3: Operation counts vs n (insert / extract / decrease-key)
# ─────────────────────────────────────────────────────────────────────
def fig_op_counts():
    for algo in ["Dijkstra", "Prim"]:
        for gt in ["sparse_random", "dense_random"]:
            rows = [r for r in summary if r["algorithm"]==algo and r["graph_type"]==gt]
            if not rows: continue
            rows.sort(key=lambda r: int(r["n"]))
            ns = sorted(set(int(r["n"]) for r in rows))

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            ops = [("mean_n_insert","n_insert","Inserts"),
                   ("mean_n_extract","n_extract-min","extract-min calls"),
                   ("mean_n_decrease","n_decrease-key","decrease-key calls")]

            for ax, (key, _, label) in zip(axes, ops):
                for heap in HEAPS:
                    sub = sorted([r for r in rows if r["heap"]==heap], key=lambda r: int(r["n"]))
                    ys = [flt(r[key]) for r in sub]
                    xs = [int(r["n"]) for r in sub]
                    ax.plot(xs, ys, label=heap, color=COLORS[heap],
                            marker=MARKERS[heap], linewidth=2)
                ax.set_xlabel("n", fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.set_title(label, fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.4)

            fig.suptitle(f"{algo} — Operation counts ({GRAPH_SHORT.get(gt,gt)} graph)", fontsize=12, y=1.01)
            fig.tight_layout()
            fname = f"{algo}_{gt}_op_counts.png"
            fig.savefig(f"{PLOT_DIR}/{fname}", dpi=150)
            plt.close(fig)
            print(f"  Saved {fname}")


# ─────────────────────────────────────────────────────────────────────
# FIG 4: Relaxation rate (successful dk / attempted relaxations)
# ─────────────────────────────────────────────────────────────────────
def fig_relax_rate():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, algo in zip(axes, ["Dijkstra", "Prim"]):
        gts = ["sparse_random","dense_random","grid"]
        n_target = 1000
        rows = [r for r in summary if r["algorithm"]==algo and int(r["n"])==n_target
                and r["graph_type"] in gts]

        x = np.arange(len(gts))
        width = 0.25
        for i, heap in enumerate(HEAPS):
            rates = []
            for gt in gts:
                match = [r for r in rows if r["heap"]==heap and r["graph_type"]==gt]
                rates.append(flt(match[0]["mean_relax_rate"])*100 if match else 0)
            ax.bar(x+(i-1)*width, rates, width, label=heap, color=COLORS[heap], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([GRAPH_SHORT.get(g,g) for g in gts], fontsize=10)
        ax.set_ylabel("Relaxation success rate (%)", fontsize=11)
        ax.set_title(f"{algo} — % edge relaxations triggering decrease-key (n=1,000)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/relaxation_rate.png", dpi=150)
    plt.close(fig)
    print("  Saved relaxation_rate.png")


# ─────────────────────────────────────────────────────────────────────
# FIG 5: Memory — bytes per node by heap & n
# ─────────────────────────────────────────────────────────────────────
def fig_memory():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Bytes per node vs n
    for heap in HEAPS:
        sub = [r for r in memory if r["heap"]==heap]
        sub.sort(key=lambda r: int(r["n"]))
        ns  = [int(r["n"]) for r in sub]
        bpn = [flt(r["bytes_per_node"]) for r in sub]
        ax1.plot(ns, bpn, label=heap, color=COLORS[heap], marker=MARKERS[heap], linewidth=2)

    ax1.set_xlabel("n (nodes in heap)", fontsize=11)
    ax1.set_ylabel("Bytes per node (allocated)", fontsize=11)
    ax1.set_title("Memory allocation per node vs heap size", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Node struct comparison at n=100
    n100 = [r for r in memory if int(r["n"])==100]
    attrs = ["node_size_bytes", "node_fields", "node_pointer_fields"]
    labels = ["Node object\nsize (bytes)", "Fields\nper node", "Pointer\nfields"]
    x = np.arange(len(attrs))
    width = 0.25
    for i, heap in enumerate(HEAPS):
        match = next((r for r in n100 if r["heap"]==heap), None)
        if not match: continue
        vals = [flt(match["node_size_bytes"]), flt(match["node_fields"]),
                flt(match["node_pointer_fields"])]
        ax2.bar(x+(i-1)*width, vals, width, label=heap, color=COLORS[heap], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Count / Size", fontsize=11)
    ax2.set_title("Node struct comparison", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/memory_analysis.png", dpi=150)
    plt.close(fig)
    print("  Saved memory_analysis.png")


# ─────────────────────────────────────────────────────────────────────
# FIG 6: Extract-min time scaling — all graph types
# ─────────────────────────────────────────────────────────────────────
def fig_extract_scaling():
    for algo in ["Dijkstra", "Prim"]:
        rows = [r for r in summary if r["algorithm"]==algo]
        gts  = sorted(set(r["graph_type"] for r in rows))
        fig, axes = plt.subplots(1, len(gts), figsize=(4*len(gts), 4), sharey=False)
        if len(gts) == 1: axes = [axes]

        for ax, gt in zip(axes, gts):
            sub = [r for r in rows if r["graph_type"]==gt]
            sub.sort(key=lambda r: int(r["n"]))
            for heap in HEAPS:
                h_rows = sorted([r for r in sub if r["heap"]==heap], key=lambda r: int(r["n"]))
                xs = [int(r["n"]) for r in h_rows]
                ys = [flt(r["mean_t_extract"])*1000 for r in h_rows]
                ax.plot(xs, ys, label=heap, color=COLORS[heap],
                        marker=MARKERS[heap], linewidth=2)
            ax.set_title(GRAPH_SHORT.get(gt,gt), fontsize=11)
            ax.set_xlabel("n", fontsize=10)
            ax.set_ylabel("extract-min time (ms)" if gt==gts[0] else "", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(f"{algo} — extract-min cumulative time", fontsize=13, y=1.02)
        fig.tight_layout()
        fname = f"{algo}_extract_scaling.png"
        fig.savefig(f"{PLOT_DIR}/{fname}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ─────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────
print("Generating Part 4 metric plots…")
fig_time_breakdown()
fig_op_fractions()
fig_op_counts()
fig_relax_rate()
fig_memory()
fig_extract_scaling()
print(f"\nAll plots saved to {PLOT_DIR}/")
print(sorted(os.listdir(PLOT_DIR)))
