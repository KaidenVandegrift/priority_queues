"""
test_correctness.py
--------------------
Unit tests verifying:
  1. All three heaps satisfy the heap contract (insert / extract / decrease_key)
  2. Dijkstra produces correct distances (vs brute-force Bellman-Ford)
  3. Prim produces a valid MST of minimum weight (vs Kruskal)

Run with:  python test_correctness.py
"""

import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))

from src.binary_heap    import BinaryHeap
from src.fibonacci_heap import FibonacciHeap
from src.pairing_heap   import PairingHeap
from src.graph          import Graph
from src.dijkstra       import dijkstra
from src.prim           import prim

HEAP_CLASSES = {
    "BinaryHeap":    BinaryHeap,
    "FibonacciHeap": FibonacciHeap,
    "PairingHeap":   PairingHeap,
}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
_errors = []


def check(cond: bool, msg: str) -> None:
    if cond:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        _errors.append(msg)


# -----------------------------------------------------------------------
# 1. Heap contract tests
# -----------------------------------------------------------------------

def test_heap_basic(name: str, Heap):
    print(f"\n[Heap: {name}] basic operations")

    h = Heap()
    check(h.is_empty(), "empty at start")
    check(len(h) == 0,  "len == 0 at start")

    # insert & find_min
    n3 = h.insert(3.0, "c")
    n1 = h.insert(1.0, "a")
    n2 = h.insert(2.0, "b")
    check(len(h) == 3,                   "len == 3 after 3 inserts")
    check(h.find_min().key == 1.0,       "find_min returns key=1")
    check(h.find_min().value == "a",     "find_min returns value='a'")

    # decrease_key
    h.decrease_key(n3, 0.5)
    check(h.find_min().key == 0.5,       "find_min after decrease_key(3→0.5)")

    # extract_min sequence
    m1 = h.extract_min()
    check(m1.key == 0.5,                 "extract 1st: key=0.5")
    m2 = h.extract_min()
    check(m2.key == 1.0,                 "extract 2nd: key=1.0")
    m3 = h.extract_min()
    check(m3.key == 2.0,                 "extract 3rd: key=2.0")
    check(h.is_empty(),                  "empty after extracting all")


def test_heap_random(name: str, Heap, n=500, seed=7):
    print(f"\n[Heap: {name}] random extract sequence (n={n})")
    rng = random.Random(seed)
    keys = [rng.uniform(0, 1000) for _ in range(n)]
    h = Heap()
    nodes = [h.insert(k, i) for i, k in enumerate(keys)]

    # decrease every other key
    for i in range(0, n, 2):
        new_k = keys[i] / 2
        h.decrease_key(nodes[i], new_k)
        keys[i] = new_k

    extracted = []
    while not h.is_empty():
        extracted.append(h.extract_min().key)

    sorted_keys = sorted(keys)
    ok = all(abs(a - b) < 1e-9 for a, b in zip(extracted, sorted_keys))
    check(ok, f"extract-min sequence matches sorted order (n={n})")


# -----------------------------------------------------------------------
# 2. Dijkstra correctness – compare against Bellman-Ford
# -----------------------------------------------------------------------

def bellman_ford(graph: Graph, source: int):
    """Reference shortest-path (slower but obviously correct)."""
    V = graph.num_vertices
    dist = [math.inf] * V
    dist[source] = 0.0
    for _ in range(V - 1):
        updated = False
        for u in range(V):
            if dist[u] == math.inf:
                continue
            for edge in graph.neighbors(u):
                v, w = edge.dst, edge.weight
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    updated = True
        if not updated:
            break
    return dist


def make_random_graph(n: int, p: float, seed: int) -> Graph:
    rng = random.Random(seed)
    g = Graph(n)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                w = rng.uniform(1, 50)
                g.add_edge(u, v, w)
    # Ensure connectivity with a path 0-1-2-…-(n-1)
    for u in range(n - 1):
        g.add_edge(u, u + 1, rng.uniform(1, 10))
    return g


def test_dijkstra(name: str, Heap):
    print(f"\n[Dijkstra / {name}] correctness")
    for n, p, seed in [(20, 0.3, 1), (50, 0.2, 2), (100, 0.15, 3)]:
        g = make_random_graph(n, p, seed)
        ref  = bellman_ford(g, 0)
        dist, _ = dijkstra(g, 0, Heap)
        ok = all(abs(a - b) < 1e-6 for a, b in zip(dist, ref))
        check(ok, f"matches Bellman-Ford (n={n}, p={p})")


# -----------------------------------------------------------------------
# 3. Prim correctness – weight matches Kruskal
# -----------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False
        self.p[rx] = ry
        return True


def kruskal(graph: Graph) -> float:
    edges = []
    for u in range(graph.n):
        for e in graph.neighbors(u):
            if e.dst > u:
                edges.append((e.weight, u, e.dst))
    edges.sort()
    uf = UnionFind(graph.n)
    total = 0.0
    for w, u, v in edges:
        if uf.union(u, v):
            total += w
    return total


def test_prim(name: str, Heap):
    print(f"\n[Prim / {name}] MST weight matches Kruskal")
    for n, p, seed in [(20, 0.4, 10), (50, 0.3, 11), (100, 0.2, 12)]:
        g = make_random_graph(n, p, seed)
        ref_w = kruskal(g)
        _, prim_w = prim(g, 0, Heap)
        ok = abs(prim_w - ref_w) < 1e-6
        check(ok, f"Prim weight={prim_w:.4f} == Kruskal={ref_w:.4f} (n={n})")


# -----------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTNESS TEST SUITE")
    print("=" * 60)

    for name, Heap in HEAP_CLASSES.items():
        test_heap_basic(name, Heap)
        test_heap_random(name, Heap)
        test_dijkstra(name, Heap)
        test_prim(name, Heap)

    print("\n" + "=" * 60)
    if _errors:
        print(f"FAILED: {len(_errors)} test(s)")
        for e in _errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"ALL TESTS PASSED")
    print("=" * 60)
