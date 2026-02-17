"""
fibonacci_heap.py
-----------------
Fibonacci Heap - amortised-optimal priority queue.

Amortised complexities
----------------------
insert       : O(1)
find_min     : O(1)
extract_min  : O(log n)   amortised
decrease_key : O(1)       amortised

Reference: Fredman & Tarjan (1987), JACM 34(3).
"""

from __future__ import annotations
from typing import Any, Optional
import math
from .heap_interface import AbstractHeap, HeapNode


class FibNode(HeapNode):
    __slots__ = ("key", "value", "degree", "mark", "parent", "child", "left", "right")

    def __init__(self, key: float, value: Any):
        super().__init__(key, value)
        self.degree: int = 0
        self.mark: bool = False
        self.parent: Optional[FibNode] = None
        self.child: Optional[FibNode] = None
        self.left: FibNode = self
        self.right: FibNode = self


def _dll_remove(x: FibNode) -> None:
    """Remove x from its doubly-linked circular list."""
    x.left.right = x.right
    x.right.left = x.left
    x.left = x
    x.right = x


def _dll_concat(a: Optional[FibNode], b: Optional[FibNode]) -> Optional[FibNode]:
    """Concatenate two circular DLLs; return node with smaller key."""
    if a is None:
        return b
    if b is None:
        return a
    a_right = a.right
    b_left  = b.left
    a.right      = b
    b.left       = a
    b_left.right = a_right
    a_right.left = b_left
    return a if a.key <= b.key else b


def _dll_to_list(start: Optional[FibNode]) -> list[FibNode]:
    if start is None:
        return []
    result = []
    cur = start
    while True:
        result.append(cur)
        cur = cur.right
        if cur is start:
            break
    return result


class FibonacciHeap(AbstractHeap):
    """Fibonacci min-heap."""

    def __init__(self):
        self._min: Optional[FibNode] = None
        self._n: int = 0

    def insert(self, key: float, value: Any) -> FibNode:
        node = FibNode(key, value)
        self._min = _dll_concat(self._min, node)
        self._n += 1
        return node

    def find_min(self) -> Optional[FibNode]:
        return self._min

    def extract_min(self) -> Optional[FibNode]:
        z = self._min
        if z is None:
            return None

        if z.child is not None:
            children = _dll_to_list(z.child)
            for c in children:
                c.parent = None
            self._min = _dll_concat(self._min, z.child)
            z.child = None

        if z.right is z:
            self._min = None
        else:
            self._min = z.right
            _dll_remove(z)
            self._consolidate()

        self._n -= 1
        z.left = z
        z.right = z
        return z

    def decrease_key(self, node: FibNode, new_key: float) -> None:
        if new_key > node.key:
            raise ValueError("decrease_key: new_key must be <= current key")
        node.key = new_key
        parent = node.parent
        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        if self._min is not None and node.key < self._min.key:
            self._min = node

    def __len__(self) -> int:
        return self._n

    def _consolidate(self) -> None:
        max_deg = int(math.log2(self._n + 1)) + 2 if self._n > 0 else 2
        A: list[Optional[FibNode]] = [None] * (max_deg + 1)

        roots = _dll_to_list(self._min)
        for r in roots:
            r.left = r
            r.right = r

        for x in roots:
            d = x.degree
            while True:
                if d >= len(A):
                    A.extend([None] * (d - len(A) + 1))
                y = A[d]
                if y is None:
                    break
                if x.key > y.key:
                    x, y = y, x
                self._heap_link(y, x)
                A[d] = None
                d += 1
            A[d] = x

        self._min = None
        for node in A:
            if node is None:
                continue
            node.left = node
            node.right = node
            self._min = _dll_concat(self._min, node)

    def _heap_link(self, y: FibNode, x: FibNode) -> None:
        """Make y a child of x (x.key <= y.key)."""
        y.parent = x
        if x.child is None:
            x.child = y
            y.left = y
            y.right = y
        else:
            y.right = x.child.right
            y.left  = x.child
            x.child.right.left = y
            x.child.right      = y
        x.degree += 1
        y.mark = False

    def _cut(self, x: FibNode, y: FibNode) -> None:
        if x.right is x:
            y.child = None
        else:
            if y.child is x:
                y.child = x.right
            _dll_remove(x)
        y.degree -= 1
        x.parent = None
        x.mark   = False
        x.left = x
        x.right = x
        self._min = _dll_concat(self._min, x)

    def _cascading_cut(self, y: FibNode) -> None:
        z = y.parent
        if z is not None:
            if not y.mark:
                y.mark = True
            else:
                self._cut(y, z)
                self._cascading_cut(z)
