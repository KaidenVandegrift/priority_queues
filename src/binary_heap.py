"""
binary_heap.py
--------------
Standard binary min-heap (array-based) used as a performance baseline.

Complexities
------------
insert       : O(log n)
find_min     : O(1)
extract_min  : O(log n)
decrease_key : O(log n)  -- requires position lookup via index map

The node handle returned by insert() is a BinaryHeapNode. The heap
maintains a pos[] map from node-id to array position so decrease_key
can locate and bubble up the entry in O(log n).
"""

from __future__ import annotations
from typing import Any, Optional
from .heap_interface import AbstractHeap, HeapNode


class BinaryHeapNode(HeapNode):
    __slots__ = ("key", "value", "_id")

    def __init__(self, key: float, value: Any, node_id: int):
        super().__init__(key, value)
        self._id = node_id   # stable identifier, never changes


class BinaryHeap(AbstractHeap):
    """
    Array-backed binary min-heap with O(log n) decrease-key via index map.
    """

    def __init__(self):
        self._heap: list[BinaryHeapNode] = []  # heap[0] = min
        self._pos: dict[int, int] = {}          # node._id -> index in _heap
        self._counter = 0                        # monotone id generator

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert(self, key: float, value: Any) -> BinaryHeapNode:
        node = BinaryHeapNode(key, value, self._counter)
        self._counter += 1
        idx = len(self._heap)
        self._heap.append(node)
        self._pos[node._id] = idx
        self._sift_up(idx)
        return node

    def find_min(self) -> Optional[BinaryHeapNode]:
        return self._heap[0] if self._heap else None

    def extract_min(self) -> Optional[BinaryHeapNode]:
        if not self._heap:
            return None
        self._swap(0, len(self._heap) - 1)
        node = self._heap.pop()
        del self._pos[node._id]
        if self._heap:
            self._sift_down(0)
        return node

    def decrease_key(self, node: BinaryHeapNode, new_key: float) -> None:
        if new_key > node.key:
            raise ValueError("decrease_key: new_key must be <= current key")
        node.key = new_key
        idx = self._pos[node._id]
        self._sift_up(idx)

    def __len__(self) -> int:
        return len(self._heap)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sift_up(self, i: int) -> None:
        while i > 0:
            parent = (i - 1) >> 1
            if self._heap[parent].key > self._heap[i].key:
                self._swap(parent, i)
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        n = len(self._heap)
        while True:
            smallest = i
            left  = 2 * i + 1
            right = 2 * i + 2
            if left < n and self._heap[left].key < self._heap[smallest].key:
                smallest = left
            if right < n and self._heap[right].key < self._heap[smallest].key:
                smallest = right
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._pos[self._heap[i]._id] = i
        self._pos[self._heap[j]._id] = j
