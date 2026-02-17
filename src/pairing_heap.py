"""
pairing_heap.py
---------------
Pairing Heap — simple, practically fast priority queue.

Complexities (amortised)
------------------------
insert       : O(1)
find_min     : O(1)
extract_min  : O(log n)   amortised
decrease_key : O(log n)   amortised  (exact bound still open; behaves as O(1) in practice)

Reference: Fredman, Sedgewick, Sleator, Tarjan (1986), "The pairing heap:
a new form of self-adjusting heap", Algorithmica 1(1).

Implementation notes
--------------------
The heap is a rooted tree stored using the *leftmost-child, right-sibling*
(LC-RS) representation:
  - node.child  : leftmost (first) child
  - node.sibling: next sibling to the right (under the same parent)
  - node.prev   : previous sibling or parent pointer (needed for decrease_key cuts)

For decrease_key we cut the subtree rooted at the node, decrease its key,
and meld it back into the root.

Meld (two-pass): on extract_min, children are merged left-to-right in pairs
then the resulting pairs are merged right-to-left.
"""

from __future__ import annotations
from typing import Any, Optional
from .heap_interface import AbstractHeap, HeapNode


class PairingNode(HeapNode):
    """Node in a pairing heap."""
    __slots__ = ("key", "value", "child", "sibling", "prev")

    def __init__(self, key: float, value: Any):
        super().__init__(key, value)
        self.child:   Optional[PairingNode] = None
        self.sibling: Optional[PairingNode] = None
        self.prev:    Optional[PairingNode] = None   # parent or left sibling


class PairingHeap(AbstractHeap):
    """Pairing min-heap."""

    def __init__(self):
        self._root: Optional[PairingNode] = None
        self._n: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def insert(self, key: float, value: Any) -> PairingNode:
        node = PairingNode(key, value)
        self._root = self._meld(self._root, node)
        self._n += 1
        return node

    def find_min(self) -> Optional[PairingNode]:
        return self._root

    def extract_min(self) -> Optional[PairingNode]:
        if self._root is None:
            return None
        old_root = self._root
        self._root = self._merge_pairs(old_root.child)
        if self._root is not None:
            self._root.prev = None
        self._n -= 1
        # Clean up extracted node's links
        old_root.child = None
        old_root.sibling = None
        old_root.prev = None
        return old_root

    def decrease_key(self, node: PairingNode, new_key: float) -> None:
        if new_key > node.key:
            raise ValueError("decrease_key: new_key must be <= current key")
        node.key = new_key
        if node is self._root:
            return   # already at root, nothing to do

        # Detach node from its position in the sibling list / child list
        if node.prev is not None:
            if node.prev.child is node:
                # node is leftmost child of node.prev
                node.prev.child = node.sibling
            else:
                # node.prev is the left sibling
                node.prev.sibling = node.sibling
        if node.sibling is not None:
            node.sibling.prev = node.prev

        node.sibling = None
        node.prev = None

        # Meld the detached subtree with the root
        self._root = self._meld(self._root, node)

    def __len__(self) -> int:
        return self._n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _meld(a: Optional[PairingNode],
              b: Optional[PairingNode]) -> Optional[PairingNode]:
        """Return the root after melding two heap-ordered trees."""
        if a is None:
            return b
        if b is None:
            return a
        # Smaller key becomes root; larger key becomes its leftmost child
        if a.key <= b.key:
            parent, child = a, b
        else:
            parent, child = b, a
        # Prepend child to parent's child list
        child.sibling = parent.child
        child.prev = parent
        if parent.child is not None:
            parent.child.prev = child
        parent.child = child
        parent.sibling = None
        parent.prev = None
        return parent

    @staticmethod
    def _merge_pairs(first: Optional[PairingNode]) -> Optional[PairingNode]:
        """Two-pass pairing of the child list after extract_min."""
        if first is None or first.sibling is None:
            if first is not None:
                first.prev = None
            return first

        # Collect children into a list (detach links as we go)
        children: list[PairingNode] = []
        cur = first
        while cur is not None:
            nxt = cur.sibling
            cur.sibling = None
            cur.prev = None
            children.append(cur)
            cur = nxt

        # Forward pass: pair adjacent children
        pairs: list[Optional[PairingNode]] = []
        i = 0
        while i < len(children) - 1:
            pairs.append(PairingHeap._meld(children[i], children[i + 1]))
            i += 2
        if i == len(children) - 1:
            pairs.append(children[i])

        # Backward pass: fold pairs right-to-left
        result = pairs[-1]
        for j in range(len(pairs) - 2, -1, -1):
            result = PairingHeap._meld(pairs[j], result)
        return result
