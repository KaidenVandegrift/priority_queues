"""
heap_interface.py
-----------------
Abstract base class defining the priority queue interface required by
Dijkstra's and Prim's algorithms. All heap implementations must subclass
this and implement every abstract method.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class HeapNode:
    """
    Returned by insert() so callers can hold a handle for decrease_key().
    Subclasses embed this (or use their own node type) as the internal node.
    """
    __slots__ = ("key", "value")

    def __init__(self, key: float, value: Any):
        self.key = key      # priority (distance / edge weight)
        self.value = value  # payload (vertex id, etc.)

    def __repr__(self):
        return f"HeapNode(key={self.key}, value={self.value})"


class AbstractHeap(ABC):
    """
    Interface every heap must satisfy.

    Operations
    ----------
    insert(key, value)  -> node handle   O(?)
    find_min()          -> node handle   O(?)
    extract_min()       -> node handle   O(?)
    decrease_key(node, new_key)          O(?)
    __len__()           -> int
    is_empty()          -> bool
    """

    @abstractmethod
    def insert(self, key: float, value: Any) -> HeapNode:
        """Insert (key, value) and return a handle for decrease_key."""

    @abstractmethod
    def find_min(self) -> Optional[HeapNode]:
        """Return (do NOT remove) the minimum-key node, or None if empty."""

    @abstractmethod
    def extract_min(self) -> Optional[HeapNode]:
        """Remove and return the minimum-key node, or None if empty."""

    @abstractmethod
    def decrease_key(self, node: HeapNode, new_key: float) -> None:
        """Lower node's key to new_key (new_key <= node.key required)."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of elements currently in the heap."""

    def is_empty(self) -> bool:
        return len(self) == 0
