"""Generic ring buffer implementation."""
from collections import deque
from typing import TypeVar, Generic

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Fixed-capacity ring buffer. Oldest evicted when full."""
    __slots__ = ("_deque", "_maxlen")

    def __init__(self, maxlen: int):
        self._maxlen = maxlen
        self._deque: deque[T] = deque(maxlen=maxlen)

    def append(self, item: T) -> None:
        self._deque.append(item)

    def extend(self, items: list[T]) -> None:
        for item in items:
            self.append(item)

    def get_all(self) -> list[T]:
        return list(self._deque)

    def __len__(self) -> int:
        return len(self._deque)

    def __getitem__(self, i: int) -> T:
        return self._deque[i]

    @property
    def maxlen(self) -> int:
        return self._maxlen
