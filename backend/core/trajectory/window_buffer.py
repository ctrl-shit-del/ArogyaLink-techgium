"""10-reading sliding window per patient (deque)."""
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class VitalReading:
    """Single vital snapshot from wearable."""
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    sys_bp_est: Optional[float] = None
    dia_bp_est: Optional[float] = None
    motion_score: Optional[int] = None
    recorded_at: Optional[float] = None  # timestamp


WINDOW_SIZE = 10


class WindowBuffer:
    """Fixed-size sliding window of VitalReading. Oldest dropped when full."""
    __slots__ = ("_deque",)

    def __init__(self, maxlen: int = WINDOW_SIZE):
        self._deque: deque[VitalReading] = deque(maxlen=maxlen)

    def append(self, reading: VitalReading) -> None:
        self._deque.append(reading)

    def get_readings(self) -> list[VitalReading]:
        return list(self._deque)

    def get_heart_rates(self) -> list[float]:
        return [r.heart_rate for r in self._deque if r.heart_rate is not None]

    def get_spo2(self) -> list[float]:
        return [r.spo2 for r in self._deque if r.spo2 is not None]

    def get_temperatures(self) -> list[float]:
        return [r.temperature for r in self._deque if r.temperature is not None]

    def last_reading(self) -> Optional[VitalReading]:
        return self._deque[-1] if self._deque else None

    def __len__(self) -> int:
        return len(self._deque)
