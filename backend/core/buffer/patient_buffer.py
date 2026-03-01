"""In-memory dict[patient_id → WindowBuffer]. Redis sync can be added later."""
from typing import Dict

from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading


class PatientBuffer:
    """Per-patient sliding windows. Thread-safe not required for single-thread async."""
    __slots__ = ("_buffers", "_window_size")

    def __init__(self, window_size: int = 10):
        self._window_size = window_size
        self._buffers: Dict[str, WindowBuffer] = {}

    def get_or_create(self, patient_id: str) -> WindowBuffer:
        if patient_id not in self._buffers:
            self._buffers[patient_id] = WindowBuffer(maxlen=self._window_size)
        return self._buffers[patient_id]

    def append(self, patient_id: str, reading: VitalReading) -> None:
        self.get_or_create(patient_id).append(reading)

    def get_buffer(self, patient_id: str) -> WindowBuffer | None:
        return self._buffers.get(patient_id)

    def get_all_patient_ids(self) -> list[str]:
        return list(self._buffers.keys())

    def clear_patient(self, patient_id: str) -> None:
        if patient_id in self._buffers:
            del self._buffers[patient_id]
