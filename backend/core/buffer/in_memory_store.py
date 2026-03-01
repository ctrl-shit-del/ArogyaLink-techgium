"""
Replaces Redis entirely.
For a hackathon with 5 simulated patients, in-memory is correct.
Thread-safe using asyncio locks.
"""
import asyncio
from collections import deque
from typing import Any

class InMemoryStore:
    """
    Replaces Redis for:
    1. Patient sliding window buffers (10 readings each)
    2. Patient state cache (STABLE / WATCH / SYNERA_STATE)
    3. Patient baseline cache (pulled from Supabase, cached 5 min)
    """
    def __init__(self):
        self._buffers: dict[str, deque] = {}
        self._states: dict[str, str] = {}
        self._baselines: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def get_buffer(self, patient_id: str, maxlen: int = 10) -> deque:
        async with self._lock:
            if patient_id not in self._buffers:
                self._buffers[patient_id] = deque(maxlen=maxlen)
            return self._buffers[patient_id]

    async def append_to_buffer(self, patient_id: str, reading: dict) -> deque:
        async with self._lock:
            if patient_id not in self._buffers:
                self._buffers[patient_id] = deque(maxlen=10)
            self._buffers[patient_id].append(reading)
            return self._buffers[patient_id]

    async def get_state(self, patient_id: str) -> str:
        return self._states.get(patient_id, "STABLE")

    async def set_state(self, patient_id: str, state: str):
        self._states[patient_id] = state

    async def get_baseline(self, patient_id: str) -> dict | None:
        return self._baselines.get(patient_id)

    async def set_baseline(self, patient_id: str, baseline: dict):
        self._baselines[patient_id] = baseline

    async def get_all_states(self) -> dict[str, str]:
        return dict(self._states)


# Singleton instance — imported by engine.py
store = InMemoryStore()
