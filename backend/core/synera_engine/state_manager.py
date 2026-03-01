"""STABLE / WATCH / SYNERA_STATE per patient."""
from enum import Enum
from typing import Dict


class PatientState(str, Enum):
    STABLE = "STABLE"
    WATCH = "WATCH"
    SYNERA_STATE = "SYNERA_STATE"


class StateManager:
    """In-memory patient state. Can be backed by Redis for multi-instance."""
    __slots__ = ("_state",)

    def __init__(self):
        self._state: Dict[str, PatientState] = {}

    def get(self, patient_id: str) -> PatientState:
        return self._state.get(patient_id, PatientState.STABLE)

    def set(self, patient_id: str, state: PatientState) -> None:
        self._state[patient_id] = state

    def set_watch(self, patient_id: str) -> None:
        self.set(patient_id, PatientState.WATCH)

    def set_synera(self, patient_id: str) -> None:
        self.set(patient_id, PatientState.SYNERA_STATE)

    def set_stable(self, patient_id: str) -> None:
        self.set(patient_id, PatientState.STABLE)


# Singleton for use by routes and engine
state_manager = StateManager()
