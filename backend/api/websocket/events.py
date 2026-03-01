"""WebSocket event schema definitions."""
from typing import Any, Optional
from pydantic import BaseModel


class VitalsSnapshot(BaseModel):
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    sys_bp_est: Optional[float] = None
    dia_bp_est: Optional[float] = None
    motion_score: Optional[int] = None


class SyneraStateEvent(BaseModel):
    event_type: str = "SYNERA_STATE"
    alert_id: str
    patient_id: str
    priority_tier: str = "TIER_1"
    trigger_timestamp: str
    trigger_summary: str
    vitals_snapshot: dict
    clinical_brief: dict
    requires_acknowledgement: bool = True


class StateChangeEvent(BaseModel):
    event_type: str = "STATE_CHANGE"
    patient_id: str
    new_state: str
    previous_state: str
    reason: str
    timestamp: str


class ExertionLoggedEvent(BaseModel):
    event_type: str = "EXERTION_LOGGED"
    patient_id: str
    timestamp: str
    motion_score: int
    hr_elevation: float
