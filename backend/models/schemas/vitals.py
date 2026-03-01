"""Pydantic schemas for vitals payload (MQTT) and history."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class VitalPayload(BaseModel):
    """Single MQTT vitals message from wearable."""
    patient_id: str
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    sys_bp_est: Optional[float] = None
    dia_bp_est: Optional[float] = None
    motion_score: Optional[int] = None
    is_active: Optional[bool] = None
    battery_pct: Optional[int] = None
    reconstruction_error: Optional[float] = None
    recorded_at: Optional[datetime] = None
