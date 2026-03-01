"""Vitals history (paginated) — Supabase."""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter

from backend.services.database.vitals_repo import get_vitals_history
from backend.core.synera_engine.state_manager import state_manager

router = APIRouter(prefix="/patients", tags=["vitals"])


@router.get("/{patient_id}/vitals")
async def get_vitals_endpoint(
    patient_id: str,
    page: int = 1,
    limit: int = 50,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
):
    vitals = get_vitals_history(patient_id=patient_id, page=page, limit=limit, from_ts=from_ts, to_ts=to_ts)
    return [
        {
            "recorded_at": str(v.get("recorded_at")),
            "heart_rate": v.get("heart_rate"),
            "spo2": v.get("spo2"),
            "temperature": v.get("temperature"),
            "motion_score": v.get("motion_score"),
        }
        for v in vitals
    ]


@router.get("/{patient_id}/state")
async def get_state_endpoint(patient_id: str):
    state = state_manager.get(patient_id)
    return {"patient_id": patient_id, "state": state.value}
