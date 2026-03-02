"""Vitals history (paginated) — Supabase. Ingest endpoint for simulator/wearable."""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter

from backend.services.database.vitals_repo import get_vitals_history
from backend.core.synera_engine.state_manager import state_manager
from backend.core.synera_engine.engine import engine

router = APIRouter(prefix="/patients", tags=["vitals"])
router_ingest = APIRouter(prefix="/vitals", tags=["vitals"])


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


@router_ingest.post("/ingest")
async def ingest_vital(payload: dict):
    """
    Receives a vital reading from the simulator or wearable.
    Runs through the Synera 3-rule pipeline.
    Returns the rule result so the simulator can log it.

    In production: ESP32 → MQTT → this endpoint (via MQTT subscriber)
    In development: mock_simulator.py → HTTP POST → this endpoint
    """
    patient_id = payload.get("patient_id")
    vitals = payload.get("vitals", payload)
    context = payload.get("context", {})
    tinyml = payload.get("tinyml", {})

    vital_payload = {
        "patient_id": patient_id,
        "heart_rate": vitals.get("heart_rate"),
        "spo2": vitals.get("spo2"),
        "temperature": vitals.get("temperature"),
        "sys_bp_est": vitals.get("sys_bp_est"),
        "dia_bp_est": vitals.get("dia_bp_est"),
        "motion_score": context.get("motion_score", vitals.get("motion_score")),
        "battery_pct": context.get("battery_pct", 100),
        "recorded_at": payload.get("recorded_at", vitals.get("recorded_at")),
        "pre_alert": tinyml.get("pre_alert", False),
    }

    result = await engine.process_vital(patient_id, vital_payload)

    return {
        "patient_id": patient_id,
        "state": result.state,
        "rule": result.rule,
        "message": result.message,
    }
