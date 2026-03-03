"""Vitals history writes + reads via Supabase."""
import datetime
from typing import Optional
from backend.config.database import get_db


def insert_vital(
    patient_id,
    heart_rate,
    spo2,
    temperature,
    motion_score,
    recorded_at=None,
    reconstruction_error: float = None,
    pre_alert: bool = False,
    **kwargs,
):
    # Convert datetime to ISO string if needed
    if isinstance(recorded_at, datetime.datetime):
        recorded_at = recorded_at.isoformat()
    if recorded_at is None:
        recorded_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    row = {
        "patient_id": patient_id,
        "heart_rate": heart_rate,
        "spo2": spo2,
        "temperature": temperature,
        "motion_score": motion_score,
        "recorded_at": recorded_at,
        # TinyML anomaly fields — written explicitly so they are never silently dropped
        "reconstruction_error": reconstruction_error,
        "pre_alert": pre_alert,
        **kwargs,
    }
    response = get_db().table("vitals_history").insert(row).execute()
    return response.data[0] if response.data else row


def get_vitals_history(
    patient_id: str,
    page: int = 1,
    limit: int = 50,
    from_ts: Optional[datetime.datetime] = None,
    to_ts: Optional[datetime.datetime] = None,
) -> list[dict]:
    try:
        query = get_db().table("vitals_history").select("*").eq("patient_id", patient_id).order("recorded_at", desc=True).range((page - 1) * limit, page * limit - 1)
        if from_ts:
            query = query.gte("recorded_at", from_ts.isoformat())
        if to_ts:
            query = query.lte("recorded_at", to_ts.isoformat())
        response = query.execute()
        return response.data or []
    except Exception:
        return []
