"""Vitals history writes + reads via Supabase."""
from datetime import datetime
from typing import Optional
from backend.config.database import get_db


def insert_vital(patient_id: str, **kwargs) -> dict:
    row = {"patient_id": patient_id, **kwargs}
    response = get_db().table("vitals_history").insert(row).execute()
    return response.data[0] if response.data else row


def get_vitals_history(
    patient_id: str,
    page: int = 1,
    limit: int = 50,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
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
