"""Alert events CRUD via Supabase."""
from typing import Optional
from datetime import datetime
from backend.config.database import get_db


def create_alert(**kwargs) -> dict:
    response = get_db().table("alert_events").insert(kwargs).execute()
    return response.data[0] if response.data else kwargs


def get_alert(alert_id: str) -> Optional[dict]:
    try:
        response = get_db().table("alert_events").select("*").eq("alert_id", alert_id).single().execute()
        return response.data
    except Exception:
        return None


def acknowledge_alert(
    alert_id: str,
    priority_tier: Optional[str] = None,
    response_time_minutes: Optional[float] = None,
) -> Optional[dict]:
    try:
        updates = {
            "clinician_acknowledged": True,
            "acknowledgement_timestamp": datetime.utcnow().isoformat(),
        }
        if priority_tier:
            updates["priority_tier_assigned"] = priority_tier
        if response_time_minutes is not None:
            updates["response_time_minutes"] = response_time_minutes
        response = get_db().table("alert_events").update(updates).eq("alert_id", alert_id).execute()
        return response.data[0] if response.data else None
    except Exception:
        return None


def list_alerts(patient_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    try:
        query = get_db().table("alert_events").select("*").order("created_at", desc=True).limit(limit)
        if patient_id:
            query = query.eq("patient_id", patient_id)
        response = query.execute()
        return response.data or []
    except Exception:
        return []
