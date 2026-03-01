"""Patient CRUD via Supabase (replaces SQLAlchemy)."""
from typing import Optional
from backend.config.database import get_db


def get_patient(patient_id: str) -> Optional[dict]:
    try:
        response = get_db().table("patients").select("*").eq("patient_id", patient_id).single().execute()
        return response.data
    except Exception:
        return None


def list_patients(ward: Optional[str] = None) -> list[dict]:
    try:
        query = get_db().table("patients").select("*")
        if ward:
            query = query.eq("ward", ward)
        response = query.execute()
        return response.data or []
    except Exception:
        return []


def create_patient(patient: dict) -> dict:
    response = get_db().table("patients").insert(patient).execute()
    return response.data[0] if response.data else patient


def update_patient(patient_id: str, updates: dict) -> Optional[dict]:
    try:
        response = get_db().table("patients").update(updates).eq("patient_id", patient_id).execute()
        return response.data[0] if response.data else None
    except Exception:
        return None


def update_baseline(patient_id: str, baseline: dict) -> bool:
    from datetime import datetime
    try:
        get_db().table("patients").update({
            "baseline_hr_mean": baseline.get("hr_mean"),
            "baseline_hr_std": baseline.get("hr_std"),
            "baseline_spo2_mean": baseline.get("spo2_mean"),
            "baseline_spo2_std": baseline.get("spo2_std"),
            "baseline_temp_mean": baseline.get("temp_mean"),
            "baseline_last_updated": datetime.utcnow().isoformat(),
            "calibration_complete": True
        }).eq("patient_id", patient_id).execute()
        return True
    except Exception:
        return False


class PatientRepository:
    """Compatibility layer for verification scripts: list_all / get_by_id."""

    def __init__(self):
        pass

    async def list_all(self, ward: Optional[str] = None) -> list[dict]:
        return list_patients(ward=ward)

    async def get_by_id(self, patient_id: str) -> Optional[dict]:
        return get_patient(patient_id)
