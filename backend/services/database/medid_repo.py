"""MedID search + update via Supabase (uses patients table)."""
from backend.services.database.patient_repo import get_patient


def search_medid(patient_id: str) -> list[dict]:
    p = get_patient(patient_id)
    if not p:
        return []
    return [{"patient_id": p["patient_id"], "name": p["name"], "ward": p.get("ward"), "bed_number": p.get("bed_number")}]
