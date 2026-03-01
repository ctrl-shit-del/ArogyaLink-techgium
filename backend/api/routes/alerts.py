"""Alert history + acknowledge (Supabase)."""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.database.alert_repo import get_alert, list_alerts, acknowledge_alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("/")
async def list_alerts_endpoint(patient_id: Optional[str] = None, limit: int = 50):
    alerts = list_alerts(patient_id=patient_id, limit=limit)
    return [
        {
            "alert_id": a["alert_id"],
            "patient_id": a["patient_id"],
            "trigger_timestamp": str(a.get("trigger_timestamp")),
            "trigger_vital": a.get("trigger_vital"),
            "trigger_value": a.get("trigger_value"),
            "clinician_acknowledged": a.get("clinician_acknowledged", False),
        }
        for a in alerts
    ]


@router.get("/{alert_id}")
async def get_alert_endpoint(alert_id: str):
    a = get_alert(alert_id)
    if not a:
        raise HTTPException(404, "Alert not found")
    return {
        "alert_id": a["alert_id"],
        "patient_id": a["patient_id"],
        "trigger_timestamp": str(a.get("trigger_timestamp")),
        "trigger_vital": a.get("trigger_vital"),
        "trigger_value": a.get("trigger_value"),
        "rag_clinical_brief": a.get("rag_clinical_brief"),
        "vitals_snapshot": a.get("vitals_snapshot"),
    }


class AcknowledgeBody(BaseModel):
    priority_tier: Optional[str] = None
    response_time_minutes: Optional[float] = None


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert_endpoint(alert_id: str, body: AcknowledgeBody):
    a = acknowledge_alert(alert_id, priority_tier=body.priority_tier, response_time_minutes=body.response_time_minutes)
    if not a:
        raise HTTPException(404, "Alert not found")
    return {"ok": True}
