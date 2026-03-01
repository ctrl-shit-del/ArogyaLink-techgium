"""Master RAG pipeline: retrieve → rerank → prompt → LLM → parse. Async with timeout."""
import asyncio
from datetime import datetime
from typing import Union

from rag.pipeline.alert_context_builder import AlertContext, PatientSummary, VitalPoint
from rag.pipeline.clinical_brief_generator import generate_brief_sync, rule_based_fallback_brief

# Re-export schema for callers
try:
    from backend.models.schemas.clinical_brief import ClinicalBrief
except ImportError:
    from pydantic import BaseModel
    class ClinicalBrief(BaseModel):
        alert_id: str
        patient_id: str
        generated_at: str
        trigger_summary: str
        differential_diagnosis: list
        recommended_actions: list
        drug_interaction_flags: list
        relevant_history: list
        sources: list
        confidence_note: str
        generation_time_ms: int


def _dict_to_alert_context(context: dict) -> AlertContext:
    """Build AlertContext from verification-style dict (patient, vitals_window, etc.)."""
    p = context.get("patient") or {}
    ps = PatientSummary(
        patient_id=p.get("patient_id", context.get("patient_id", "")),
        name=p.get("name", ""),
        age=p.get("age", 40),
        gender=p.get("gender"),
        blood_group=p.get("blood_group"),
        ward=p.get("ward"),
        bed_number=p.get("bed_number"),
        diagnosed_conditions=p.get("diagnosed_conditions", []),
        current_medications=p.get("current_medications", []),
        known_allergies=p.get("known_allergies", []),
        genomic_risk_cardiac=p.get("genomic_risk_cardiac", "Unknown"),
        genomic_risk_respiratory=p.get("genomic_risk_respiratory", "Unknown"),
        genomic_risk_sepsis=p.get("genomic_risk_sepsis", "Unknown"),
        last_clinical_notes=p.get("last_clinical_notes"),
    )
    vitals_window = [
        VitalPoint(
            heart_rate=v.get("heart_rate"),
            spo2=v.get("spo2"),
            temperature=v.get("temperature"),
            motion_score=v.get("motion_score"),
        )
        for v in context.get("vitals_window", [])
    ]
    return AlertContext(
        patient=ps,
        patient_age=context.get("patient_age", p.get("age", 40)),
        trigger_vital=context.get("trigger_vital", "heart_rate"),
        trigger_value=float(context.get("trigger_value", 0)),
        baseline_value=float(context.get("baseline_value", 0)),
        deviation_sigma=float(context.get("deviation_sigma", 0)),
        second_derivative=float(context.get("second_derivative", 0)),
        motion_score=int(context.get("motion_score", 0)),
        vitals_window=vitals_window,
        trigger_timestamp=context.get("trigger_timestamp"),
    )


async def generate_brief(
    alert_context: Union[AlertContext, dict],
    alert_id: str = "",
    timeout_seconds: float = 3.0,
):
    """Async generate with timeout. Accepts AlertContext or dict (converted to AlertContext)."""
    if isinstance(alert_context, dict):
        alert_context = _dict_to_alert_context(alert_context)
        alert_id = alert_id or alert_context.patient.patient_id
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(generate_brief_sync, alert_context, alert_id),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        return rule_based_fallback_brief(alert_context, alert_id)
