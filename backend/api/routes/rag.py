"""Manual RAG trigger and vector search (in-process, Supabase pgvector)."""
from fastapi import APIRouter
from rag.retrieval.retriever import search

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/trigger")
async def trigger_rag(body: dict):
    """Manually trigger RAG with mock AlertContext (for testing). Build AlertContext and call generate_brief_sync."""
    try:
        from rag.pipeline.alert_context_builder import AlertContext, PatientSummary, VitalPoint
        from rag.pipeline.clinical_brief_generator import generate_brief_sync
        p = body.get("patient", {})
        ps = PatientSummary(
            patient_id=p.get("patient_id", ""),
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
        vitals_window = [VitalPoint(heart_rate=v.get("heart_rate"), spo2=v.get("spo2"), temperature=v.get("temperature"), motion_score=v.get("motion_score")) for v in body.get("vitals_window", [])]
        ctx = AlertContext(
            patient=ps,
            patient_age=body.get("patient_age", 40),
            trigger_vital=body.get("trigger_vital", "heart_rate"),
            trigger_value=float(body.get("trigger_value", 0)),
            baseline_value=float(body.get("baseline_value", 0)),
            deviation_sigma=float(body.get("deviation_sigma", 0)),
            second_derivative=float(body.get("second_derivative", 0)),
            motion_score=int(body.get("motion_score", 0)),
            vitals_window=vitals_window,
            trigger_timestamp=body.get("trigger_timestamp"),
        )
        brief = generate_brief_sync(ctx, alert_id=body.get("alert_id", ""))
        return brief.model_dump() if hasattr(brief, "model_dump") else brief.dict()
    except Exception as e:
        return {"error": str(e)}


@router.get("/search")
async def rag_search(q: str, top_k: int = 5):
    """Test vector retrieval (Supabase pgvector)."""
    chunks = search("medical_knowledge", q, top_k=top_k)
    return [{"id": c.id, "text": c.text[:200], "metadata": c.metadata} for c in chunks]
