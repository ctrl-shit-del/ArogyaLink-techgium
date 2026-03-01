"""RAG service FastAPI app — POST /generate_brief, GET /search, GET /health."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any

# Optional: wire pipeline
try:
    from rag.pipeline.rag_pipeline import generate_brief
    from rag.pipeline.alert_context_builder import AlertContext, PatientSummary, VitalPoint
    from rag.retrieval.retriever import search
    from rag.retrieval.chroma_store import list_collections
except Exception as e:
    generate_brief = None
    AlertContext = None
    search = None
    list_collections = lambda: []


class PatientSummaryInput(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    ward: Optional[str] = None
    bed_number: Optional[str] = None
    diagnosed_conditions: List[dict] = []
    current_medications: List[dict] = []
    known_allergies: List[dict] = []
    genomic_risk_cardiac: str = "Unknown"
    genomic_risk_respiratory: str = "Unknown"
    genomic_risk_sepsis: str = "Unknown"
    last_clinical_notes: Optional[str] = None


class VitalPointInput(BaseModel):
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    motion_score: Optional[int] = None


class AlertContextInput(BaseModel):
    patient: PatientSummaryInput
    patient_age: int
    trigger_vital: str
    trigger_value: float
    baseline_value: float
    deviation_sigma: float
    second_derivative: float
    motion_score: int
    vitals_window: List[VitalPointInput]
    trigger_timestamp: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Synera RAG Service", version="2.0.0", lifespan=lifespan)


def _to_patient_summary(p: PatientSummaryInput) -> PatientSummary:
    return PatientSummary(
        patient_id=p.patient_id,
        name=p.name,
        age=p.age,
        gender=p.gender,
        blood_group=p.blood_group,
        ward=p.ward,
        bed_number=p.bed_number,
        diagnosed_conditions=p.diagnosed_conditions,
        current_medications=p.current_medications,
        known_allergies=p.known_allergies,
        genomic_risk_cardiac=p.genomic_risk_cardiac,
        genomic_risk_respiratory=p.genomic_risk_respiratory,
        genomic_risk_sepsis=p.genomic_risk_sepsis,
        last_clinical_notes=p.last_clinical_notes,
    )


def _to_vital_point(v: VitalPointInput) -> VitalPoint:
    return VitalPoint(
        heart_rate=v.heart_rate,
        spo2=v.spo2,
        temperature=v.temperature,
        motion_score=v.motion_score,
    )


@app.post("/api/v1/generate_brief")
async def api_generate_brief(body: AlertContextInput, alert_id: Optional[str] = None):
    """Generate ClinicalBrief from AlertContext. Used by backend on SYNERA_STATE."""
    if AlertContext is None or generate_brief is None:
        raise HTTPException(503, "RAG pipeline not available")
    ctx = AlertContext(
        patient=_to_patient_summary(body.patient),
        patient_age=body.patient_age,
        trigger_vital=body.trigger_vital,
        trigger_value=body.trigger_value,
        baseline_value=body.baseline_value,
        deviation_sigma=body.deviation_sigma,
        second_derivative=body.second_derivative,
        motion_score=body.motion_score,
        vitals_window=[_to_vital_point(v) for v in body.vitals_window],
        trigger_timestamp=body.trigger_timestamp,
    )
    brief = await generate_brief(ctx, alert_id=alert_id or "")
    return brief.model_dump() if hasattr(brief, "model_dump") else brief.dict()


@app.get("/api/v1/knowledge/chunks")
async def api_list_chunks(collection: Optional[str] = None):
    """List collections or chunks. For dev/chroma-check."""
    if list_collections is None:
        return {"collections": []}
    colls = list_collections()
    if collection:
        try:
            from rag.retrieval.chroma_store import get_or_create_collection
            c = get_or_create_collection(collection)
            res = c.get(include=["documents", "metadatas"])
            return {"collection": collection, "count": len(res["ids"]), "ids": res["ids"][:50]}
        except Exception as e:
            raise HTTPException(500, str(e))
    return {"collections": colls}


@app.get("/api/v1/rag/search")
async def api_search(q: str, top_k: int = 5):
    """Test ChromaDB retrieval."""
    if search is None:
        raise HTTPException(503, "Retriever not available")
    results = search("medical_knowledge", q, top_k=top_k)
    return [{"id": r.id, "text": r.text[:200], "metadata": r.metadata} for r in results]


@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag"}
