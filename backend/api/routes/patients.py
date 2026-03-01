"""CRUD for MedID records (Supabase)."""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.database.patient_repo import get_patient, list_patients, create_patient, update_patient

router = APIRouter(prefix="/patients", tags=["patients"])


class PatientCreate(BaseModel):
    patient_id: str
    name: str
    ward: Optional[str] = None
    bed_number: Optional[str] = None


@router.post("/")
async def create_patient_endpoint(body: PatientCreate):
    existing = get_patient(body.patient_id)
    if existing:
        raise HTTPException(400, "Patient already exists")
    p = create_patient({"patient_id": body.patient_id, "name": body.name, "ward": body.ward, "bed_number": body.bed_number})
    return p


@router.get("/")
async def list_patients_endpoint(ward: Optional[str] = None):
    patients = list_patients(ward=ward)
    return [{"patient_id": p["patient_id"], "name": p["name"], "ward": p.get("ward"), "bed_number": p.get("bed_number")} for p in patients]


@router.get("/{patient_id}")
async def get_patient_endpoint(patient_id: str):
    p = get_patient(patient_id)
    if not p:
        raise HTTPException(404, "Patient not found")
    return {
        "patient_id": p["patient_id"],
        "name": p["name"],
        "dob": str(p.get("dob")) if p.get("dob") else None,
        "gender": p.get("gender"),
        "ward": p.get("ward"),
        "bed_number": p.get("bed_number"),
        "diagnosed_conditions": p.get("diagnosed_conditions", []),
        "current_medications": p.get("current_medications", []),
        "known_allergies": p.get("known_allergies", []),
        "baseline_hr_mean": p.get("baseline_hr_mean"),
        "baseline_spo2_mean": p.get("baseline_spo2_mean"),
    }


@router.put("/{patient_id}")
async def update_patient_endpoint(patient_id: str, body: dict):
    p = update_patient(patient_id, body)
    if not p:
        raise HTTPException(404, "Patient not found")
    return {"ok": True}
