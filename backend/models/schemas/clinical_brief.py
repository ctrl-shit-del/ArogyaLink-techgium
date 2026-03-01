"""ClinicalBrief output schema — RAG pipeline produces this."""
from typing import Literal
from pydantic import BaseModel


class DifferentialItem(BaseModel):
    condition: str
    likelihood: Literal["Most Likely", "Possible", "Rule Out"]
    reasoning: str


class RecommendedAction(BaseModel):
    priority: int  # 1=immediate, 2=urgent, 3=when possible
    action: str
    rationale: str


class DrugInteractionFlag(BaseModel):
    medication: str
    flag: str
    severity: Literal["Critical", "Warning", "Info"]


class Source(BaseModel):
    document: str
    section: str
    relevance: str


class ClinicalBrief(BaseModel):
    alert_id: str
    patient_id: str
    generated_at: str  # ISO-8601
    trigger_summary: str
    differential_diagnosis: list[DifferentialItem]
    recommended_actions: list[RecommendedAction]
    drug_interaction_flags: list[DrugInteractionFlag]
    relevant_history: list[str]
    sources: list[Source]
    confidence_note: str
    generation_time_ms: int
