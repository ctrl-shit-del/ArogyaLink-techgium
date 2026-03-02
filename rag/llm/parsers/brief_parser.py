"""JSON extraction + Pydantic validation for ClinicalBrief."""
import json
import re
from typing import Optional

# Use backend schema if available
try:
    from backend.models.schemas.clinical_brief import ClinicalBrief
except ImportError:
    from pydantic import BaseModel
    from typing import Literal
    class DifferentialItem(BaseModel):
        condition: str
        likelihood: Literal["Most Likely", "Possible", "Rule Out"]
        reasoning: str
    class RecommendedAction(BaseModel):
        priority: int
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
        generated_at: str
        trigger_summary: str
        differential_diagnosis: list
        recommended_actions: list
        drug_interaction_flags: list
        relevant_history: list
        sources: list
        confidence_note: str
        generation_time_ms: int


def parse(raw_llm_output: str, alert_id: str = "", patient_id: str = "") -> Optional[ClinicalBrief]:
    """Extract JSON from LLM output and validate to ClinicalBrief."""
    text = raw_llm_output.strip()
    # Strip markdown code block if present
    if "```json" in text:
        text = re.sub(r"```json\s*", "", text).replace("```", "").strip()
    elif "```" in text:
        text = re.sub(r"```\w*\s*", "", text).replace("```", "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("alert_id", alert_id)
    data["patient_id"] = patient_id  # Always use real patient_id, never LLM output (e.g. "MedID")
    try:
        return ClinicalBrief.model_validate(data)
    except Exception:
        return None
