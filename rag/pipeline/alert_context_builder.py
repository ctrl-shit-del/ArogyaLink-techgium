"""Assembles context from patient + vitals. Builds retrieval query from patient data."""
from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class PatientSummary:
    """Minimal patient (MedID) for RAG context."""
    patient_id: str
    name: str
    age: int
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    ward: Optional[str] = None
    bed_number: Optional[str] = None
    diagnosed_conditions: List[dict] = None
    current_medications: List[dict] = None
    known_allergies: List[dict] = None
    genomic_risk_cardiac: str = "Unknown"
    genomic_risk_respiratory: str = "Unknown"
    genomic_risk_sepsis: str = "Unknown"
    last_clinical_notes: Optional[str] = None

    def __post_init__(self):
        if self.diagnosed_conditions is None:
            self.diagnosed_conditions = []
        if self.current_medications is None:
            self.current_medications = []
        if self.known_allergies is None:
            self.known_allergies = []


@dataclass
class VitalPoint:
    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    temperature: Optional[float] = None
    motion_score: Optional[int] = None


@dataclass
class AlertContext:
    patient: PatientSummary
    patient_age: int
    trigger_vital: str
    trigger_value: float
    baseline_value: float
    deviation_sigma: float
    second_derivative: float
    motion_score: int
    vitals_window: List[VitalPoint]
    trigger_timestamp: Optional[str] = None


CONDITION_KEYWORD_MAP = {
    "Type 2 Diabetes": "diabetic hyperglycaemia hypoglycaemia insulin",
    "Hypertension": "hypertensive blood pressure antihypertensive",
    "COPD": "COPD chronic obstructive pulmonary respiratory exacerbation",
    "Atrial Fibrillation": "atrial fibrillation arrhythmia cardiac anticoagulation",
    "CKD": "chronic kidney disease renal impairment electrolytes",
    "Post-appendectomy": "post-operative surgical site infection sepsis wound",
    "Post-operative": "post-operative surgical complication sepsis wound infection",
}

VITAL_TERM_MAP = {
    "heart_rate": "tachycardia heart rate acceleration",
    "spo2": "hypoxia oxygen desaturation SpO2 decline",
    "blood_pressure": "hypotension blood pressure drop haemodynamic instability",
}


def build_retrieval_query(alert_context: AlertContext) -> str:
    """Constructs semantically rich query from alert + patient MedID."""
    query_parts = []
    query_parts.append(VITAL_TERM_MAP.get(alert_context.trigger_vital, alert_context.trigger_vital))
    query_parts.append("at rest unexplained acceleration trajectory")

    for condition in alert_context.patient.diagnosed_conditions:
        name = (condition.get("name") or condition.get("condition") or "").strip()
        if not name:
            continue
        found = False
        for key, keywords in CONDITION_KEYWORD_MAP.items():
            if key.lower() in name.lower():
                query_parts.append(keywords)
                found = True
                break
        if not found:
            query_parts.append(name)

    if alert_context.patient_age < 18:
        query_parts.append("paediatric child IMCI danger signs")

    if alert_context.patient.current_medications:
        med_names = [m.get("name", "") for m in alert_context.patient.current_medications if m]
        query_parts.append("drug interaction " + " ".join(med_names))

    if alert_context.patient.genomic_risk_cardiac in ("High", "Critical"):
        query_parts.append("sudden cardiac event high cardiac risk arrhythmia")
    if alert_context.patient.genomic_risk_sepsis in ("High", "Critical"):
        query_parts.append("sepsis susceptibility high risk early sepsis recognition")
    if alert_context.patient.genomic_risk_respiratory in ("High", "Critical"):
        query_parts.append("respiratory failure high risk oxygen supplementation")

    if alert_context.vitals_window and len(alert_context.vitals_window) > 0:
        last = alert_context.vitals_window[-1]
        if getattr(last, "spo2", None) is not None and last.spo2 < 95:
            query_parts.append("oxygen supplementation pulse oximetry SpO2 management")

    return " ".join(query_parts)


def format_vitals_as_table(vitals_window: List[VitalPoint]) -> str:
    """Format last 10 readings as text table for prompt."""
    lines = []
    n = len(vitals_window)
    for i, v in enumerate(vitals_window):
        t = (n - 1 - i) * 5
        hr = getattr(v, "heart_rate", None) or "-"
        spo2 = getattr(v, "spo2", None) or "-"
        temp = getattr(v, "temperature", None) or "-"
        motion = getattr(v, "motion_score", None) or "-"
        lines.append(f"T-{t}s: HR={hr} SpO2={spo2}% Temp={temp} Motion={motion}")
    return "\n".join(lines) if lines else "No vitals."
