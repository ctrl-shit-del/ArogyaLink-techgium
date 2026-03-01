"""AlertContext → ClinicalBrief JSON (or rule-based fallback)."""
import time
from datetime import datetime
from pathlib import Path

from rag.pipeline.alert_context_builder import AlertContext, format_vitals_as_table
from rag.retrieval.retriever import search, RetrievedChunk
from rag.retrieval.reranker import rerank
from rag.knowledge_base.processed.embedder import embed_query
from rag.llm.chains.clinical_chain import invoke_clinical_chain
from rag.llm.parsers.brief_parser import parse
from rag.llm.parsers.citation_parser import format_citations

# In-container we may not have backend
try:
    from backend.models.schemas.clinical_brief import ClinicalBrief, DifferentialItem, RecommendedAction, DrugInteractionFlag, Source
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


def rule_based_fallback_brief(alert_context: AlertContext, alert_id: str = "") -> ClinicalBrief:
    """When LLM fails or times out."""
    p = alert_context.patient
    conditions = [c.get("name", "") for c in p.diagnosed_conditions]
    return ClinicalBrief(
        alert_id=alert_id or "fallback",
        patient_id=p.patient_id,
        generated_at=datetime.utcnow().isoformat() + "Z",
        trigger_summary=f"{alert_context.trigger_vital} deviation {alert_context.deviation_sigma:.1f}σ. Trajectory acceleration detected. Verify clinically.",
        differential_diagnosis=[
            DifferentialItem(condition="Clinical verification required", likelihood="Most Likely", reasoning="Rule-based fallback; no LLM output.")
        ],
        recommended_actions=[
            RecommendedAction(priority=1, action="Assess patient at bedside", rationale="Trajectory alert fired."),
            RecommendedAction(priority=2, action="Review vitals and history", rationale="Structured assessment."),
        ],
        drug_interaction_flags=[],
        relevant_history=conditions[:5],
        sources=[],
        confidence_note="This is decision support. Verify clinically.",
        generation_time_ms=0,
    )


def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    formatted = []
    for i, c in enumerate(chunks, 1):
        meta = c.metadata or {}
        if meta.get("source_document") == "clinical_cases":
            formatted.append(f"[CASE EXAMPLE {i}]\n{c.text}\n")
        else:
            formatted.append(
                f"[GUIDELINE {i} — {meta.get('source_label', 'Unknown')}, {meta.get('section', '')}]\n{c.text}\n"
            )
    return "\n".join(formatted)


JSON_SCHEMA_SNIPPET = """
{
  "alert_id": "string",
  "patient_id": "string",
  "generated_at": "ISO8601",
  "trigger_summary": "string",
  "differential_diagnosis": [{"condition": "string", "likelihood": "Most Likely|Possible|Rule Out", "reasoning": "string"}],
  "recommended_actions": [{"priority": 1, "action": "string", "rationale": "string"}],
  "drug_interaction_flags": [{"medication": "string", "flag": "string", "severity": "Critical|Warning|Info"}],
  "relevant_history": ["string"],
  "sources": [{"document": "string", "section": "string", "relevance": "string"}],
  "confidence_note": "string (must end with: This is decision support. Verify clinically.)",
  "generation_time_ms": 0
}
"""


def generate_brief_sync(alert_context: AlertContext, alert_id: str = "") -> ClinicalBrief:
    """Synchronous generate: retrieve, rerank, prompt, parse. Returns ClinicalBrief or fallback."""
    start = time.perf_counter()
    from rag.pipeline.alert_context_builder import build_retrieval_query
    query = build_retrieval_query(alert_context)
    query_embedding = embed_query(
        f"Represent this sentence for searching relevant passages: {query}"
    )
    # Stage 1: general + drug-filtered
    stage1_general = search("medical_knowledge", query, top_k=8, query_embedding=query_embedding)
    stage1_drug = []
    if alert_context.patient.current_medications:
        stage1_drug = search("medical_knowledge", query, top_k=4, where={"clinical_domain": "pharmacology"}, query_embedding=query_embedding)
    stage2_cases = search("clinical_cases", query, top_k=4, where={"trigger_vital": alert_context.trigger_vital}, query_embedding=query_embedding)
    merged = stage1_general + stage1_drug + stage2_cases
    seen = set()
    dedup = []
    for c in merged:
        if c.id not in seen:
            seen.add(c.id)
            dedup.append(c)
    reranked = rerank(query, dedup, top_k=6)
    chunks_formatted = format_chunks_for_prompt(reranked)

    p = alert_context.patient
    conditions_list = ", ".join([c.get("name", "") for c in p.diagnosed_conditions]) or "None"
    meds_list = ", ".join([f"{m.get('name', '')} {m.get('dose', '')} {m.get('frequency', '')}" for m in p.current_medications]) or "None"
    allergies_list = ", ".join([a.get("substance", "") for a in p.known_allergies]) or "None known"
    trend_description = (
        f"{alert_context.trigger_vital} value {alert_context.trigger_value} "
        f"(baseline {alert_context.baseline_value}). Deviation {alert_context.deviation_sigma:.1f}σ. "
        f"Second derivative {alert_context.second_derivative:.4f}. Patient at rest (motion={alert_context.motion_score})."
    )
    vitals_table = format_vitals_as_table(alert_context.vitals_window)

    prompt_dir = Path(__file__).parent.parent / "llm" / "prompts"
    tpl = (prompt_dir / "clinical_brief.txt").read_text(encoding="utf-8")
    prompt = tpl.format(
        patient_id=p.patient_id,
        ward=p.ward or "",
        bed_number=p.bed_number or "",
        trigger_timestamp=alert_context.trigger_timestamp or datetime.utcnow().isoformat() + "Z",
        trend_description=trend_description,
        trigger_vital=alert_context.trigger_vital,
        trigger_value=alert_context.trigger_value,
        baseline_value=alert_context.baseline_value,
        deviation_sigma=alert_context.deviation_sigma,
        second_derivative=alert_context.second_derivative,
        motion_score=alert_context.motion_score,
        vitals_table=vitals_table,
        patient_name=p.name,
        patient_age=alert_context.patient_age,
        patient_gender=p.gender or "",
        blood_group=p.blood_group or "",
        conditions_list=conditions_list,
        medications_list=meds_list,
        allergies_list=allergies_list,
        genomic_cardiac=p.genomic_risk_cardiac,
        genomic_respiratory=p.genomic_risk_respiratory,
        genomic_sepsis=p.genomic_risk_sepsis,
        last_clinical_notes=p.last_clinical_notes or "None recorded",
        retrieved_chunks_formatted=chunks_formatted,
        json_schema=JSON_SCHEMA_SNIPPET,
    )
    raw = invoke_clinical_chain(prompt)
    brief = parse(raw, alert_id=alert_id or "rag", patient_id=p.patient_id)
    if brief is None:
        brief = rule_based_fallback_brief(alert_context, alert_id)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    try:
        brief = brief.model_copy(update={"generation_time_ms": elapsed_ms})
    except AttributeError:
        brief.generation_time_ms = elapsed_ms
    return brief
