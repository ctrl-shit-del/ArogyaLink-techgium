"""End-to-end RAG pipeline with mock alert."""
import pytest
from rag.pipeline.alert_context_builder import AlertContext, PatientSummary, VitalPoint, build_retrieval_query


def test_build_retrieval_query_pt0002():
    patient = PatientSummary(
        patient_id="PT-0002",
        name="Priya Sharma",
        age=34,
        ward="Surgical Ward B",
        bed_number="4A",
        diagnosed_conditions=[{"name": "Post-appendectomy Day 2"}, {"name": "Type 2 Diabetes"}],
        current_medications=[
            {"name": "Cefazolin", "dose": "1g IV", "frequency": "Q8H"},
            {"name": "Metformin", "dose": "500mg", "frequency": "BD"},
        ],
    )
    ctx = AlertContext(
        patient=patient,
        patient_age=34,
        trigger_vital="heart_rate",
        trigger_value=125.0,
        baseline_value=82.0,
        deviation_sigma=2.5,
        second_derivative=0.1,
        motion_score=1,
        vitals_window=[VitalPoint(heart_rate=82 + i * 5, spo2=97 - i * 0.5, motion_score=1) for i in range(10)],
    )
    q = build_retrieval_query(ctx)
    assert "tachycardia" in q
    assert "post-operative" in q or "diabetic" in q
    assert "drug interaction" in q
    assert "Cefazolin" in q or "Metformin" in q
