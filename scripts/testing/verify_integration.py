"""
SYNERA 2.0 — Integration Verification Script
Tests TinyML, DRL, and RAG without needing a running server.
Run: python scripts/testing/verify_integration.py
"""
import os
import sys
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

PASS = "  PASS \u2713"
FAIL = "  FAIL \u2717"
SEP  = "=" * 60

print(SEP)
print("  SYNERA 2.0 \u2014 Integration Verification")
print(SEP)

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# 1. TinyML
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/3] TinyML (ONNX INT8 autoencoder)")
try:
    from backend.services.tinyml_service import tinyml_service as svc

    assert svc.ready, "Service not ready — check models/autoencoder_int8.onnx"

    normal_buf = [
        {"heart_rate": 80, "spo2": 97, "temperature": 37.0,
         "sys_bp_est": 120, "dia_bp_est": 80, "motion_score": 0}
    ] * 10
    anomaly_buf = [
        {"heart_rate": 80 + i * 7, "spo2": 97 - i * 0.4,
         "temperature": 37.0 + i * 0.08, "sys_bp_est": 120 + i * 3,
         "dia_bp_est": 80, "motion_score": 0}
        for i in range(10)
    ]

    err_n, alert_n = svc.score(normal_buf)
    err_a, alert_a = svc.score(anomaly_buf)

    assert err_a > err_n,  "Anomaly MSE should be > normal MSE"
    assert alert_a is True,  "Anomaly window must trigger pre_alert"
    assert alert_n is False, "Normal window must NOT trigger pre_alert"

    print(f"  threshold : {svc._threshold:.5f}")
    print(f"  normal    : MSE={err_n:.5f}  pre_alert={alert_n}")
    print(f"  anomaly   : MSE={err_a:.5f}  pre_alert={alert_a}  ({err_a/err_n:.1f}x higher)")
    print(PASS)
    results["tinyml"] = True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"{FAIL} \u2014 {e}")
    results["tinyml"] = False

# ─────────────────────────────────────────────────────────────────────────────
# 2. DRL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/3] DRL Triage Agent (PPO)")
try:
    from drl.agent import triage_agent
    from drl.state_builder import build_state_vector

    model_path = _ROOT / "models" / "drl" / "synera_triage_ppo.zip"
    print(f"  model file: {'FOUND' if model_path.exists() else 'NOT FOUND (rule-based fallback will be used)'}")

    loaded = triage_agent.load()
    print(f"  loaded    : {loaded}")

    # Simulate a SYNERA_STATE context (high-HR deterioration, medium sepsis risk)
    alert_ctx = {
        "vitals": {
            "heart_rate": 119, "spo2": 93, "temperature": 37.3,
            "sys_bp_est": 130, "motion_score": 0,
        },
        "trajectory": {
            "hr_baseline_mean": 80, "hr_baseline_std": 5,
            "spo2_baseline_mean": 97, "spo2_baseline_std": 1,
            "temp_baseline_mean": 37, "temp_baseline_std": 0.3,
            "bp_baseline_mean": 120, "bp_baseline_std": 8,
            "hr_first_derivative": 0.8,
            "hr_second_derivative": 0.12,
            "spo2_second_derivative": -0.05,
        },
        "patient": {
            "age": 34,
            "genomic_risk": {
                "cardiac": "Low", "respiratory": "Low",
                "diabetic": "Medium", "sepsis": "Medium",
            },
        },
        "context": {"hour": 14, "concurrent_alerts": 0, "ward_occupancy": 25},
    }

    sv = build_state_vector(alert_ctx)
    tier, conf = triage_agent.predict(sv)

    assert sv.shape == (17,), f"State vector wrong shape: {sv.shape}"
    assert tier in ("IMMEDIATE", "URGENT", "ELEVATED"), f"Unexpected tier: {tier}"

    print(f"  state_vec : shape={sv.shape}  range=[{sv.min():.2f}, {sv.max():.2f}]")
    print(f"  prediction: tier={tier}  confidence={conf:.2f}")
    print(f"  mode      : {'PPO model' if loaded else 'rule-based fallback'}")
    print(PASS)
    results["drl"] = True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"{FAIL} \u2014 {e}")
    results["drl"] = False

# ─────────────────────────────────────────────────────────────────────────────
# 3. RAG
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] RAG Pipeline (Groq LLM + Cohere embeddings)")
try:
    from rag.pipeline.alert_context_builder import AlertContext, PatientSummary, VitalPoint
    from backend.config.settings import settings

    groq_key   = os.getenv("GROQ_API_KEY")   or getattr(settings, "GROQ_API_KEY",   None)
    cohere_key = os.getenv("COHERE_API_KEY") or getattr(settings, "COHERE_API_KEY", None)
    llm_prov   = getattr(settings, "LLM_PROVIDER", "unknown")

    print(f"  LLM provider  : {llm_prov}")
    print(f"  GROQ_API_KEY  : {'SET (' + groq_key[:8] + '...)' if groq_key else 'NOT SET'}")
    print(f"  COHERE_API_KEY: {'SET (' + cohere_key[:8] + '...)' if cohere_key else 'NOT SET'}")

    # Build the context objects (no network call)
    ps = PatientSummary(
        patient_id="PT-0002",
        name="Priya Sharma",
        age=34,
        gender="Female",
        blood_group="B+",
        ward="Surgical Ward B",
        bed_number="4A",
        diagnosed_conditions=[{"name": "Post-appendectomy Day 2"}, {"name": "Type 2 Diabetes"}],
        current_medications=[
            {"name": "Cefazolin", "dose": "1g IV", "frequency": "Q8H"},
            {"name": "Metformin", "dose": "500mg", "frequency": "BD"},
        ],
        known_allergies=[],
        genomic_risk_cardiac="Low",
        genomic_risk_respiratory="Low",
        genomic_risk_sepsis="Medium",
        last_clinical_notes=None,
    )
    ctx = AlertContext(
        patient=ps,
        patient_age=34,
        trigger_vital="heart_rate",
        trigger_value=119.0,
        baseline_value=82.0,
        deviation_sigma=11.2,
        second_derivative=0.12,
        motion_score=0,
        vitals_window=[
            VitalPoint(heart_rate=119, spo2=93, temperature=37.3, motion_score=0)
        ],
        trigger_timestamp="2026-03-03T10:00:00Z",
    )
    print("  AlertContext  : build OK")

    if groq_key:
        print("  Live Groq call: testing...")
        from rag.pipeline.clinical_brief_generator import generate_brief_sync
        import time
        t0 = time.perf_counter()
        brief = generate_brief_sync(ctx, alert_id="VERIFY-001")
        elapsed = time.perf_counter() - t0
        b = brief.model_dump() if hasattr(brief, "model_dump") else brief.dict()
        ts = b.get("trigger_summary", "")
        actions = b.get("recommended_actions", [])
        assert ts, "trigger_summary is empty"
        assert len(actions) > 0, "No recommended actions"
        # Check patient data made it into the brief
        patient_ctx_ok = any(
            kw in ts.lower() for kw in ["priya", "post-op", "post-appendectomy", "sepsis"]
        )
        print(f"  response time : {elapsed*1000:.0f}ms")
        print(f"  trigger_summary: {ts[:100]}")
        print(f"  actions       : {len(actions)}")
        print(f"  patient ctx   : {'OK (patient data in brief)' if patient_ctx_ok else 'WARN - patient keywords not found in summary'}")
        print(PASS)
        results["rag"] = True
    else:
        # Even without Groq key, verify the builder/retrieval pipeline works offline
        print("  GROQ_API_KEY not set \u2014 skipping live LLM call")
        print("  (RAG will work when GROQ_API_KEY is present in .env)")
        print("  SKIP (keys not configured)")
        results["rag"] = None

except Exception as e:
    import traceback; traceback.print_exc()
    print(f"{FAIL} \u2014 {e}")
    results["rag"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  Summary")
print(SEP)
status_map = {True: "PASS \u2713", False: "FAIL \u2717", None: "SKIP (no key)"}
print(f"  TinyML  : {status_map[results.get('tinyml')]}")
print(f"  DRL     : {status_map[results.get('drl')]}")
print(f"  RAG     : {status_map[results.get('rag')]}")
print(SEP)

all_critical = results.get("tinyml") and results.get("drl")
if all_critical and results.get("rag") is not False:
    print("  VERDICT: Core pipeline is functional.")
    if results.get("rag") is None:
        print("  Set GROQ_API_KEY in .env to enable live RAG briefs.")
else:
    print("  VERDICT: One or more components FAILED. See errors above.")
print()
