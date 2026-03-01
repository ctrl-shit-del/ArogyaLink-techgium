"""60 synthetic anonymised clinical case summaries (12 per archetype × 5)."""
import random
ARCHETYPES = ["post_op_sepsis", "cardiac_event", "PE", "hypoglycaemia", "COPD_exacerbation"]
OUTCOMES = ["sepsis_confirmed", "cardiac_event", "PE_treated", "hypoglycaemia_corrected", "COPD_stabilised"]
TEMPLATES = [
    "[CASE] 58F, post-operative day 3, Type 2 Diabetes, Hypertension. HR accelerated from 84 to 138 over 22 minutes at rest. SpO2 declined to 90%. Temperature 38.4°C. Motion score 1 throughout. DIAGNOSIS: Surgical site sepsis (confirmed blood culture: E. coli). INTERVENTION: Blood cultures x2, IV Meropenem 1g started within 1 hour, O2 supplementation 4L/min, ICU transfer. OUTCOME: Recovered. ICU stay 4 days. [MOHFW Sepsis Protocol §4.2]",
    "[CASE] 62M, cardiac history, genomic_risk_cardiac High. HR 78 to 142 at rest over 15 min. SpO2 stable. DIAGNOSIS: Atrial fibrillation with RVR. INTERVENTION: Rate control, anticoagulation review. OUTCOME: Converted to sinus rhythm. [MOHFW Cardiac §2.3]",
    "[CASE] 45F, post-op day 2, immobile. HR rise + SpO2 drop, no fever. DIAGNOSIS: Pulmonary embolism. INTERVENTION: CTPA, anticoagulation. OUTCOME: Improved. [MOHFW PostOp]",
    "[CASE] 55M, diabetic. Tachycardia, diaphoresis, low temp. Blood glucose 52 mg/dL. INTERVENTION: Dextrose, recheck. OUTCOME: Hypoglycaemia corrected. [IP2022 + Sepsis]",
    "[CASE] 71F, COPD. SpO2 decline, respiratory distress. INTERVENTION: Nebulisation, O2, steroids. OUTCOME: COPD exacerbation stabilised. [MOHFW Respiratory]",
]
SYNTHETIC_CLINICAL_CASES = []
for i in range(60):
    arch = ARCHETYPES[i % 5]
    t = TEMPLATES[i % 5]
    SYNTHETIC_CLINICAL_CASES.append({
        "text": t,
        "metadata": {
            "source_document": "clinical_cases",
            "archetype": arch,
            "patient_age_range": random.choice(["50-65", "40-55", "65-80"]),
            "conditions": ["diabetes", "post_operative"] if "post_op" in arch else ["COPD"] if "COPD" in arch else [],
            "trigger_vital": random.choice(["heart_rate", "spo2"]),
            "outcome": OUTCOMES[i % 5],
            "severity": random.choice(["high", "medium"]),
        }
    })
