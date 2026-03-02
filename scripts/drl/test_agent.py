"""
scripts/drl/test_agent.py
Live terminal demo — DRL decisions for 5 mock patients.
Usage: set PYTHONPATH=<repo> && python scripts/drl/test_agent.py
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from drl.agent import triage_agent
from drl.state_builder import build_state_vector

DEMO_SCENARIOS = [
    {
        "label": "PT-0001 — Rahul (Stable, Minor Deviation)",
        "context": {
            "vitals": {"heart_rate": 88, "spo2": 96, "temperature": 37.1,
                       "sys_bp_est": 125, "motion_score": 1},
            "trajectory": {
                "hr_baseline_mean": 82, "hr_baseline_std": 4.5,
                "spo2_baseline_mean": 97, "spo2_baseline_std": 0.8,
                "temp_baseline_mean": 36.8, "temp_baseline_std": 0.3,
                "bp_baseline_mean": 120, "bp_baseline_std": 7,
                "hr_first_derivative": 1.5, "hr_second_derivative": 0.3,
                "spo2_second_derivative": 0.0,
            },
            "patient": {"age": 34, "genomic_risk": {
                "cardiac": "Low", "respiratory": "Low",
                "diabetic": "Low", "sepsis": "Low"}},
            "context": {"hour": 10, "concurrent_alerts": 0, "ward_occupancy": 20},
        },
        "expected": "ELEVATED",
    },
    {
        "label": "PT-0002 — Priya (Post-Op Sepsis, CRITICAL)",
        "context": {
            "vitals": {"heart_rate": 119, "spo2": 93, "temperature": 37.8,
                       "sys_bp_est": 135, "motion_score": 1},
            "trajectory": {
                "hr_baseline_mean": 82, "hr_baseline_std": 4.2,
                "spo2_baseline_mean": 97, "spo2_baseline_std": 0.7,
                "temp_baseline_mean": 36.9, "temp_baseline_std": 0.3,
                "bp_baseline_mean": 118, "bp_baseline_std": 6,
                "hr_first_derivative": 8.5, "hr_second_derivative": 2.8,
                "spo2_second_derivative": 0.9,
            },
            "patient": {"age": 45, "genomic_risk": {
                "cardiac": "Medium", "respiratory": "Low",
                "diabetic": "High", "sepsis": "High"}},
            "context": {"hour": 3, "concurrent_alerts": 1, "ward_occupancy": 38},
        },
        "expected": "IMMEDIATE",
    },
    {
        "label": "PT-0003 — Arjun (Exertion)",
        "context": {
            "vitals": {"heart_rate": 116, "spo2": 95, "temperature": 36.9,
                       "sys_bp_est": 130, "motion_score": 8},
            "trajectory": {
                "hr_baseline_mean": 78, "hr_baseline_std": 5.0,
                "spo2_baseline_mean": 97, "spo2_baseline_std": 1.0,
                "temp_baseline_mean": 36.7, "temp_baseline_std": 0.3,
                "bp_baseline_mean": 122, "bp_baseline_std": 8,
                "hr_first_derivative": 12.0, "hr_second_derivative": 3.5,
                "spo2_second_derivative": 0.2,
            },
            "patient": {"age": 28, "genomic_risk": {
                "cardiac": "Low", "respiratory": "Low",
                "diabetic": "Low", "sepsis": "Low"}},
            "context": {"hour": 9, "concurrent_alerts": 0, "ward_occupancy": 25},
        },
        "expected": "ELEVATED",
    },
    {
        "label": "PT-0004 — Fatima (Glitch — Rule A discards)",
        "context": {
            "vitals": {"heart_rate": 228, "spo2": 97, "temperature": 37.0,
                       "sys_bp_est": 120, "motion_score": 2},
            "trajectory": {
                "hr_baseline_mean": 79, "hr_baseline_std": 3.5,
                "spo2_baseline_mean": 98, "spo2_baseline_std": 0.5,
                "temp_baseline_mean": 36.8, "temp_baseline_std": 0.2,
                "bp_baseline_mean": 115, "bp_baseline_std": 5,
                "hr_first_derivative": 0.5, "hr_second_derivative": 0.1,
                "spo2_second_derivative": 0.0,
            },
            "patient": {"age": 30, "genomic_risk": {
                "cardiac": "Low", "respiratory": "Low",
                "diabetic": "Low", "sepsis": "Low"}},
            "context": {"hour": 14, "concurrent_alerts": 0, "ward_occupancy": 22},
        },
        "expected": "Rule A rejects",
    },
    {
        "label": "PT-0005 — Govind (COPD Exacerbation, URGENT)",
        "context": {
            "vitals": {"heart_rate": 105, "spo2": 91, "temperature": 37.5,
                       "sys_bp_est": 142, "motion_score": 0},
            "trajectory": {
                "hr_baseline_mean": 88, "hr_baseline_std": 5.5,
                "spo2_baseline_mean": 93, "spo2_baseline_std": 1.2,
                "temp_baseline_mean": 37.0, "temp_baseline_std": 0.4,
                "bp_baseline_mean": 135, "bp_baseline_std": 10,
                "hr_first_derivative": 4.5, "hr_second_derivative": 1.8,
                "spo2_second_derivative": 0.7,
            },
            "patient": {"age": 62, "genomic_risk": {
                "cardiac": "Medium", "respiratory": "Critical",
                "diabetic": "Low", "sepsis": "Medium"}},
            "context": {"hour": 22, "concurrent_alerts": 2, "ward_occupancy": 45},
        },
        "expected": "URGENT",
    },
]

TIER_COLORS = {"IMMEDIATE": "\033[91m", "URGENT": "\033[93m", "ELEVATED": "\033[92m"}
RESET = "\033[0m"


def run_demo():
    print("\n" + "=" * 65)
    print("  SYNERA 2.0 — DRL TRIAGE AGENT LIVE DEMO")
    print("  Team ArogyaLink | Techgium Season 9")
    print("=" * 65)

    loaded = triage_agent.load()
    if not loaded:
        print("\n[WARNING] No pre-trained model. Run: python scripts/drl/run_pretrain.py")
        print("  Using rule-based fallback.\n")
    else:
        print("\n[OK] Pre-trained PPO model loaded.\n")

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n── Patient {i}: {scenario['label']}")
        if "Rule A" in scenario["expected"]:
            print("   [Rule A] Artifact rejected. DRL not invoked.")
            time.sleep(0.5)
            continue

        state = build_state_vector(scenario["context"])
        action_label, confidence = triage_agent.predict(state)
        color = TIER_COLORS.get(action_label, "")
        source = "PPO Agent" if confidence > 0 else "Rule Fallback"
        conf_str = f"{confidence:.0%}" if confidence > 0 else "N/A"

        print(f"   Priority Tier : {color}{action_label}{RESET}")
        print(f"   Expected      : {scenario['expected']}")
        print(f"   Source        : {source} | Confidence: {conf_str}")
        print(f"   Match         : {'✓ YES' if action_label == scenario['expected'] else '✗ NO'}")
        time.sleep(0.8)

    print("\n" + "=" * 65)
    print("  DEMO COMPLETE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
