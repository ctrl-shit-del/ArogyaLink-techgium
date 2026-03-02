"""Print first 20 readings for PT-0002 to verify exponential_acceleration trajectory."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.data_gen.patient_profiles import PATIENT_PROFILES
from scripts.data_gen.scenario_generator import generate_next_reading

profile = PATIENT_PROFILES["PT-0002"]
for i in range(20):
    r = generate_next_reading("PT-0002", profile, i)
    print(f"Reading {i}: HR={r['vitals']['heart_rate']:.1f} motion={r['context']['motion_score']}")
