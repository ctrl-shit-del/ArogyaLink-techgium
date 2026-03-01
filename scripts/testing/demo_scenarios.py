"""Runs all 3 judge demo scenarios (PT-0002, PT-0003, PT-0004/5)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.data_gen.patient_profiles import PATIENT_PROFILES


def main():
    print("Demo scenarios (expected outcomes):")
    for pid, p in PATIENT_PROFILES.items():
        print(f"  {pid} {p['name']}: {p['expected_outcome']}")
    print("\nRun simulator: python scripts/data_gen/mock_simulator.py")
    print("Then watch backend logs for SYNERA_STATE / EXERTION_LOGGED / ARTIFACT / WATCH.")


if __name__ == "__main__":
    main()
