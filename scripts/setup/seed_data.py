"""Insert 5 mock patients into Supabase. Run after schema."""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config.database import get_db
from scripts.data_gen.patient_profiles import PATIENT_PROFILES


def seed():
    print("Seeding 5 mock patients into Supabase...")
    db = get_db()
    for pid, profile in PATIENT_PROFILES.items():
        try:
            existing = db.table("patients").select("patient_id").eq("patient_id", pid).execute()
            if existing.data and len(existing.data) > 0:
                print(f"  Skip {pid} (exists)")
                continue
        except Exception:
            pass
        row = {
            "patient_id": pid,
            "name": profile["name"],
            "dob": date.today().replace(year=date.today().year - profile["age"]).isoformat(),
            "gender": profile.get("gender"),
            "ward": profile.get("ward"),
            "bed_number": profile.get("bed_number") or profile.get("bed"),
            "diagnosed_conditions": profile.get("conditions", []),
            "current_medications": profile.get("medications", []),
            "known_allergies": [],
            "baseline_hr_mean": profile.get("baseline_hr"),
            "baseline_hr_std": 5.0,
            "baseline_spo2_mean": profile.get("baseline_spo2"),
            "baseline_spo2_std": 1.5,
            "baseline_temp_mean": profile.get("baseline_temp", 37.0),
            "calibration_complete": True,
        }
        db.table("patients").insert(row).execute()
        print(f"  OK  {pid} {profile['name']}")
    print("Done. 5 patients seeded.")


if __name__ == "__main__":
    seed()
