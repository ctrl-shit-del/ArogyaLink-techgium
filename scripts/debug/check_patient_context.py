"""Verify PT-0002 (and other patients) record in Supabase for RAG context."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from backend.services.database.patient_repo import PatientRepository
import asyncio


async def check():
    repo = PatientRepository()
    patient = await repo.get_by_id("PT-0002")
    if not patient:
        print("FAIL: PT-0002 not found")
        return
    print("Patient record:")
    for k, v in patient.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    asyncio.run(check())
