"""Publishes vitals to internal event bus (replaces MQTT). Run in separate terminal."""
import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.data_gen.patient_profiles import PATIENT_PROFILES
from scripts.data_gen.scenario_generator import (
    flat_with_noise,
    exponential_acceleration,
    exertion_spike,
    glitch_spike,
    slow_linear_drift,
)
from backend.services.event_bus import bus

INTERVAL_SEC = 5


def build_trajectory(patient_id: str, n: int = 60):
    p = PATIENT_PROFILES[patient_id]
    traj = p["trajectory"]
    bh = p["baseline_hr"]
    bs = p["baseline_spo2"]
    bt = p.get("baseline_temp", 37.0)
    if traj == "flat_with_noise":
        pts = flat_with_noise(bh, bs, bt, n)
        return [(hr, spo2, temp, 1) for hr, spo2, temp in pts]
    if traj == "exponential_acceleration":
        pts = exponential_acceleration(bh, 140, bs, 91, bt, 37.8, n)
        return [(hr, spo2, temp, 1) for hr, spo2, temp in pts]
    if traj == "exertion_spike":
        return exertion_spike(bh, bs, 116, rise_readings=6, total_readings=n, motion_score=8)
    if traj == "glitch_spike":
        pts = glitch_spike(bh, bs, bt, n, p.get("glitch_value", 228), p.get("glitch_index", 3))
        return [(hr, spo2, temp) for hr, spo2, temp in pts]
    if traj == "slow_linear_drift":
        pts = slow_linear_drift(bh, 92, bs, bt, n)
        return [(hr, spo2, temp, 1) for hr, spo2, temp in pts]
    return flat_with_noise(bh, bs, bt, n)


async def run():
    trajectories = {pid: build_trajectory(pid) for pid in PATIENT_PROFILES}
    indices = {pid: 0 for pid in PATIENT_PROFILES}
    print("Synera Patient Simulator — publishing to event bus every 5s. Ctrl+C to stop.")
    while True:
        for patient_id in PATIENT_PROFILES:
            idx = indices[patient_id] % len(trajectories[patient_id])
            pt = trajectories[patient_id][idx]
            if len(pt) == 4:
                hr, spo2, temp, motion = pt
            else:
                hr, spo2, temp = pt[0], pt[1], pt[2]
                motion = 1
            payload = {
                "patient_id": patient_id,
                "heart_rate": round(hr, 1),
                "spo2": round(spo2, 1),
                "temperature": round(temp, 1),
                "motion_score": motion,
                "recorded_at": datetime.utcnow().isoformat() + "Z",
            }
            topic = f"synera/patient/{patient_id}/vitals"
            await bus.publish(topic, payload)
            indices[patient_id] += 1
        await asyncio.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(run())
