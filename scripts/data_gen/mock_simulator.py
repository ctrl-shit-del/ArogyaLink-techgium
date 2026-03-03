"""Publishes vitals to backend via HTTP (ingest endpoint). Run in separate terminal."""
import asyncio
import os
import sys
from datetime import datetime, timezone
import random
import random as _rng

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx
from scripts.data_gen.patient_profiles import PATIENT_PROFILES
from scripts.data_gen.scenario_generator import (
    flat_with_noise,
    exponential_acceleration,
    exertion_spike,
    glitch_spike,
    slow_linear_drift,
)

BACKEND_URL = os.getenv("SYNERA_BACKEND_URL", "http://localhost:8000")
INTERVAL_SEC = 5


def build_trajectory(patient_id: str, n: int = 120):
    p = PATIENT_PROFILES[patient_id]
    traj = p["trajectory"]
    bh = p["baseline_hr"]
    bs = p["baseline_spo2"]
    bt = p.get("baseline_temp", 37.0)
    if traj == "flat_with_noise":
        pts = flat_with_noise(bh, bs, bt, n)
        return [(hr, spo2, temp, 1) for hr, spo2, temp in pts]
    if traj == "exponential_acceleration":
        stable = [(bh + _rng.gauss(0, 2), bs + _rng.gauss(0, 0.3), bt + _rng.gauss(0, 0.1), 1) for _ in range(15)]
        accel = exponential_acceleration(bh, 160, bs, 88, bt, 38.3, n - 15)
        return stable + [(hr, spo2, temp, 1) for hr, spo2, temp in accel]
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

    print("Synera Patient Simulator — publishing to backend via HTTP every 5s. Ctrl+C to stop.")
    print("  PT-0002 (Priya Sharma) will trigger SYNERA_STATE in ~3–4 minutes")
    print()

    async with httpx.AsyncClient(timeout=10.0) as client:
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
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                }

                try:
                    response = await client.post(
                        f"{BACKEND_URL}/api/v1/vitals/ingest",
                        json=payload,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        state = result.get("state", "")
                        hr_val = payload.get("heart_rate", "?")
                        if state == "SYNERA_STATE":
                            print(f"  🚨 [{patient_id}] SYNERA_STATE FIRED | HR={hr_val}")
                        elif state == "WATCH":
                            print(f"  ⚠️  [{patient_id}] WATCH | HR={hr_val}")
                        elif state == "EXERTION":
                            print(f"  🚶 [{patient_id}] EXERTION | HR={hr_val}")
                        elif state == "ARTIFACT":
                            print(f"  🗑️  [{patient_id}] ARTIFACT REJECTED | HR={hr_val}")
                    else:
                        print(f"  [{patient_id}] HTTP {response.status_code}")
                except Exception as e:
                    print(f"  [{patient_id}] Error: {e}")

                indices[patient_id] += 1

            await asyncio.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(run())


# ─────────────────────────────────────────────────────────────────────────────
# PatientSimulator — generator class (used by demo_scenarios.py)
# ─────────────────────────────────────────────────────────────────────────────

import math
import time
import random
from typing import Generator, Optional, Tuple
import numpy as np

from scripts.data_gen.patient_profiles import get_profile, ArchetypeProfile
from ml.shared.constants import (
    PHYSIOLOGICAL_NOISE_COV,
    ARCHETYPE_ARTIFACT_GLITCH,
    ARCHETYPE_EXERTION,
    ARCHETYPE_TRAJECTORY_ACCEL,
    SAMPLING_INTERVAL_SEC,
    CALIBRATION_WINDOW_READINGS,
)


class PatientSimulator:
    """
    Generates a physiologically realistic vital sign stream for one patient.
    Non-blocking generator — yields payloads matching the MQTT JSON structure.
    """

    def __init__(
        self,
        archetype: str,
        patient_id: str,
        seed: int = 42,
        personal_variation: float = 0.5,
    ):
        self.archetype  = archetype
        self.patient_id = patient_id
        self.seed       = seed
        self.profile    = get_profile(archetype)
        self._rng       = np.random.default_rng(seed)
        self._py_rng    = random.Random(seed)

        var = personal_variation
        self._baseline_hr   = self.profile.rest_hr_mean   + self._rng.normal(0, var * 4)
        self._baseline_spo2 = self.profile.rest_spo2_mean + self._rng.normal(0, var * 0.5)
        self._baseline_temp = self.profile.rest_temp_mean + self._rng.normal(0, var * 0.1)
        self._baseline_sys  = self.profile.rest_sys_bp    + self._rng.normal(0, var * 5)
        self._baseline_dia  = self.profile.rest_dia_bp    + self._rng.normal(0, var * 3)

        self._reading_count        = 0
        self._calibration_complete = False
        self._battery_pct          = self._py_rng.randint(60, 99)
        self._next_glitch_at       = self._py_rng.randint(
            self.profile.glitch_interval_min,
            self.profile.glitch_interval_max,
        )
        self._in_burst      = False
        self._burst_end     = 0
        self._next_burst_at = self.profile.stable_readings_before_event

    def generate_reading(self, timestamp: Optional[int] = None) -> dict:
        if timestamp is None:
            timestamp = int(time.time()) + self._reading_count * SAMPLING_INTERVAL_SEC

        n = self._reading_count
        is_glitch = (
            self.profile.hr_trajectory == "glitch_spike"
            and n == self._next_glitch_at
        )
        if is_glitch:
            self._next_glitch_at = n + self._py_rng.randint(
                self.profile.glitch_interval_min,
                self.profile.glitch_interval_max,
            )

        hr, spo2, temp, sys_bp, dia_bp, motion = self._compute_vitals(n, is_glitch)

        if n >= CALIBRATION_WINDOW_READINGS:
            self._calibration_complete = True
        if n % 720 == 0 and self._battery_pct > 5:
            self._battery_pct = max(5, self._battery_pct - self._py_rng.randint(0, 2))

        payload = {
            "patient_id": self.patient_id,
            "timestamp": timestamp,
            "firmware_ver": "2.0.1",
            "vitals": {
                "heart_rate":  round(hr, 1),
                "spo2":        round(spo2, 1),
                "temperature": round(temp, 2),
                "sys_bp_est":  round(sys_bp, 1),
                "dia_bp_est":  round(dia_bp, 1),
            },
            "context": {
                "motion_score": round(motion, 2),
                "is_active":    motion > 4.0,
                "battery_pct":  self._battery_pct,
            },
            "tinyml": {
                "reconstruction_error": None,
                "pre_alert":            None,
                "calibration_complete": self._calibration_complete,
            },
        }
        self._reading_count += 1
        return payload

    def generate_stream(
        self,
        n_readings: int,
        interval_seconds: int = SAMPLING_INTERVAL_SEC,
        start_timestamp: Optional[int] = None,
    ) -> Generator[dict, None, None]:
        """Yields n_readings payloads. Non-blocking — no sleep."""
        t0 = start_timestamp if start_timestamp is not None else int(time.time())
        for i in range(n_readings):
            yield self.generate_reading(timestamp=t0 + i * interval_seconds)

    def reset(self, seed: Optional[int] = None) -> None:
        new_seed = seed if seed is not None else self.seed
        self.__init__(self.archetype, self.patient_id, new_seed)

    @property
    def personal_baseline(self) -> dict:
        return {
            "hr":          round(self._baseline_hr, 1),
            "spo2":        round(self._baseline_spo2, 1),
            "temperature": round(self._baseline_temp, 2),
            "sys_bp":      round(self._baseline_sys, 1),
            "dia_bp":      round(self._baseline_dia, 1),
        }

    def _compute_vitals(
        self, n: int, is_glitch: bool
    ) -> Tuple[float, float, float, float, float, float]:
        profile = self.profile
        rng     = self._rng

        noise = rng.multivariate_normal(np.zeros(5), PHYSIOLOGICAL_NOISE_COV)
        hr_noise, spo2_noise, temp_noise, bp_noise, motion_noise = noise

        period_readings = int(120 / SAMPLING_INTERVAL_SEC)
        drift_hr = 3.0 * math.sin(2 * math.pi * n / period_readings)

        traj_hr_offset = traj_spo2_offset = traj_temp_offset = 0.0
        traj_sys_offset = traj_dia_offset = 0.0
        trajectory_motion = None

        if is_glitch:
            hr     = profile.glitch_hr_spike  + rng.uniform(-5, 5)
            spo2   = profile.glitch_spo2_drop + rng.uniform(-3, 3)
            temp   = self._baseline_temp + temp_noise * 0.3
            sys_bp = 180.0 + rng.uniform(-10, 10)
            dia_bp = 110.0 + rng.uniform(-5, 5)
            motion = float(np.clip(abs(motion_noise) * 3 + 6, 5, 10))
            return hr, spo2, temp, sys_bp, dia_bp, motion

        if profile.hr_trajectory == "exertion_burst":
            if n == self._next_burst_at and not self._in_burst:
                self._in_burst  = True
                self._burst_end = n + profile.trajectory_duration_readings
            if self._in_burst:
                if n < self._burst_end:
                    progress = (n - self._next_burst_at) / max(1, profile.trajectory_duration_readings)
                    if progress < 0.2:
                        factor = progress / 0.2
                    elif progress > 0.8:
                        factor = (1.0 - progress) / 0.2
                    else:
                        factor = 1.0
                    traj_hr_offset   = (profile.trajectory_target_hr - self._baseline_hr) * factor
                    traj_spo2_offset = -profile.spo2_drop_total * factor
                    trajectory_motion = profile.exertion_motion_level * factor + abs(motion_noise)
                else:
                    recovery_n = n - self._burst_end
                    if recovery_n < 60:
                        factor = max(0.0, 1.0 - recovery_n / 60.0)
                        traj_hr_offset   = (profile.trajectory_target_hr - self._baseline_hr) * factor * 0.5
                        traj_spo2_offset = -profile.spo2_drop_total * factor * 0.3
                    else:
                        self._in_burst      = False
                        self._next_burst_at = n + self._py_rng.randint(180, 360)

        elif profile.hr_trajectory == "exponential":
            event_start = profile.stable_readings_before_event
            if n >= event_start:
                elapsed    = n - event_start
                duration   = max(1, profile.trajectory_duration_readings)
                k          = 2.5
                base_delta = profile.trajectory_target_hr - self._baseline_hr
                exp_factor = (math.exp(k * min(elapsed, duration) / duration) - 1) / (math.exp(k) - 1)
                traj_hr_offset   = base_delta * exp_factor
                traj_spo2_offset = -profile.spo2_drop_total * exp_factor
                traj_temp_offset =  profile.temp_rise_total * exp_factor
                traj_sys_offset  = 20.0 * exp_factor
                traj_dia_offset  = 12.0 * exp_factor

        elif profile.hr_trajectory == "linear":
            duration = max(1, profile.trajectory_duration_readings)
            progress = min(n / duration, 1.0)
            traj_hr_offset   = (profile.trajectory_target_hr - self._baseline_hr) * progress
            traj_spo2_offset = -profile.spo2_drop_total * progress

        hr     = float(np.clip(self._baseline_hr + drift_hr + traj_hr_offset + hr_noise * 0.4, 30, 220))
        spo2   = float(np.clip(self._baseline_spo2 + traj_spo2_offset + spo2_noise * 0.3, 85, 100))
        temp   = float(np.clip(self._baseline_temp + traj_temp_offset + temp_noise * 0.15, 35, 42))
        sys_bp = float(np.clip(self._baseline_sys + traj_sys_offset + bp_noise * 0.8, 70, 200))
        dia_bp = float(np.clip(self._baseline_dia + traj_dia_offset + bp_noise * 0.5, 40, 120))

        if trajectory_motion is not None:
            motion = float(np.clip(trajectory_motion, 0, 10))
        else:
            base_motion = profile.rest_motion_mean
            if self._py_rng.random() < 0.02:
                base_motion += self._py_rng.uniform(1.5, 3.0)
            motion = float(np.clip(base_motion + abs(motion_noise) * 0.3, 0, 10))

        return hr, spo2, temp, sys_bp, dia_bp, motion

