"""Parse MQTT JSON payload → VitalPayload schema."""
import json
from datetime import datetime
from typing import Any

from backend.models.schemas.vitals import VitalPayload


def parse_mqtt_payload(raw: bytes | str) -> VitalPayload | None:
    """Parse MQTT message body to VitalPayload. Returns None on parse error."""
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(data, dict):
            return None
        patient_id = data.get("patient_id") or data.get("patientId")
        if not patient_id:
            return None
        recorded_at = data.get("recorded_at") or data.get("timestamp") or data.get("recordedAt")
        if isinstance(recorded_at, (int, float)):
            recorded_at = datetime.utcfromtimestamp(recorded_at)
        elif isinstance(recorded_at, str):
            try:
                recorded_at = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
            except Exception:
                recorded_at = datetime.utcnow()
        elif recorded_at is None:
            recorded_at = datetime.utcnow()
        return VitalPayload(
            patient_id=str(patient_id),
            heart_rate=_float(data, "heart_rate", "heartRate", "hr"),
            spo2=_float(data, "spo2", "SpO2"),
            temperature=_float(data, "temperature", "temp"),
            sys_bp_est=_float(data, "sys_bp_est", "sys_bp", "systolic_bp"),
            dia_bp_est=_float(data, "dia_bp_est", "dia_bp", "diastolic_bp"),
            motion_score=_int(data, "motion_score", "motionScore", "motion"),
            is_active=data.get("is_active", data.get("isActive")),
            battery_pct=_int(data, "battery_pct", "battery", "batteryPct"),
            reconstruction_error=_float(data, "reconstruction_error", "reconstructionError"),
            recorded_at=recorded_at,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _float(d: dict, *keys: str) -> float | None:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _int(d: dict, *keys: str) -> int | None:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    return None
