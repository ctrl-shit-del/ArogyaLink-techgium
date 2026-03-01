"""Main orchestrator: event_bus → rules → RAG → WS. No Docker/MQTT/Redis."""
import asyncio
import logging
from datetime import datetime
from uuid import uuid4

from backend.models.schemas.vitals import VitalPayload
from backend.core.trajectory.window_buffer import WindowBuffer, VitalReading
from backend.core.synera_engine.state_manager import state_manager, PatientState
from backend.core.synera_engine.pipeline import run_pipeline
from backend.core.rules.rule_runner import RuleResult, RuleRunnerConfig
from backend.config.settings import settings
from backend.core.buffer.in_memory_store import store as in_memory_store
from backend.services.database.patient_repo import get_patient
from backend.services.database.alert_repo import create_alert
from backend.services.database.vitals_repo import insert_vital
from backend.services.notifications.alert_dispatcher import (
    dispatch_synera_state,
    dispatch_state_change,
    dispatch_exertion_logged,
)

logger = logging.getLogger(__name__)

rule_config = RuleRunnerConfig(
    artifact_hr_delta=int(settings.ARTIFACT_REJECT_HR_DELTA),
    artifact_spo2_delta=int(settings.ARTIFACT_REJECT_SPO2_DELTA),
    exertion_motion_threshold=settings.EXERTION_MOTION_THRESHOLD,
    exertion_hr_elevation=int(settings.EXERTION_HR_ELEVATION),
    synera_sigma_threshold=settings.SYNERA_STATE_SIGMA_THRESHOLD,
    synera_motion_max=settings.SYNERA_STATE_MOTION_MAX,
    synera_acceleration_window=settings.SYNERA_ACCELERATION_WINDOW,
)


def _deque_to_window_buffer(deque_buf) -> WindowBuffer:
    """Build WindowBuffer from store deque (list of dicts)."""
    buf = WindowBuffer(maxlen=10)
    for d in deque_buf:
        buf.append(VitalReading(
            heart_rate=d.get("heart_rate"),
            spo2=d.get("spo2"),
            temperature=d.get("temperature"),
            sys_bp_est=d.get("sys_bp_est"),
            dia_bp_est=d.get("dia_bp_est"),
            motion_score=d.get("motion_score"),
            recorded_at=d.get("recorded_at"),
        ))
    return buf


def _payload_to_reading_dict(payload: VitalPayload) -> dict:
    return {
        "heart_rate": payload.heart_rate,
        "spo2": payload.spo2,
        "temperature": payload.temperature,
        "sys_bp_est": payload.sys_bp_est,
        "dia_bp_est": payload.dia_bp_est,
        "motion_score": payload.motion_score,
        "recorded_at": payload.recorded_at.timestamp() if payload.recorded_at else None,
    }


def _build_rag_context_and_invoke(patient: dict, buffer: WindowBuffer, trigger_vital: str, trigger_value: float,
                                   baseline_value: float, deviation_sigma: float, second_derivative: float,
                                   motion_score: int, alert_id: str) -> dict:
    """Build AlertContext and call in-process RAG; return clinical_brief dict."""
    from datetime import date
    from rag.pipeline.alert_context_builder import AlertContext, PatientSummary, VitalPoint
    from rag.pipeline.clinical_brief_generator import generate_brief_sync

    def age_from_dob(dob):
        if not dob:
            return 40
        if isinstance(dob, str):
            try:
                dob = date.fromisoformat(dob[:10])
            except Exception:
                return 40
        if not isinstance(dob, date):
            return 40
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    age = age_from_dob(patient.get("dob"))
    vitals_window = [VitalPoint(heart_rate=r.heart_rate, spo2=r.spo2, temperature=r.temperature, motion_score=r.motion_score) for r in buffer.get_readings()]
    ps = PatientSummary(
        patient_id=patient["patient_id"],
        name=patient["name"],
        age=age,
        gender=patient.get("gender"),
        blood_group=patient.get("blood_group"),
        ward=patient.get("ward"),
        bed_number=patient.get("bed_number"),
        diagnosed_conditions=patient.get("diagnosed_conditions") or [],
        current_medications=patient.get("current_medications") or [],
        known_allergies=patient.get("known_allergies") or [],
        genomic_risk_cardiac=patient.get("genomic_risk_cardiac") or "Unknown",
        genomic_risk_respiratory=patient.get("genomic_risk_respiratory") or "Unknown",
        genomic_risk_sepsis=patient.get("genomic_risk_sepsis") or "Unknown",
        last_clinical_notes=patient.get("last_clinical_notes"),
    )
    ctx = AlertContext(
        patient=ps,
        patient_age=age,
        trigger_vital=trigger_vital,
        trigger_value=trigger_value,
        baseline_value=baseline_value,
        deviation_sigma=deviation_sigma,
        second_derivative=second_derivative,
        motion_score=motion_score,
        vitals_window=vitals_window,
        trigger_timestamp=datetime.utcnow().isoformat() + "Z",
    )
    brief = generate_brief_sync(ctx, alert_id=alert_id)
    return brief.model_dump() if hasattr(brief, "model_dump") else brief.dict()


async def process_vital_payload(payload: VitalPayload) -> None:
    """Handle one vital payload: rules → optional RAG → DB → WS."""
    patient_id = payload.patient_id
    deque_buf = await in_memory_store.get_buffer(patient_id)
    buffer = _deque_to_window_buffer(deque_buf)

    patient = get_patient(patient_id)
    if not patient:
        logger.warning("Unknown patient_id=%s, skipping", patient_id)
        return

    baseline_hr = patient.get("baseline_hr_mean") or 80.0
    baseline_hr_std = patient.get("baseline_hr_std") or 5.0
    baseline_spo2 = patient.get("baseline_spo2_mean")
    baseline_spo2_std = patient.get("baseline_spo2_std")
    baseline_temp = patient.get("baseline_temp_mean")
    baseline_temp_std = patient.get("baseline_temp_std")

    result, extra = run_pipeline(
        payload=payload,
        buffer=buffer,
        baseline_hr_mean=baseline_hr,
        baseline_hr_std=baseline_hr_std,
        baseline_spo2_mean=baseline_spo2,
        baseline_spo2_std=baseline_spo2_std,
        baseline_temp_mean=baseline_temp,
        baseline_temp_std=baseline_temp_std,
        config=rule_config,
    )

    if result == RuleResult.ARTIFACT:
        logger.debug("ARTIFACT discarded patient_id=%s", patient_id)
        return

    await in_memory_store.append_to_buffer(patient_id, _payload_to_reading_dict(payload))

    if result == RuleResult.EXERTION:
        await dispatch_exertion_logged(
            patient_id=patient_id,
            motion_score=extra.get("motion_score", 0),
            hr_elevation=extra.get("hr_elevation", 0),
        )
        insert_vital(
            patient_id=patient_id,
            recorded_at=payload.recorded_at or datetime.utcnow(),
            heart_rate=payload.heart_rate,
            spo2=payload.spo2,
            temperature=payload.temperature,
            sys_bp_est=payload.sys_bp_est,
            dia_bp_est=payload.dia_bp_est,
            motion_score=payload.motion_score,
        )
        return

    if result == RuleResult.WATCH:
        prev = state_manager.get(patient_id)
        state_manager.set_watch(patient_id)
        if prev != PatientState.WATCH:
            await dispatch_state_change(
                patient_id=patient_id,
                new_state="WATCH",
                previous_state=prev.value,
                reason=f"Deviation {extra.get('deviation_sigma', 0):.1f}σ. Trajectory flat. Monitoring.",
            )
        return

    if result == RuleResult.SYNERA_STATE:
        alert_id = str(uuid4())
        trigger_vital = extra.get("trigger_vital", "heart_rate")
        trigger_value = extra.get("trigger_value", 0.0)
        baseline_value = extra.get("baseline_value", baseline_hr)
        deviation_sigma = extra.get("deviation_sigma", 0.0)
        second_derivative = extra.get("second_derivative", 0.0)
        motion_score = extra.get("motion_score", 0)

        try:
            clinical_brief = await asyncio.wait_for(
                asyncio.to_thread(
                    _build_rag_context_and_invoke,
                    patient, buffer, trigger_vital, trigger_value,
                    baseline_value, deviation_sigma, second_derivative, motion_score, alert_id,
                ),
                timeout=settings.RAG_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("RAG timeout for alert_id=%s", alert_id)
            clinical_brief = {
                "trigger_summary": f"{trigger_vital} trajectory alert. Rule-based fallback (RAG timeout).",
                "differential_diagnosis": [],
                "recommended_actions": [{"priority": 1, "action": "Assess patient at bedside", "rationale": "Alert fired."}],
                "drug_interaction_flags": [],
                "relevant_history": [],
                "sources": [],
                "confidence_note": "This is decision support. Verify clinically.",
                "generation_time_ms": 0,
            }

        vitals_snapshot = {
            "heart_rate": payload.heart_rate,
            "spo2": payload.spo2,
            "temperature": payload.temperature,
            "sys_bp_est": payload.sys_bp_est,
            "dia_bp_est": payload.dia_bp_est,
            "motion_score": payload.motion_score,
        }
        prev_state = state_manager.get(patient_id)
        state_manager.set_synera(patient_id)

        create_alert(
            alert_id=alert_id,
            patient_id=patient_id,
            trigger_vital=trigger_vital,
            trigger_value=trigger_value,
            baseline_value=baseline_value,
            deviation_sigma=deviation_sigma,
            second_derivative=second_derivative,
            motion_score=motion_score,
            vitals_snapshot=vitals_snapshot,
            rag_clinical_brief=clinical_brief,
            patient_state_before=prev_state.value,
            patient_state_after=PatientState.SYNERA_STATE.value,
            llm_provider=getattr(settings, "LLM_PROVIDER", "ollama"),
        )

        trigger_summary = clinical_brief.get("trigger_summary", f"{trigger_vital}={trigger_value} deviation {deviation_sigma:.1f}σ")
        await dispatch_synera_state(
            alert_id=alert_id,
            patient_id=patient_id,
            trigger_summary=trigger_summary,
            vitals_snapshot=vitals_snapshot,
            clinical_brief=clinical_brief,
            trigger_timestamp=datetime.utcnow().isoformat() + "Z",
        )
        return


class SyneraEngine:
    """Wrapper so event_bus can subscribe engine.handle_vital_payload(topic, payload)."""

    async def handle_vital_payload(self, topic: str, payload: dict):
        """Parse dict to VitalPayload and run process_vital_payload."""
        from backend.services.mqtt.parser import parse_mqtt_payload
        # payload may already be dict from event_bus
        if isinstance(payload, dict):
            import json
            # VitalPayload expects same shape as MQTT parser output
            from datetime import datetime
            recorded_at = payload.get("recorded_at")
            if isinstance(recorded_at, str):
                try:
                    recorded_at = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
                except Exception:
                    recorded_at = datetime.utcnow()
            elif recorded_at is None:
                recorded_at = datetime.utcnow()
            vp = VitalPayload(
                patient_id=payload.get("patient_id", ""),
                heart_rate=payload.get("heart_rate"),
                spo2=payload.get("spo2"),
                temperature=payload.get("temperature"),
                sys_bp_est=payload.get("sys_bp_est"),
                dia_bp_est=payload.get("dia_bp_est"),
                motion_score=payload.get("motion_score"),
                recorded_at=recorded_at,
            )
        else:
            vp = parse_mqtt_payload(payload if isinstance(payload, bytes) else str(payload))
            if not vp:
                return
        await process_vital_payload(vp)
