"""Sends WebSocket event when Synera State fires."""
import logging
from datetime import datetime
from typing import Any

from backend.api.websocket.manager import ws_manager

logger = logging.getLogger(__name__)


async def dispatch_synera_state(
    alert_id: str,
    patient_id: str,
    trigger_summary: str,
    vitals_snapshot: dict,
    clinical_brief: dict,
    priority_tier: str = "ELEVATED",
    drl_confidence: float | None = None,
    trigger_timestamp: str | None = None,
) -> None:
    """Emit SYNERA_STATE event to all WebSocket clients (IMMEDIATE/URGENT/ELEVATED from DRL)."""
    payload = {
        "event_type": "SYNERA_STATE",
        "alert_id": alert_id,
        "patient_id": patient_id,
        "priority_tier": priority_tier,
        "drl_confidence": drl_confidence,
        "trigger_timestamp": trigger_timestamp or datetime.utcnow().isoformat() + "Z",
        "trigger_summary": trigger_summary,
        "vitals_snapshot": vitals_snapshot,
        "clinical_brief": clinical_brief,
        "requires_acknowledgement": True,
    }
    await ws_manager.broadcast(payload)
    logger.info("Dispatched SYNERA_STATE alert_id=%s patient_id=%s", alert_id, patient_id)


async def dispatch_state_change(
    patient_id: str,
    new_state: str,
    previous_state: str,
    reason: str,
) -> None:
    payload = {
        "event_type": "STATE_CHANGE",
        "patient_id": patient_id,
        "new_state": new_state,
        "previous_state": previous_state,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    await ws_manager.broadcast(payload)


async def dispatch_exertion_logged(patient_id: str, motion_score: int, hr_elevation: float) -> None:
    payload = {
        "event_type": "EXERTION_LOGGED",
        "patient_id": patient_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "motion_score": motion_score,
        "hr_elevation": hr_elevation,
    }
    await ws_manager.broadcast(payload)
