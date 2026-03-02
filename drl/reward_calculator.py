"""
drl/reward_calculator.py
Computes the reward signal from a clinician acknowledgement event.
Called when POST /api/v1/alerts/{alert_id}/acknowledge fires.
"""

from typing import Optional
from datetime import datetime, timezone

REWARD_ATTENDED_5MIN   = +1.0
REWARD_ATTENDED_15MIN  = +0.5
REWARD_DISMISSED       = -0.5
REWARD_MISSED_CRITICAL = -1.0
REWARD_TIMEOUT         = -0.3


def compute_reward(
    alert_fired_at: datetime,
    acknowledged_at: Optional[datetime],
    was_dismissed: bool,
    patient_later_deteriorated: bool,
) -> float:
    """
    Compute the reward signal for one alert event.
    """
    if was_dismissed and patient_later_deteriorated:
        return REWARD_MISSED_CRITICAL
    if was_dismissed:
        return REWARD_DISMISSED
    if acknowledged_at is None:
        return REWARD_TIMEOUT

    response_minutes = (acknowledged_at - alert_fired_at).total_seconds() / 60.0
    if response_minutes <= 5.0:
        return REWARD_ATTENDED_5MIN
    elif response_minutes <= 15.0:
        return REWARD_ATTENDED_15MIN
    else:
        return 0.0


def compute_reward_from_supabase_row(alert_row: dict) -> Optional[float]:
    """
    Compute reward from a raw Supabase alert_events row.
    Returns None if the alert has not yet been resolved (still pending).

    Uses columns: trigger_timestamp, acknowledgement_timestamp,
    alert_dismissed, patient_deteriorated_after_dismissal.
    """
    fired_str = alert_row.get("trigger_timestamp") or alert_row.get("fired_at")
    ack_str   = alert_row.get("acknowledgement_timestamp") or alert_row.get("acknowledged_at")
    dismissed = alert_row.get("alert_dismissed", alert_row.get("dismissed", False))
    deteriorated = alert_row.get("patient_deteriorated_after_dismissal", alert_row.get("patient_deteriorated", False))

    if not dismissed and ack_str is None and not deteriorated:
        return None

    try:
        if isinstance(fired_str, str):
            fired_at = datetime.fromisoformat(fired_str.replace("Z", "+00:00"))
        else:
            fired_at = fired_str or datetime.now(timezone.utc)
        if fired_at.tzinfo is None:
            fired_at = fired_at.replace(tzinfo=timezone.utc)
        ack_at = None
        if ack_str:
            if isinstance(ack_str, str):
                ack_at = datetime.fromisoformat(ack_str.replace("Z", "+00:00"))
            else:
                ack_at = ack_str
            if ack_at and ack_at.tzinfo is None:
                ack_at = ack_at.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return REWARD_TIMEOUT

    return compute_reward(
        alert_fired_at=fired_at,
        acknowledged_at=ack_at,
        was_dismissed=dismissed,
        patient_later_deteriorated=deteriorated,
    )
