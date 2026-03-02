"""
drl/online_trainer.py
Background asyncio task that runs daily fine-tuning using Supabase alert_events.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Tuple
import numpy as np

from drl.agent import triage_agent
from drl.reward_calculator import compute_reward_from_supabase_row
from drl.state_builder import build_state_from_supabase_alert

logger = logging.getLogger("synera.drl.online_trainer")

TRAINING_INTERVAL_HOURS = 24
MIN_EXPERIENCES_TO_TRAIN = 10


async def online_training_loop(supabase_client) -> None:
    """Run every 24 hours: collect resolved alerts, compute rewards, fine-tune PPO."""
    logger.info("[DRL] Online training loop started. First run in 24 hours.")

    while True:
        await asyncio.sleep(TRAINING_INTERVAL_HOURS * 3600)

        try:
            logger.info("[DRL] Starting daily fine-tuning run...")
            experiences = await _collect_experiences(supabase_client)

            if len(experiences) < MIN_EXPERIENCES_TO_TRAIN:
                logger.info("[DRL] Only %s experiences — skipping (need %s)",
                            len(experiences), MIN_EXPERIENCES_TO_TRAIN)
                continue

            logger.info("[DRL] Fine-tuning on %s experiences...", len(experiences))
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: triage_agent.fine_tune(experiences))
            logger.info("[DRL] Daily fine-tuning complete.")

        except Exception as e:
            logger.error("[DRL] Online training error: %s", e, exc_info=True)


async def _collect_experiences(supabase_client) -> List[Tuple[np.ndarray, int, float]]:
    """Query Supabase for resolved alerts in last 24h; return (state, action, reward)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    try:
        response = (
            supabase_client.table("alert_events")
            .select("*")
            .gte("trigger_timestamp", cutoff)
            .execute()
        )
    except Exception as e:
        logger.warning("[DRL] Supabase query failed: %s", e)
        return []

    rows = response.data or []
    experiences = []

    for row in rows:
        reward = compute_reward_from_supabase_row(row)
        if reward is None:
            continue

        priority_tier = row.get("priority_tier_assigned", "ELEVATED")
        action_map = {"ELEVATED": 0, "URGENT": 1, "IMMEDIATE": 2}
        action = action_map.get(priority_tier, 0)

        patient_id = row.get("patient_id")
        patient_row = {}
        if patient_id:
            try:
                pr = supabase_client.table("patients").select("*").eq("patient_id", patient_id).single().execute()
                if pr.data:
                    patient_row = pr.data
            except Exception:
                pass

        state = build_state_from_supabase_alert(row, patient_row)
        experiences.append((state, action, reward))

    logger.info("[DRL] Collected %s resolved experiences from last 24h", len(experiences))
    return experiences
