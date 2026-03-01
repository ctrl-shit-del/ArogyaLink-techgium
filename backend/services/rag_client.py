"""HTTP client to call RAG service for ClinicalBrief generation."""
import logging
from typing import Optional
import httpx

from backend.config.settings import settings

logger = logging.getLogger(__name__)
RAG_TIMEOUT = settings.rag_timeout_seconds


async def generate_brief_remote(alert_context_dict: dict, alert_id: str = "") -> Optional[dict]:
    """POST alert context to RAG service; returns ClinicalBrief dict or None."""
    url = f"{settings.rag_service_url.rstrip('/')}/api/v1/generate_brief"
    if alert_id:
        url += f"?alert_id={alert_id}"
    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:
            r = await client.post(url, json=alert_context_dict)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.exception("RAG service call failed: %s", e)
        return None
