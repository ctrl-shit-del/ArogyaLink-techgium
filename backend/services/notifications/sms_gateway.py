"""SMS stub for rural fallback (no internet)."""
import logging

logger = logging.getLogger(__name__)


async def send_sms(phone: str, message: str) -> bool:
    """Stub: log only. Real gateway would use Twilio/local SMS API."""
    logger.info("SMS stub: to=%s msg=%s", phone, message[:50])
    return True
