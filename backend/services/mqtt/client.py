"""aiomqtt connection setup."""
import asyncio
import logging
from typing import AsyncIterator

import aiomqtt

from backend.config.mqtt import MQTT_BROKER, MQTT_PORT

logger = logging.getLogger(__name__)


async def create_mqtt_client() -> AsyncIterator[aiomqtt.Client]:
    """Context manager for MQTT client connection."""
    try:
        async with aiomqtt.Client(hostname=MQTT_BROKER, port=MQTT_PORT) as client:
            yield client
    except Exception as e:
        logger.exception("MQTT client error: %s", e)
        raise
