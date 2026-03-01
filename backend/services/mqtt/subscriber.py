"""Async subscribe loop: synera/patient/+/vitals. Parses and yields VitalPayload."""
import asyncio
import logging
from typing import AsyncIterator, Callable, Awaitable

import aiomqtt

from backend.config.mqtt import MQTT_BROKER, MQTT_PORT, MQTT_QOS, SYNERA_TOPIC_PATTERN
from backend.services.mqtt.parser import parse_mqtt_payload
from backend.models.schemas.vitals import VitalPayload

logger = logging.getLogger(__name__)


async def subscribe_vitals(
    on_message: Callable[[VitalPayload], Awaitable[None]],
) -> None:
    """Subscribe to synera/patient/+/vitals and call on_message for each parsed payload."""
    while True:
        try:
            async with aiomqtt.Client(hostname=MQTT_BROKER, port=MQTT_PORT) as client:
                async with client.messages() as msgs:
                    await client.subscribe(SYNERA_TOPIC_PATTERN, qos=MQTT_QOS)
                    logger.info("Subscribed to %s", SYNERA_TOPIC_PATTERN)
                    async for msg in msgs:
                        payload = parse_mqtt_payload(msg.payload)
                        if payload:
                            await on_message(payload)
                        else:
                            logger.warning("Failed to parse MQTT payload from %s", msg.topic)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("MQTT subscriber error: %s", e)
            await asyncio.sleep(5)
