"""
Replaces Eclipse Mosquitto MQTT broker entirely.
Internal asyncio event bus for simulator → engine communication.
Same interface the MQTT subscriber used — engine.py doesn't need to change.
"""
import asyncio
import json
from typing import Callable, Awaitable

class InternalEventBus:
    """
    Lightweight async pub/sub.
    Publisher: mock_simulator.py calls bus.publish(topic, payload)
    Subscriber: engine.py calls bus.subscribe(topic, handler)

    For real wearable hardware in production:
    Replace this class with aiomqtt subscriber — same handler interface.
    The engine.py code doesn't change either way.
    """
    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    def subscribe(self, topic_pattern: str, handler: Callable[[str, dict], Awaitable[None]]):
        if topic_pattern not in self._subscribers:
            self._subscribers[topic_pattern] = []
        self._subscribers[topic_pattern].append(handler)

    async def publish(self, topic: str, payload: dict):
        await self._queue.put((topic, payload))

    async def run(self):
        """Start consuming queue. Call this as an asyncio task on startup."""
        while True:
            topic, payload = await self._queue.get()
            for pattern, handlers in self._subscribers.items():
                if self._matches(pattern, topic):
                    for handler in handlers:
                        asyncio.create_task(handler(topic, payload))
            self._queue.task_done()

    def _matches(self, pattern: str, topic: str) -> bool:
        if pattern == topic:
            return True
        pattern_parts = pattern.split("/")
        topic_parts = topic.split("/")
        if pattern_parts[-1] == "#":
            return topic_parts[:len(pattern_parts)-1] == pattern_parts[:-1]
        if len(pattern_parts) != len(topic_parts):
            return False
        return all(p == "+" or p == t for p, t in zip(pattern_parts, topic_parts))


# Singleton
bus = InternalEventBus()
