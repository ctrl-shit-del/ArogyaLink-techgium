"""ConnectionManager: broadcast to all connected WebSocket clients."""
import asyncio
import json
import logging
from typing import Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("WebSocket connected. Total: %d", len(self._connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass

    async def broadcast(self, message: dict) -> None:
        if not self._connections:
            return
        payload = json.dumps(message)
        dead = set()
        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(payload)
                except Exception as e:
                    logger.warning("WS send failed: %s", e)
                    dead.add(ws)
            for ws in dead:
                self._connections.discard(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


ws_manager = ConnectionManager()
