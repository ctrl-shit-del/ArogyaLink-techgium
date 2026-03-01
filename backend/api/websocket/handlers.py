"""WebSocket route handlers."""
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.websocket.manager import ws_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Optional: handle client pings or commands
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        await ws_manager.disconnect(websocket)
