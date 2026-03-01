"""FastAPI app entry point. MQTT subscriber and WebSocket wired on startup."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.core.synera_engine.engine import process_vital_payload
from backend.services.mqtt.subscriber import subscribe_vitals
from backend.api.websocket.handlers import router as ws_router
from backend.api.routes import patients, alerts, vitals, medid, auth, health, rag

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start MQTT subscriber in background
    task = asyncio.create_task(subscribe_vitals(process_vital_payload))
    logger.info("MQTT subscriber started")
    yield
    # Shutdown: cancel subscriber
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Synera 2.0 RAG Clinical Co-Pilot",
    description="Physiological trajectory intelligence — backend only",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# API v1
app.include_router(patients.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")
app.include_router(vitals.router, prefix="/api/v1")
app.include_router(medid.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
# WebSocket
app.include_router(ws_router)


@app.get("/")
async def root():
    return {"service": "synera-backend", "version": "2.0.0"}
