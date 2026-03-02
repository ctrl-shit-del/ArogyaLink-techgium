"""
SYNERA 2.0 — Single entry point (no Docker).
Run:  pip install -r requirements.txt && python run.py
Or:   uvicorn run:app --reload --port 8000
"""
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config.settings import settings
from backend.api.routes import patients, alerts, vitals, medid, auth, health, rag
from backend.api.websocket.handlers import router as ws_router
from backend.services.event_bus import bus
from backend.core.synera_engine.engine import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "cohere")
    if embedding_provider == "cohere":
        # Verify Cohere API key is set
        if not os.getenv("COHERE_API_KEY"):
            raise ValueError("COHERE_API_KEY not set in .env")
        print("✅ Embedding provider: Cohere API (no local model needed)")
    else:
        print("Loading embedding model (BGE-M3)... this takes ~15s on first run")
        from rag.knowledge_base.processed.embedder import get_embedding_model
        get_embedding_model()
        print("✅ Embedding model ready (local BGE-M3)")

    # Subscribe engine to internal event bus (for same-process MQTT-style clients)
    bus.subscribe("synera/patient/+/vitals", engine.handle_vital_payload)
    bus_task = asyncio.create_task(bus.run())

    print("✅ Synera engine started")
    print(f"✅ LLM provider: {settings.LLM_PROVIDER}")
    print("✅ Event bus running")
    print("📡 Ready. Start simulator: python scripts/data_gen/mock_simulator.py")
    yield
    bus_task.cancel()
    try:
        await bus_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Synera 2.0 — Clinical Co-Pilot API",
    description="RAG-powered physiological trajectory intelligence",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(health.router, prefix="/api/v1")
app.include_router(patients.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")
app.include_router(vitals.router, prefix="/api/v1")
app.include_router(vitals.router_ingest, prefix="/api/v1")
app.include_router(medid.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(ws_router)


@app.get("/")
async def root():
    return {"service": "synera-backend", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "run:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
    )
