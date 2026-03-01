"""Health: Supabase DB, pgvector, LLM provider."""
import logging
from fastapi import APIRouter
from backend.config.settings import settings

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)


@router.get("")
async def health_check():
    status = {
        "status": "ok",
        "llm_provider": getattr(settings, "LLM_PROVIDER", "ollama"),
        "llm_model": settings.GROQ_MODEL if getattr(settings, "LLM_PROVIDER", "") == "groq" else settings.OLLAMA_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "database": "unknown",
        "vector_store": "unknown",
        "collections": {},
        "llm_status": "unknown",
    }

    try:
        from backend.config.database import get_db
        db = get_db()
        db.table("patients").select("patient_id").limit(1).execute()
        status["database"] = "connected (Supabase)"
    except Exception as e:
        status["database"] = f"error: {str(e)}"
        status["status"] = "degraded"

    try:
        from rag.retrieval.supabase_vector_store import vector_store
        counts = vector_store.get_collection_counts()
        status["vector_store"] = "connected (Supabase pgvector)"
        status["collections"] = counts
    except Exception as e:
        status["vector_store"] = f"error: {str(e)}"
        status["status"] = "degraded"

    try:
        if getattr(settings, "LLM_PROVIDER", "ollama") == "groq":
            if not getattr(settings, "GROQ_API_KEY", ""):
                raise ValueError("GROQ_API_KEY not set in .env")
            status["llm_status"] = "api_key_present"
        else:
            import httpx
            r = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
            status["llm_status"] = "ollama_running" if r.status_code == 200 else "ollama_not_responding"
    except Exception as e:
        status["llm_status"] = f"error: {str(e)}"

    return status
