"""Application settings via pydantic-settings. Loads from .env."""
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Supabase (replaces PostgreSQL + ChromaDB + Redis)
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_KEY: str = ""
    DATABASE_URL: str = ""

    # LLM toggle
    LLM_PROVIDER: Literal["groq", "ollama"] = "groq"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Embeddings (always local)
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # RAG
    RAG_TIMEOUT_SECONDS: float = 3.0
    RETRIEVAL_TOP_K_PROTOCOLS: int = 8
    RETRIEVAL_TOP_K_CASES: int = 4
    RETRIEVAL_FINAL_TOP_K: int = 6

    # Vector table names
    VECTOR_TABLE_PROTOCOLS: str = "medical_knowledge"
    VECTOR_TABLE_CASES: str = "clinical_cases"

    # Alert thresholds
    ARTIFACT_REJECT_HR_DELTA: float = 40.0
    ARTIFACT_REJECT_SPO2_DELTA: float = 5.0
    EXERTION_MOTION_THRESHOLD: int = 4
    EXERTION_HR_ELEVATION: float = 15.0
    SYNERA_STATE_SIGMA_THRESHOLD: float = 1.5
    SYNERA_STATE_MOTION_MAX: int = 2
    SYNERA_ACCELERATION_WINDOW: int = 3

    # App
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    API_KEY: str = "synera-dev-key"


settings = Settings()
