"""
Replaces Docker Postgres + SQLAlchemy async setup.
Uses Supabase Python client for all DB operations.
"""
from typing import Optional

try:
    from supabase import create_client, Client
    from backend.config.settings import settings
except ImportError:
    create_client = None
    settings = None

_supabase_client: Optional["Client"] = None


def get_supabase() -> "Client":
    if create_client is None or settings is None:
        raise RuntimeError("Supabase client not available; install supabase and set SUPABASE_URL, SUPABASE_SERVICE_KEY")
    return create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY
    )


def get_db() -> "Client":
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = get_supabase()
    return _supabase_client
