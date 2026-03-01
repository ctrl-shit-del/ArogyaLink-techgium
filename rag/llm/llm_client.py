"""
LLM provider toggle. Controlled entirely by LLM_PROVIDER in .env.
LLM_PROVIDER=groq   → Uses Groq API (fast, free tier, requires GROQ_API_KEY)
LLM_PROVIDER=ollama → Uses local Ollama (run: ollama serve in a terminal)

All other code calls get_llm() — provider is invisible to the pipeline.
"""
try:
    from backend.config.settings import settings
except ImportError:
    settings = None

_llm_instance = None


def get_llm_client():
    if settings is None:
        raise ValueError("Settings not available")
    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.1,
            max_tokens=2000,
        )
    elif settings.LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.1,
        )
    else:
        raise ValueError(
            f"LLM_PROVIDER='{settings.LLM_PROVIDER}' is not valid. "
            "Set to 'groq' or 'ollama' in .env"
        )


def get_provider_name() -> str:
    return getattr(settings, "LLM_PROVIDER", "ollama") if settings else "ollama"


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = get_llm_client()
    return _llm_instance
