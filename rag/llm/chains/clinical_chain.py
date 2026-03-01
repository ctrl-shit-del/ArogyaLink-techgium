"""LangChain chain: retrieve → prompt → LLM (Groq or Ollama via llm_client)."""
from rag.llm.llm_client import get_llm


def invoke_clinical_chain(prompt: str) -> str:
    """Sync invoke: prompt → LLM → raw string. Uses LLM_PROVIDER from .env."""
    llm = get_llm()
    out = llm.invoke(prompt)
    return out.content if hasattr(out, "content") else str(out)
