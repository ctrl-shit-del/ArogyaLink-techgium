"""Thin wrapper over Supabase pgvector (replaces ChromaDB). For backward compatibility with rag/api.py."""

try:
    from rag.retrieval.supabase_vector_store import vector_store
except Exception:
    vector_store = None


def get_client():
    """No-op; Chroma client replaced by Supabase."""
    return None


def list_collections() -> list[str]:
    if vector_store:
        return vector_store.list_collection_names()
    return ["medical_knowledge", "clinical_cases"]


class _SupabaseCollectionProxy:
    """Proxy so .get(include=...) works like ChromaDB for rag/api.py."""

    def __init__(self, name: str):
        self.name = name

    def get(self, include=None):
        if not vector_store:
            return {"ids": [], "documents": [], "metadatas": []}
        return vector_store.get_chunks(self.name, limit=50)
    
    def query(self, *args, **kwargs):
        raise NotImplementedError("Use retriever.search() for Supabase pgvector")
    
    def add(self, *args, **kwargs):
        raise NotImplementedError("Use indexer or vector_store.insert_* for Supabase")


def get_or_create_collection(name: str):
    return _SupabaseCollectionProxy(name)
