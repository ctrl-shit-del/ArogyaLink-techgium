"""Vector search: delegates to Supabase pgvector (replaces ChromaDB)."""
from dataclasses import dataclass
from typing import List, Optional

from rag.knowledge_base.processed.embedder import embed_query

try:
    from rag.retrieval.supabase_vector_store import vector_store, RetrievedChunk
except Exception:
    vector_store = None

    @dataclass
    class RetrievedChunk:
        id: str = ""
        text: str = ""
        metadata: dict = None
        distance: float = 0.0
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}


def search(
    collection: str,
    query: str,
    top_k: int = 5,
    where: Optional[dict] = None,
    query_embedding: Optional[List[float]] = None,
) -> List:
    """Search protocols or cases via Supabase pgvector. Returns list of RetrievedChunk (id, text, metadata, distance)."""
    if query_embedding is None:
        query_embedding = embed_query(
            f"Represent this sentence for searching relevant passages: {query}"
        )
    if vector_store is None:
        return []

    if collection == "medical_knowledge":
        domain_filter = where.get("clinical_domain") if where else None
        return vector_store.search_protocols(query_embedding, top_k=top_k, domain_filter=domain_filter)
    if collection == "clinical_cases":
        vital_filter = where.get("trigger_vital") if where else None
        return vector_store.search_cases(query_embedding, top_k=top_k, vital_filter=vital_filter)
    return []
