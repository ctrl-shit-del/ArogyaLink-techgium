"""BGE-M3 batch embedding. Uses instruction prefix for queries, raw text for documents."""
from typing import List
import os

# Optional: use sentence-transformers when available
def get_embedding_model():
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except ImportError:
        return None


_model = None


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed document chunks (no query prefix)."""
    global _model
    if _model is None:
        _model = get_embedding_model()
    if _model is None:
        # Fallback: random 1024-dim for dev without sentence-transformers
        return [[0.0] * 1024 for _ in texts]
    return _model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> List[float]:
    """Embed query with instruction prefix for BGE-M3."""
    global _model
    if _model is None:
        _model = get_embedding_model()
    prefix = "Represent this sentence for searching relevant passages: "
    if _model is None:
        return [0.0] * 1024
    return _model.encode([prefix + query], normalize_embeddings=True)[0].tolist()
