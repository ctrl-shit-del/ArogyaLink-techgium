"""Cross-encoder reranking: top-10 → top-5/6."""
from typing import List
import os

from rag.retrieval.retriever import RetrievedChunk

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_model = None


def _get_reranker():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(RERANKER_MODEL)
        except Exception:
            _model = None
    return _model


def rerank(query: str, chunks: List[RetrievedChunk], top_k: int = 6) -> List[RetrievedChunk]:
    """Rerank chunks by relevance to query. Returns top_k."""
    if not chunks:
        return []
    enc = _get_reranker()
    if enc is None:
        return chunks[:top_k]
    pairs = [(query, c.text) for c in chunks]
    scores = enc.predict(pairs)
    indexed = list(zip(scores, chunks))
    indexed.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in indexed[:top_k]]
