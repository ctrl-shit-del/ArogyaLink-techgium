# Temporary: skip cross-encoder reranking, use similarity scores directly
# Cross-encoder requires model download which is slow on first run


def rerank(query: str, chunks: list, top_k: int = 6) -> list:
    """
    Passthrough reranker — sorts by similarity score already computed by pgvector.
    Replace with cross-encoder once model is cached locally.
    """
    if not chunks:
        return []
    sorted_chunks = sorted(chunks, key=lambda c: getattr(c, "similarity", 0.0), reverse=True)
    return sorted_chunks[:top_k]
