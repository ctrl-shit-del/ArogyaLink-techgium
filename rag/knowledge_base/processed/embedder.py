import os
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cohere")


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    if EMBEDDING_PROVIDER == "cohere":
        return _cohere_embed([text], input_type="search_query")[0]
    else:
        return _local_embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document chunks."""
    if EMBEDDING_PROVIDER == "cohere":
        return _cohere_embed(texts, input_type="search_document")
    else:
        return _local_embed_documents(texts)


def _cohere_embed(texts: list[str], input_type: str) -> list[list[float]]:
    import cohere
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    model = os.getenv("COHERE_EMBEDDING_MODEL", "embed-multilingual-v3.0")
    response = co.embed(
        texts=texts,
        model=model,
        input_type=input_type,   # "search_query" for queries, "search_document" for chunks
        embedding_types=["float"]
    )
    return [list(e) for e in response.embeddings.float_]


def _local_embed_query(text: str) -> list[float]:
    """Fallback: local BGE-M3 if Cohere unavailable."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return model.encode(prefixed, normalize_embeddings=True).tolist()


def _local_embed_documents(texts: list[str]) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    return model.encode(texts, normalize_embeddings=True).tolist()


# Keep backwards compatibility
def get_embedding_model():
    """Legacy function — returns None when using Cohere (no local model needed)."""
    if EMBEDDING_PROVIDER == "cohere":
        return None
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
