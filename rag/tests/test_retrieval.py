"""Vector search accuracy tests (require ChromaDB seeded)."""
import pytest
from rag.retrieval.chroma_store import list_collections
from rag.retrieval.retriever import search


def test_search_returns_list():
    # Without seed may be empty; just check API
    results = search("medical_knowledge", "tachycardia sepsis", top_k=3)
    assert isinstance(results, list)
