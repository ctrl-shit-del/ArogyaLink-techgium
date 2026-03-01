"""Extracts and formats source citations from retrieved chunks."""
from typing import List
from rag.retrieval.retriever import RetrievedChunk


def format_citations(chunks: List[RetrievedChunk]) -> List[dict]:
    """Return list of {document, section, relevance} from chunk metadata."""
    out = []
    for c in chunks:
        meta = c.metadata or {}
        out.append({
            "document": meta.get("source_label") or meta.get("source_document", "Unknown"),
            "section": meta.get("section", ""),
            "relevance": meta.get("keywords", "") or "Retrieved for this alert.",
        })
    return out
