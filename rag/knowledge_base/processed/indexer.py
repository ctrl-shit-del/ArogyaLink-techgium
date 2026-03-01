"""Writes chunks + embeddings to Supabase pgvector (replaces ChromaDB)."""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from rag.retrieval.supabase_vector_store import vector_store
from rag.knowledge_base.processed.embedder import embed_documents
from rag.knowledge_base.processed.seed_protocols import SYNTHETIC_PROTOCOL_CHUNKS
from rag.knowledge_base.processed.seed_cases import SYNTHETIC_CLINICAL_CASES


def index_protocols():
    for i, d in enumerate(SYNTHETIC_PROTOCOL_CHUNKS):
        meta = d.get("metadata", {})
        chunk = {
            "chunk_id": f"proto_{i}",
            "source_document": meta.get("source_document", "unknown"),
            "source_label": meta.get("source_label", ""),
            "section": meta.get("section", ""),
            "clinical_domain": meta.get("clinical_domain", ""),
            "keywords": meta.get("keywords", ""),
            "chunk_text": d["text"],
        }
        if isinstance(chunk.get("keywords"), str):
            chunk["keywords"] = [k.strip() for k in chunk["keywords"].split(",") if k.strip()] if chunk.get("keywords") else []
        elif not isinstance(chunk.get("keywords"), list):
            chunk["keywords"] = []
        embeddings = embed_documents([chunk["chunk_text"]])
        vector_store.insert_protocol_chunk(chunk, embeddings[0])
    print(f"  Inserted {len(SYNTHETIC_PROTOCOL_CHUNKS)} protocol chunks")


def index_cases():
    for i, d in enumerate(SYNTHETIC_CLINICAL_CASES):
        meta = d.get("metadata", {})
        case = {
            "case_id": f"case_{i}",
            "archetype": meta.get("archetype", ""),
            "patient_age_range": meta.get("patient_age_range", ""),
            "conditions": meta.get("conditions", []) or [],
            "trigger_vital": meta.get("trigger_vital", "heart_rate"),
            "outcome": meta.get("outcome", ""),
            "severity": meta.get("severity", "high"),
            "case_text": d["text"],
        }
        embeddings = embed_documents([case["case_text"]])
        vector_store.insert_case(case, embeddings[0])
    print(f"  Inserted {len(SYNTHETIC_CLINICAL_CASES)} clinical cases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", choices=["medical_knowledge", "clinical_cases", "all"], default="all")
    args = parser.parse_args()
    if args.collection in ("medical_knowledge", "all"):
        index_protocols()
    if args.collection in ("clinical_cases", "all"):
        index_cases()
    print("Done.")


if __name__ == "__main__":
    main()
