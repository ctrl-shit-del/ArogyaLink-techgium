"""
Replaces ChromaDB entirely.
Uses Supabase pgvector via the RPC functions defined in schema.sql.
Same interface as the old retriever — rag_pipeline/clinical_brief_generator use retriever which we'll point here.
"""
from typing import Optional
from dataclasses import dataclass, field

try:
    from supabase import create_client, Client
    from backend.config.settings import settings
except ImportError:
    create_client = None
    settings = None


@dataclass
class RetrievedChunk:
    """Compatible with existing retriever.RetrievedChunk (id, text, metadata, distance)."""
    chunk_id: str
    source_document: str
    source_label: str
    section_title: str
    clinical_domain: str
    chunk_text: str
    similarity: float
    collection: str
    keywords: list = field(default_factory=list)
    archetype: Optional[str] = None
    outcome: Optional[str] = None

    @property
    def id(self) -> str:
        return self.chunk_id

    @property
    def text(self) -> str:
        return self.chunk_text

    @property
    def metadata(self) -> dict:
        m = {
            "source_document": self.source_document,
            "source_label": self.source_label,
            "section": self.section_title,
            "clinical_domain": self.clinical_domain,
        }
        if self.keywords:
            m["keywords"] = self.keywords
        if self.collection == "clinical_cases":
            m["source_document"] = "clinical_cases"
            m["archetype"] = self.archetype
            m["outcome"] = self.outcome
        return m

    @property
    def distance(self) -> float:
        return 1.0 - self.similarity


class SupabaseVectorStore:
    def __init__(self):
        if create_client is None or settings is None:
            self.client = None
            return
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )

    def search_protocols(
        self,
        query_embedding: list[float],
        top_k: int = 8,
        domain_filter: Optional[str] = None
    ) -> list[RetrievedChunk]:
        if not self.client:
            return []
        try:
            response = self.client.rpc(
                "search_medical_knowledge",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k,
                    "domain_filter": domain_filter
                }
            ).execute()
        except Exception:
            return []
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                source_document=row["source_document"],
                source_label=row.get("source_label") or "",
                section_title=row.get("section_title") or "",
                clinical_domain=row.get("clinical_domain") or "",
                chunk_text=row["chunk_text"],
                similarity=float(row.get("similarity", 0)),
                collection="medical_knowledge",
                keywords=list(row.get("keywords") or [])
            )
            for row in (response.data or [])
        ]

    def search_cases(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        vital_filter: Optional[str] = None
    ) -> list[RetrievedChunk]:
        if not self.client:
            return []
        try:
            response = self.client.rpc(
                "search_clinical_cases",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k,
                    "vital_filter": vital_filter
                }
            ).execute()
        except Exception:
            return []
        return [
            RetrievedChunk(
                chunk_id=row["case_id"],
                source_document="clinical_cases",
                source_label="Clinical Case Example",
                section_title=row.get("archetype") or "",
                clinical_domain="case_example",
                chunk_text=row["case_text"],
                similarity=float(row.get("similarity", 0)),
                collection="clinical_cases",
                archetype=row.get("archetype"),
                outcome=row.get("outcome")
            )
            for row in (response.data or [])
        ]

    def insert_protocol_chunk(self, chunk: dict, embedding: list[float]):
        if not self.client:
            return
        self.client.table("medical_knowledge").upsert({
            "chunk_id": chunk.get("chunk_id", ""),
            "source_document": chunk.get("source_document", ""),
            "source_label": chunk.get("source_label", ""),
            "section_title": chunk.get("section", ""),
            "clinical_domain": chunk.get("clinical_domain", ""),
            "keywords": chunk.get("keywords", []),
            "chunk_text": chunk.get("chunk_text", ""),
            "embedding": embedding
        }, on_conflict="chunk_id").execute()

    def insert_case(self, case: dict, embedding: list[float]):
        if not self.client:
            return
        self.client.table("clinical_cases").upsert({
            "case_id": case.get("case_id", ""),
            "archetype": case.get("archetype", ""),
            "patient_age_range": case.get("patient_age_range", ""),
            "conditions": case.get("conditions", []),
            "trigger_vital": case.get("trigger_vital", ""),
            "outcome": case.get("outcome", ""),
            "severity": case.get("severity", "high"),
            "case_text": case.get("case_text", ""),
            "embedding": embedding
        }, on_conflict="case_id").execute()

    def get_collection_counts(self) -> dict:
        if not self.client:
            return {"medical_knowledge": 0, "clinical_cases": 0}
        try:
            r1 = self.client.table("medical_knowledge").select("id").execute()
            r2 = self.client.table("clinical_cases").select("id").execute()
            return {
                "medical_knowledge": len(r1.data) if r1.data else 0,
                "clinical_cases": len(r2.data) if r2.data else 0
            }
        except Exception:
            return {"medical_knowledge": 0, "clinical_cases": 0}

    def list_collection_names(self) -> list[str]:
        return list(self.get_collection_counts().keys())

    def get_chunks(self, collection: str, limit: int = 50) -> dict:
        """Return {ids, documents, metadatas} for compatibility with Chroma-style .get()."""
        if not self.client:
            return {"ids": [], "documents": [], "metadatas": []}
        try:
            if collection == "medical_knowledge":
                r = self.client.table("medical_knowledge").select("chunk_id, chunk_text, source_document, source_label, section_title, clinical_domain").limit(limit).execute()
                data = r.data or []
                return {
                    "ids": [d.get("chunk_id", "") for d in data],
                    "documents": [d.get("chunk_text", "") for d in data],
                    "metadatas": [{"source_document": d.get("source_document"), "section": d.get("section_title"), "clinical_domain": d.get("clinical_domain")} for d in data],
                }
            if collection == "clinical_cases":
                r = self.client.table("clinical_cases").select("case_id, case_text, archetype, trigger_vital, outcome").limit(limit).execute()
                data = r.data or []
                return {
                    "ids": [d.get("case_id", "") for d in data],
                    "documents": [d.get("case_text", "") for d in data],
                    "metadatas": [{"archetype": d.get("archetype"), "trigger_vital": d.get("trigger_vital"), "outcome": d.get("outcome")} for d in data],
                }
        except Exception:
            pass
        return {"ids": [], "documents": [], "metadatas": []}


# Singleton
vector_store = SupabaseVectorStore()
