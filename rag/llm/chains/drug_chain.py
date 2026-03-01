"""Drug interaction retrieval + check chain."""
from pathlib import Path
from rag.retrieval.retriever import search
from rag.llm.llm_client import get_llm


def run_drug_chain(medications_list: str, alert_summary: str, top_k: int = 3) -> str:
    """Retrieve pharmacology chunks, build prompt, invoke LLM."""
    chunks = search("medical_knowledge", "drug interaction " + medications_list, top_k=top_k, where={"clinical_domain": "pharmacology"})
    if not chunks:
        chunks = search("medical_knowledge", "drug interaction " + medications_list, top_k=top_k)
    ip_chunks = "\n\n".join([c.text for c in chunks])
    tpl = (Path(__file__).parent.parent / "prompts" / "drug_interaction.txt").read_text(encoding="utf-8")
    prompt = tpl.format(
        medications_list=medications_list,
        alert_summary=alert_summary,
        ip2022_chunks=ip_chunks or "No pharmacology excerpts found.",
    )
    llm = get_llm()
    out = llm.invoke(prompt)
    return out.content if hasattr(out, "content") else str(out)
