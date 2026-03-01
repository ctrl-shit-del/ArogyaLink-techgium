"""Retrieval precision@k measurement (stub)."""
def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / k
