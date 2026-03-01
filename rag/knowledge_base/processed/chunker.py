"""PDF → section-aware chunks (300-400 tokens). For synthetic seed, split by paragraph."""
import re
from typing import List


def chunk_text(text: str, min_tokens: int = 300, max_tokens: int = 400) -> List[dict]:
    """
    Split text into chunks. Token count approximate (4 chars ≈ 1 token).
    Returns list of {"text": str, "metadata": {}}.
    """
    chunks = []
    # Simple paragraph/sentence split
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    current = []
    current_len = 0
    target_min = min_tokens * 4
    target_max = max_tokens * 4
    for p in paragraphs:
        p_len = len(p) + 1
        if current_len + p_len > target_max and current:
            chunks.append({"text": "\n\n".join(current), "metadata": {}})
            current = []
            current_len = 0
        current.append(p)
        current_len += p_len
        if current_len >= target_min:
            chunks.append({"text": "\n\n".join(current), "metadata": {}})
            current = []
            current_len = 0
    if current:
        chunks.append({"text": "\n\n".join(current), "metadata": {}})
    return chunks
