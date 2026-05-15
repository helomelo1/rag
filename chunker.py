from __future__ import annotations

import re
from typing import Dict, List


def _split_on_boundaries(text: str) -> List[str]:
    # Try paragraph breaks, lines, sentences, breaks

    for sep in ["\n\n", "\n"]:
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        if len(parts) > 1:
            return parts
        
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) > 1:
        return sentences

    words = [w.strip() for w in text.split() if w.strip()]
    return words


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    units = _split_on_boundaries(text)
    chunks: List[str] = []
    current = ""

    for unit in units:
        # +1 for a space if needed
        next_len = len(current) + (1 if current else 0) + len(unit)
        if next_len <= chunk_size:
            current = f"{current} {unit}".strip()
        else:
            if current:
                chunks.append(current)
            current = unit

    if current:
        chunks.append(current)

    # Add overlap: prepend last N chars of previous chunk to next
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prefix = chunks[i - 1][-overlap:]
            overlapped.append(f"{prefix}{chunks[i]}")
        chunks = overlapped

    return chunks


def chunk_document(doc: Dict[str, str], chunk_size: int, overlap: int) -> List[Dict[str, str]]:
    chunks = split_text(doc["text"], chunk_size, overlap)
    total = len(chunks)
    return [
        {
            "text": chunk,
            "source": doc.get("source", ""),
            "type": doc.get("type", ""),
            "chunk_index": i,
            "total_chunks": total,
        }
        for i, chunk in enumerate(chunks)
    ]