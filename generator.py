from __future__ import annotations

from typing import Dict, List

from huggingface_hub import InferenceClient

from config import CHAT_MODEL, HF_TOKEN

SYSTEM_PROMPT = (
    "You are a helpful assisstant. Answer the user's question using ONLY "
    "the provided context. If the context doesn't contain enough information, "
    "say so clearly. Cite the source when possible."
)

def _build_prompt(query: str, context_chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        context_parts.append(f"[{i}] (source: {source})\n{chunk['text']}")

    context_block = "\n\n".join(context_parts)

    return (
        f"Context:\n{context_block}\n\n"
        f"---\n"
        f"Question: {query}\n"
        f"Answer:"
    )

def generate(query: str, context_chunks: List[Dict]) -> str:
    client = InferenceClient(token=HF_TOKEN)
    prompt = _build_prompt(query, context_chunks)

    response = client.chat_completion(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()