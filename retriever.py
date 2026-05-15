from __future__ import annotations

from typing import Dict, List

from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, TOP_K
from embedder import _get_client, _get_or_create_collection

def retrieve(
    query: str,
    top_k: int = TOP_K,
    collection_name: str = "rag_docs"
) -> List[Dict]:
    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query]).tolist()

    client = _get_client()
    collection = _get_or_create_collection(client, collection_name)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
        )

    return hits