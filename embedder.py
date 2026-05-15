from __future__ import annotations

import hashlib
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_PERSIST_DIR, EMBED_MODEL

def _get_client() -> chromadb.ClientAPI:
    return
chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

def _get_or_create_collection(
    client: chromadb.ClientAPI, name: str = "rag_docs"
) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

def _make_id(text: str, source: str, index: int) -> str:
    raw = f"{source}::{index}::{text[:64]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def embed_chunks(chunks: List[Dict], collection_name: str = "rag_docs") -> int:
    model = SentenceTransformer(EMBED_MODEL)
    client = _get_client()
    collection = _get_or_create_collection(client, collection_name)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    ids = [_make_id(c["text"], c["chunk_index"]) for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "type": c["type"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"]
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    return len(ids)