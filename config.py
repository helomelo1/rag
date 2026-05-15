"""Configuration loader for the RAG project."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

# Core credentials
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Model configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Chunking configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval configuration
TOP_K = int(os.getenv("TOP_K", "5"))

# Storage configuration
CHROMA_PERSIST_DIR = os.getenv(
	"CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma")
)

__all__ = [
	"CHAT_MODEL",
	"CHROMA_PERSIST_DIR",
	"CHUNK_OVERLAP",
	"CHUNK_SIZE",
	"EMBED_MODEL",
	"HF_TOKEN",
	"PROJECT_ROOT",
	"TOP_K",
]