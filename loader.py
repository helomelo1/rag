from __future__ import annotations

import os
from urllib.parse import urlparse

import pypdf
import requests
from bs4 import BeautifulSoup


def load_pdf(path: str) -> dict:
    reader = pypdf.PdfReader(path)
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return {"text": "\n".join(pages), "source": path, "type": "pdf"}


def load_text(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    return {"text": text, "source": path, "type": "text"}


def load_url(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (RAG Pipeline) requests/2.0"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script and get readable text
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return {"text": text, "source": url, "type": "url"}


def load_document(source: str) -> dict:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return load_url(source)
    
    ext = os.path.splitext(source)[1].lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext in {".txt", ".md"}:
        return load_text(source)
    
    raise ValueError(f"Unsupported source type {source}")