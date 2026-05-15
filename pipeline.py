from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from chunker import chunk_document
from config import CHUNK_OVERLAP, CHUNK_SIZE, TOP_K
from generator import generate
from loader import load_document
from retriever import retrieve
from embedder import embed_chunks

console = Console()


def ingest(sources: list[str]) -> int:
    total = 0
    for source in sources:
        console.print(f"[bold cyan]Loading:[/] {source}")
        doc = load_document(source)
        chunks = chunk_document(doc, CHUNK_SIZE, CHUNK_OVERLAP)
        count = embed_chunks(chunks)
        console.print(f"   -> {count} chunks stored")

        total += count

    return total


def ask(query: str) -> str:
    console.print(f"\n[bold yellow]Query:[/] {query}")

    hits = retrieve(query, top_k=TOP_K)
    if not hits:
        return "No relevant documents found. Ingest some first."

    console.print(f"[dim]Retrieved {len(hits)} chunks[/dim]")
    for i, h in enumerate(hits, 1):
        src = h["metadata"].get("source", "?")
        dist = h["distance"]
        console.print(f"  [dim][{i}] {src} (dist={dist:.4f})[/dim]")

    answer = generate(query, hits)
    return answer


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        console.print("[red]Usage:[/]")
        console.print("  Ingest: python pipeline.py ingest file1.pdf file2.txt https://...")
        console.print("  Ask: python pipeline.py ask 'your qn here'")

        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        sources = sys.argv[2:]
        if not sources:
            console.print("[red]Provide at least one source to ingest.[/]")
            sys.exit(1)

        total = ingest(sources)
        console.print(Panel(f"[green]Done — {total} chunks ingested[/]"))

    elif command == "ask":
        query = " ".join(sys.argv[2:])
        if not query:
            console.print("[red]Provide a question.[/]")
            sys.exit(1)

        answer = ask(query)
        console.print(Panel(answer, title="Answer", border_style="green"))

    else:
        console.print(f"[red]Unknown command: {command}[/]")