from __future__ import annotations

import shlex
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from chunker import chunk_document
from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, CHAT_MODEL, TOP_K
from embedder import embed_chunks, _get_client, _get_or_create_collection
from generator import generate
from loader import load_document
from retriever import retrieve

console = Console()

BANNER = r"""
[bold cyan]
  ╦═╗  ╔═╗  ╔═╗
  ╠╦╝  ╠═╣  ║ ╦
  ╩╚═  ╩ ╩  ╚═╝[/bold cyan]
  [dim]Retrieval-Augmented Generation Pipeline[/dim]
"""

COMMANDS_HELP = {
    "ingest <source> ...": "Load files or URLs into the knowledge base",
    "ask <question>":      "Ask a question over your ingested documents",
    "status":              "Show what's currently in the knowledge base",
    "clear":               "Wipe the entire knowledge base",
    "help":                "Show this help message",
    "quit / exit":         "Exit the pipeline",
}


# ── Core actions ────────────────────────────────────────────────


def do_ingest(sources: list[str]) -> None:
    if not sources:
        console.print("[red]  ✗ Provide at least one source (file path or URL).[/]")
        return

    total = 0
    for source in sources:
        try:
            console.print(f"  [bold cyan]↓[/] Loading [underline]{source}[/] …")
            doc = load_document(source)
            chunks = chunk_document(doc, CHUNK_SIZE, CHUNK_OVERLAP)
            count = embed_chunks(chunks)
            console.print(f"    [green]✓[/] {count} chunks stored")
            total += count
        except Exception as e:
            console.print(f"    [red]✗ Failed:[/] {e}")

    console.print(f"\n  [bold green]Done — {total} total chunks ingested.[/]")


def do_ask(query: str) -> None:
    if not query.strip():
        console.print("[red]  ✗ Provide a question.[/]")
        return

    console.print(f"\n  [bold yellow]?[/] {query}\n")

    with console.status("[dim]Searching knowledge base…[/]", spinner="dots"):
        hits = retrieve(query, top_k=TOP_K)

    if not hits:
        console.print("  [red]No relevant documents found. Ingest some first.[/]")
        return

    # Show sources
    src_table = Table(
        show_header=True, header_style="bold dim", box=box.SIMPLE,
        padding=(0, 1), pad_edge=False,
    )
    src_table.add_column("#", style="dim", width=3)
    src_table.add_column("Source", style="cyan")
    src_table.add_column("Distance", justify="right", style="dim")

    for i, h in enumerate(hits, 1):
        src = h["metadata"].get("source", "?")
        # Truncate long source paths
        display_src = src if len(src) <= 60 else f"…{src[-57:]}"
        src_table.add_row(str(i), display_src, f"{h['distance']:.4f}")

    console.print(src_table)

    with console.status("[dim]Generating answer…[/]", spinner="dots"):
        answer = generate(query, hits)

    console.print()
    console.print(Panel(
        Markdown(answer),
        title="[bold green]Answer[/]",
        border_style="green",
        padding=(1, 2),
    ))


def do_status() -> None:
    client = _get_client()
    collection = _get_or_create_collection(client)
    count = collection.count()

    if count == 0:
        console.print("  [dim]Knowledge base is empty. Use[/] [bold]ingest[/] [dim]to add documents.[/]")
        return

    # Peek at stored sources
    peek = collection.peek(limit=min(count, 20))
    sources = set()
    if peek.get("metadatas"):
        for m in peek["metadatas"]:
            if m and m.get("source"):
                sources.add(m["source"])

    table = Table(
        title="Knowledge Base",
        box=box.ROUNDED,
        show_lines=False,
        title_style="bold cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Total chunks", str(count))
    table.add_row("Embed model", EMBED_MODEL)
    table.add_row("Chat model", CHAT_MODEL)
    table.add_row("Chunk size / overlap", f"{CHUNK_SIZE} / {CHUNK_OVERLAP}")
    table.add_row("Top-K retrieval", str(TOP_K))

    console.print()
    console.print(table)

    if sources:
        console.print(f"\n  [bold]Sources indexed ({len(sources)}):[/]")
        for s in sorted(sources):
            display = s if len(s) <= 70 else f"…{s[-67:]}"
            console.print(f"    [dim]•[/] {display}")
    console.print()


def do_clear() -> None:
    client = _get_client()
    try:
        client.delete_collection("rag_docs")
        console.print("  [green]✓ Knowledge base cleared.[/]")
    except Exception:
        console.print("  [dim]Nothing to clear.[/]")


def show_help() -> None:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold",
        padding=(0, 2),
    )
    table.add_column("Command", style="bold cyan")
    table.add_column("Description")

    for cmd, desc in COMMANDS_HELP.items():
        table.add_row(cmd, desc)

    console.print()
    console.print(table)


# ── REPL ────────────────────────────────────────────────────────


def repl() -> None:
    console.print(BANNER)
    console.print("  Type [bold cyan]help[/] for commands, [bold cyan]quit[/] to exit.\n")

    while True:
        try:
            raw = console.input("[bold magenta]rag >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not raw:
            continue

        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = raw.split()

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break
        elif cmd == "help":
            show_help()
        elif cmd == "ingest":
            do_ingest(args)
        elif cmd == "ask":
            do_ask(" ".join(args))
        elif cmd == "status":
            do_status()
        elif cmd == "clear":
            do_clear()
        else:
            console.print(f"  [red]Unknown command:[/] {cmd}. Type [bold]help[/] for usage.")

        console.print()


# ── Entry point ─────────────────────────────────────────────────


if __name__ == "__main__":
    # If args are passed, run in one-shot mode (backward compatible)
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "ingest":
            do_ingest(sys.argv[2:])
        elif cmd == "ask":
            do_ask(" ".join(sys.argv[2:]))
        elif cmd == "status":
            do_status()
        elif cmd == "clear":
            do_clear()
        else:
            console.print(f"[red]Unknown command: {cmd}[/]")
            sys.exit(1)
    else:
        # No args → interactive mode
        repl()