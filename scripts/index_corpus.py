#!/usr/bin/env python3
"""Build, inspect, and query a `core.rag` local corpus index.

Turns a directory of PDFs / markdown / text / HTML into a persistable
`HybridRetriever` on disk. The resulting index can be queried directly from
the CLI, or attached to the research-assistant production pipeline via
`LOCAL_CORPUS_PATH`.

Usage:
    # Build an index (embeddings use OPENAI_BASE_URL + EMBED_MODEL)
    python scripts/index_corpus.py build ~/papers --out ~/papers.idx

    # Inspect
    python scripts/index_corpus.py info ~/papers.idx

    # Query
    python scripts/index_corpus.py query ~/papers.idx "attention mechanism" --k 5

Local embeddings on Mac with Ollama:
    OPENAI_BASE_URL=http://localhost:11434/v1 \\
    OPENAI_API_KEY=ollama \\
    EMBED_MODEL=nomic-embed-text \\
    python scripts/index_corpus.py build ~/papers --out ~/papers.idx

File types supported:
    .md / .markdown / .txt   raw text
    .pdf                      via pypdf (requires `pip install pypdf`)
    .html / .htm              via trafilatura

Unsupported files are silently skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.rag import CorpusIndex  # noqa: E402


def _build(args: argparse.Namespace) -> None:
    src = Path(args.source_dir).expanduser()
    out = Path(args.out).expanduser()
    print(f"[build] scanning {src} …")
    idx = CorpusIndex.build(
        src,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    n_sources = len({c.source for c in idx.chunks})
    print(f"[build] {len(idx.chunks)} chunks from {n_sources} sources")
    if not idx.chunks:
        print("[build] nothing to index (no supported files found)")
        return
    idx.save(out)
    print(f"[build] saved to {out}")


def _info(args: argparse.Namespace) -> None:
    manifest = json.loads((Path(args.index_dir).expanduser() / "manifest.json").read_text())
    print(json.dumps(manifest, indent=2))


def _query(args: argparse.Namespace) -> None:
    idx = CorpusIndex.load(Path(args.index_dir).expanduser())
    hits = idx.query(args.query, k=args.k)
    if not hits:
        print("(no hits)")
        return
    for chunk, score in hits:
        loc = chunk.source
        if chunk.page is not None:
            loc += f"#p{chunk.page}"
        loc += f"#c{chunk.chunk_idx}"
        preview = chunk.text[:240].replace("\n", " ")
        if len(chunk.text) > 240:
            preview += "…"
        print(f"[{score:.3f}] {loc}")
        print(f"  {preview}")
        print()


def main() -> None:
    p = argparse.ArgumentParser(
        prog="index_corpus",
        description="Build / inspect / query a core.rag corpus index.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build", help="Build an index from a directory")
    bp.add_argument("source_dir", help="Directory to scan (recursive)")
    bp.add_argument("--out", required=True, help="Output index directory")
    bp.add_argument("--chunk-size", type=int, default=800)
    bp.add_argument("--overlap", type=int, default=200)
    bp.set_defaults(func=_build)

    ip = sub.add_parser("info", help="Show index metadata")
    ip.add_argument("index_dir")
    ip.set_defaults(func=_info)

    qp = sub.add_parser("query", help="Query an existing index")
    qp.add_argument("index_dir")
    qp.add_argument("query")
    qp.add_argument("--k", type=int, default=5)
    qp.set_defaults(func=_query)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
