"""Local corpus indexing — turn a directory of PDFs / markdown / text / HTML
into a persistable HybridRetriever, queryable alongside (or instead of) web
search.

Why: SearXNG gives you live web coverage; a local corpus gives you your own
papers, notes, internal docs. Same retrieval primitives (`HybridRetriever`),
same two-stage rerank story (`CrossEncoderReranker`), just served from disk.

v1 scope (deliberately small):
  - Readers: .pdf (via pypdf), .md/.markdown/.txt (raw), .html/.htm (trafilatura).
  - Chunking: paragraph-aware; long paragraphs sliced with character overlap.
  - Persistence: one directory with `manifest.json` + `index.pkl`. Load rebuilds
    the BM25 index on the fly (rank_bm25 is fast; not worth pickling).
  - API surface: `CorpusIndex.build(dir) → .save(path) → .load(path) → .query(q, k)`.
  - Integration: the production research-assistant pipeline can attach a corpus
    via `LOCAL_CORPUS_PATH`; hits are merged into `evidence` with `corpus://` URLs.

Not yet shipped (deliberate, easy to add later):
  - Incremental indexing (rebuild whole thing from source on change).
  - Contextualize-chunks integration (contextual.py is available; opt-in
    at build time when that stops being experimental on small local models).
  - Reranker-at-load-time (the production pipeline already reranks at query
    time when `ENABLE_RERANK=1`; that covers corpus hits transparently).
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .hybrid import HybridRetriever, _tokenize
from .rag import Embedder, _openai_embedder

# Callable signature for a per-file text extractor. Returns a list of "pages"
# — one element for most formats, multiple for PDFs so we can keep page refs.
Extractor = Callable[[Path], list[str]]


# ── Extractors ───────────────────────────────────────────────────────

def _extract_text(path: Path) -> list[str]:
    return [path.read_text(encoding="utf-8", errors="replace")]


def _extract_pdf(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pypdf is required for PDF indexing. "
            "Install with: pip install 'pypdf>=4.0'"
        ) from e
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_html(path: Path) -> list[str]:
    try:
        import trafilatura  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "trafilatura is required for HTML indexing. "
            "Install with: pip install 'trafilatura>=1.12.0'"
        ) from e
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = trafilatura.extract(raw, include_comments=False, include_tables=False) or ""
    return [text]


EXTRACTORS: dict[str, Extractor] = {
    ".md": _extract_text,
    ".markdown": _extract_text,
    ".txt": _extract_text,
    ".pdf": _extract_pdf,
    ".html": _extract_html,
    ".htm": _extract_html,
}


# ── Chunking ─────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    """Paragraph-aware character-window chunker.

    Short paragraphs pass through as single chunks. Long paragraphs are
    sliced with overlap so context isn't lost at boundaries. Empty/whitespace
    paragraphs are dropped.
    """
    if not text or not text.strip():
        return []
    chunks: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
        i = 0
        while i < len(para):
            end = min(i + chunk_size, len(para))
            chunks.append(para[i:end])
            if end == len(para):
                break
            i = max(end - overlap, i + 1)
    return chunks


# ── Corpus index ─────────────────────────────────────────────────────

@dataclass
class CorpusChunk:
    """One searchable unit: chunk text + where it came from."""
    text: str
    source: str           # path relative to the source_dir, or external URI
    page: int | None = None
    chunk_idx: int = 0


@dataclass
class CorpusIndex:
    """Persistable HybridRetriever with source-tracked chunks.

    Usage:
        idx = CorpusIndex.build("./my_papers")
        idx.save("./my_papers_index")
        # later, possibly in a different process:
        idx = CorpusIndex.load("./my_papers_index")
        hits = idx.query("attention mechanism", k=5)  # → [(CorpusChunk, score), …]
    """

    chunks: list[CorpusChunk] = field(default_factory=list)
    retriever: HybridRetriever = field(default_factory=HybridRetriever)
    embed_model: str = ""
    version: int = 1

    # ── Build ────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        source_dir: str | Path,
        chunk_size: int = 800,
        overlap: int = 200,
        embedder: Embedder = _openai_embedder,
    ) -> "CorpusIndex":
        src = Path(source_dir).resolve()
        if not src.is_dir():
            raise ValueError(f"not a directory: {src}")
        idx = cls(retriever=HybridRetriever(embedder=embedder))
        idx.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")

        batch_texts: list[str] = []
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            extractor = EXTRACTORS.get(path.suffix.lower())
            if extractor is None:
                continue
            try:
                pages = extractor(path)
            except Exception as e:  # noqa: BLE001 — log & keep going
                print(f"[corpus] skip {path}: {e}")
                continue
            multi_page = len(pages) > 1
            for page_i, page_text in enumerate(pages):
                for chunk_i, chunk in enumerate(_chunk_text(page_text, chunk_size, overlap)):
                    idx.chunks.append(CorpusChunk(
                        text=chunk,
                        source=str(path.relative_to(src)),
                        page=(page_i if multi_page else None),
                        chunk_idx=chunk_i,
                    ))
                    batch_texts.append(chunk)

        if batch_texts:
            idx.retriever.add(batch_texts)
        return idx

    # ── Query ────────────────────────────────────────────────────

    def query(self, q: str, k: int = 5) -> list[tuple[CorpusChunk, float]]:
        hits = self.retriever.retrieve(q, k=k)
        by_text: dict[str, CorpusChunk] = {c.text: c for c in self.chunks}
        return [(by_text[t], s) for t, s in hits if t in by_text]

    # ── Persistence ──────────────────────────────────────────────

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps({
            "version": self.version,
            "embed_model": self.embed_model,
            "n_chunks": len(self.chunks),
            "n_sources": len({c.source for c in self.chunks}),
        }, indent=2))
        with open(out / "index.pkl", "wb") as f:
            pickle.dump({
                "chunks": [(c.text, c.source, c.page, c.chunk_idx) for c in self.chunks],
                "docs": self.retriever.docs,
                "vectors": self.retriever.vectors,
                "tokenized": self.retriever._tokenized,
                "version": self.version,
                "embed_model": self.embed_model,
            }, f)

    @classmethod
    def load(
        cls,
        in_dir: str | Path,
        embedder: Embedder = _openai_embedder,
    ) -> "CorpusIndex":
        with open(Path(in_dir) / "index.pkl", "rb") as f:
            data = pickle.load(f)
        r = HybridRetriever(embedder=embedder)
        r.docs = list(data["docs"])
        r.vectors = list(data["vectors"])
        r._tokenized = list(data["tokenized"])
        if r._tokenized:
            from rank_bm25 import BM25Okapi
            r._bm25 = BM25Okapi(r._tokenized)
        idx = cls(
            retriever=r,
            embed_model=data.get("embed_model", ""),
            version=data.get("version", 1),
        )
        idx.chunks = [
            CorpusChunk(text=t, source=s, page=p, chunk_idx=ci)
            for t, s, p, ci in data["chunks"]
        ]
        return idx


__all__ = ["CorpusChunk", "CorpusIndex", "EXTRACTORS"]
