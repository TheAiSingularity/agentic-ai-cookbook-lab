"""Tests for core.rag.python.corpus — uses a mocked embedder, no network.

Covers: paragraph-aware chunking, mixed-format directory indexing, save/load
round-trip, query returning source-tagged CorpusChunks, and graceful
skipping of unsupported / broken files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from core.rag import CorpusChunk, CorpusIndex  # noqa: E402
from core.rag.python.corpus import _chunk_text  # noqa: E402


def fake_embedder(batch: list[str]) -> list[list[float]]:
    """Deterministic 3-d embedding — length, word count, vowel count."""
    return [
        [float(len(s)), float(len(s.split())), float(sum(1 for c in s.lower() if c in "aeiou"))]
        for s in batch
    ]


# ── Chunking ─────────────────────────────────────────────────────────

def test_chunk_text_short_paragraphs_pass_through():
    text = "First para.\n\nSecond para.\n\nThird para."
    assert _chunk_text(text, chunk_size=800) == ["First para.", "Second para.", "Third para."]


def test_chunk_text_empty_and_whitespace_returns_empty():
    assert _chunk_text("") == []
    assert _chunk_text("   \n\n   ") == []


def test_chunk_text_splits_long_paragraph_with_overlap():
    text = "a" * 1000  # one paragraph, too long
    chunks = _chunk_text(text, chunk_size=400, overlap=100)
    assert len(chunks) > 1
    # Every chunk is under the cap.
    assert all(len(c) <= 400 for c in chunks)
    # Neighbors share `overlap` characters at the boundary.
    for a, b in zip(chunks, chunks[1:]):
        assert a[-100:] == b[:100]
    # Reassembling minus overlaps reconstructs the original.
    reassembled = chunks[0] + "".join(c[100:] for c in chunks[1:])
    assert reassembled == text


def test_chunk_text_progress_guaranteed_on_small_overlap_vs_chunk_size():
    # If overlap >= chunk_size, the naive loop would spin forever — we bound
    # progress to at least +1 char per iteration.
    chunks = _chunk_text("x" * 300, chunk_size=200, overlap=200)
    assert len(chunks) >= 2  # terminated, didn't hang


# ── Directory indexing ───────────────────────────────────────────────

def test_build_indexes_markdown_and_text(tmp_path: Path):
    (tmp_path / "a.md").write_text("# Title\n\nSome content about cats.\n\nMore about cats.")
    (tmp_path / "b.txt").write_text("A plain text file about dogs.")
    (tmp_path / "skip_me.bin").write_bytes(b"\x00\x01\x02")

    idx = CorpusIndex.build(tmp_path, embedder=fake_embedder)
    # 3 paragraphs from a.md + 1 from b.txt = 4 chunks
    assert len(idx.chunks) == 4
    sources = {c.source for c in idx.chunks}
    assert sources == {"a.md", "b.txt"}
    # Pages None because md/txt are single-page.
    assert all(c.page is None for c in idx.chunks)


def test_build_rejects_non_directory(tmp_path: Path):
    fp = tmp_path / "just_a_file.txt"
    fp.write_text("hi")
    with pytest.raises(ValueError):
        CorpusIndex.build(fp, embedder=fake_embedder)


def test_build_handles_empty_directory(tmp_path: Path):
    idx = CorpusIndex.build(tmp_path, embedder=fake_embedder)
    assert idx.chunks == []
    assert idx.retriever.docs == []


def test_build_skips_broken_file_logs_and_continues(tmp_path: Path, capsys):
    # We register a fake ".crash" extension that always throws.
    from core.rag.python import corpus as corpus_mod

    def boom(path: Path) -> list[str]:
        raise RuntimeError(f"fake failure on {path.name}")

    corpus_mod.EXTRACTORS[".crash"] = boom
    try:
        (tmp_path / "good.md").write_text("Keep me.")
        (tmp_path / "broken.crash").write_text("whatever")
        idx = CorpusIndex.build(tmp_path, embedder=fake_embedder)
        assert len(idx.chunks) == 1
        assert idx.chunks[0].source == "good.md"
        assert "skip" in capsys.readouterr().out.lower()
    finally:
        corpus_mod.EXTRACTORS.pop(".crash", None)


# ── Query ────────────────────────────────────────────────────────────

def test_query_returns_source_tagged_chunks(tmp_path: Path):
    (tmp_path / "cats.md").write_text(
        "Cats are small furry animals.\n\n"
        "BAAI bge-reranker-v2-m3 is the one we use.\n\n"
        "Quantum chromodynamics is unrelated."
    )
    idx = CorpusIndex.build(tmp_path, embedder=fake_embedder)
    hits = idx.query("BAAI bge-reranker-v2-m3", k=3)
    assert hits, "expected at least one hit"
    top_chunk, top_score = hits[0]
    assert isinstance(top_chunk, CorpusChunk)
    assert "BAAI" in top_chunk.text
    assert top_chunk.source == "cats.md"
    # Scores are descending.
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_query_on_empty_index_returns_empty(tmp_path: Path):
    idx = CorpusIndex.build(tmp_path, embedder=fake_embedder)
    assert idx.query("anything", k=5) == []


# ── Persistence ──────────────────────────────────────────────────────

def test_save_load_roundtrip_preserves_query_results(tmp_path: Path):
    (tmp_path / "a.md").write_text(
        "Alpha discusses cats.\n\n"
        "Beta discusses BAAI bge-reranker-v2-m3 in detail.\n\n"
        "Gamma discusses quantum gluons."
    )
    built = CorpusIndex.build(tmp_path, embedder=fake_embedder)
    out_dir = tmp_path / "_index"
    built.save(out_dir)

    # Manifest is human-readable.
    import json
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["n_chunks"] == len(built.chunks)
    assert manifest["n_sources"] == 1
    assert manifest["version"] == 1

    # Loading round-trips everything; query answers match.
    loaded = CorpusIndex.load(out_dir, embedder=fake_embedder)
    assert len(loaded.chunks) == len(built.chunks)
    assert {c.source for c in loaded.chunks} == {c.source for c in built.chunks}

    q = "BAAI bge-reranker-v2-m3"
    built_top = built.query(q, k=3)
    loaded_top = loaded.query(q, k=3)
    assert [c.text for c, _ in built_top] == [c.text for c, _ in loaded_top]


def test_load_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CorpusIndex.load(tmp_path / "does_not_exist", embedder=fake_embedder)


# ── Corpus-chunk identity ────────────────────────────────────────────

def test_chunk_ids_are_stable_across_rebuilds(tmp_path: Path):
    (tmp_path / "a.md").write_text("x" * 1000)  # long enough to trigger chunking
    a = CorpusIndex.build(tmp_path, chunk_size=300, overlap=50, embedder=fake_embedder)
    b = CorpusIndex.build(tmp_path, chunk_size=300, overlap=50, embedder=fake_embedder)
    assert [(c.source, c.chunk_idx, c.text) for c in a.chunks] == \
           [(c.source, c.chunk_idx, c.text) for c in b.chunks]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
