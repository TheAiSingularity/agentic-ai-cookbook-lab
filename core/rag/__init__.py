"""core.rag — shared retrieval primitives.

v0 (ships): Retriever, index, retrieve — naive dense baseline.
v1 (ships): HybridRetriever + CrossEncoderReranker + contextualize_chunks + CorpusIndex.

Public API is the Python implementation. A future `core.rag.rust` submodule
may land for perf-sensitive recipes; it will expose the same surface.
"""

from .python import (
    CorpusChunk,
    CorpusIndex,
    CrossEncoderReranker,
    HybridRetriever,
    Retriever,
    contextualize_chunks,
    hybrid_index,
    index,
    make_openai_llm,
    retrieve,
)

__all__ = [
    "Retriever",
    "index",
    "retrieve",
    "HybridRetriever",
    "hybrid_index",
    "CrossEncoderReranker",
    "contextualize_chunks",
    "make_openai_llm",
    "CorpusIndex",
    "CorpusChunk",
]
__version__ = "0.1.0"
