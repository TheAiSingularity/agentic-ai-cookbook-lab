"""core.rag.python — Python implementation of the shared retrieval primitives.

Public API (stable):
    Retriever, index, retrieve                    # v0 — naive dense baseline
    HybridRetriever, hybrid_index                 # v1 — BM25 + dense + RRF
    CrossEncoderReranker                          # v1 — second-stage rerank
    contextualize_chunks                          # v1 — Anthropic contextual chunks
    CorpusIndex, CorpusChunk                      # v1 — persistable local corpus
"""

from .contextual import contextualize_chunks, make_openai_llm
from .corpus import CorpusChunk, CorpusIndex
from .hybrid import HybridRetriever, hybrid_index
from .rag import Retriever, index, retrieve
from .rerank import CrossEncoderReranker

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
