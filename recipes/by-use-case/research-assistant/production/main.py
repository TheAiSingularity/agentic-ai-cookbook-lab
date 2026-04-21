"""Research-assistant production — **thin shim** over `engine.core`.

The pipeline code moved to `engine/core/` during the research-engine
rework. This file stays at its original path so existing bookmarks,
recipe READMEs, and the test file keep working without URL churn.
Every public symbol is imported here so downstream callers (tests +
docs + the recipe Makefile) see the same namespace they always did.

Design note (monkey-patching): tests monkey-patch names at *this* module's
namespace. Because `engine.core.pipeline` imports its helpers locally (e.g.
`from engine.core.models import _chat`), patching `pipeline._chat` affects
the pipeline's own calls. We also re-export here so `main._chat = fake`
in tests reaches whatever binding the pipeline resolves against.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable for `engine` + `core.rag`.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-export the full pipeline surface from engine.core.pipeline so
# `main._chat`, `main._retrieve`, `main.ENABLE_STREAM`, etc. all resolve.
from engine.core.pipeline import *  # noqa: F401, F403
from engine.core.pipeline import (  # explicit for name-mangled private symbols
    END,  # noqa: F401
    ENABLE_ACTIVE_RETR,  # noqa: F401
    ENABLE_COMPRESS,  # noqa: F401
    ENABLE_CONSISTENCY,  # noqa: F401
    ENABLE_FETCH,  # noqa: F401
    ENABLE_HYDE,  # noqa: F401
    ENABLE_PLAN_REFINE,  # noqa: F401
    ENABLE_RERANK,  # noqa: F401
    ENABLE_ROUTER,  # noqa: F401
    ENABLE_STEP_VERIFY,  # noqa: F401
    ENABLE_STREAM,  # noqa: F401
    ENABLE_TRACE,  # noqa: F401
    ENABLE_VERIFY,  # noqa: F401
    FETCH_MAX_CHARS,  # noqa: F401
    FETCH_MAX_URLS,  # noqa: F401
    FETCH_TIMEOUT_SEC,  # noqa: F401
    LOCAL_CORPUS_PATH,  # noqa: F401
    LOCAL_CORPUS_TOP_K,  # noqa: F401
    MAX_ITERATIONS,  # noqa: F401
    MODEL_COMPRESSOR,  # noqa: F401
    MODEL_CRITIC,  # noqa: F401
    MODEL_PLANNER,  # noqa: F401
    MODEL_ROUTER,  # noqa: F401
    MODEL_SEARCHER,  # noqa: F401
    MODEL_SYNTHESIZER,  # noqa: F401
    MODEL_VERIFIER,  # noqa: F401
    NUM_RESULTS_PER_QUERY,  # noqa: F401
    NUM_SUBQUERIES,  # noqa: F401
    OpenAI,  # noqa: F401
    PER_CHUNK_CHAR_CAP,  # noqa: F401
    RERANK_CANDIDATES,  # noqa: F401
    SEARXNG_URL,  # noqa: F401
    State,  # noqa: F401
    StateGraph,  # noqa: F401
    TOP_K_EVIDENCE,  # noqa: F401
    _CITE_RE,  # noqa: F401
    _CORPUS,  # noqa: F401
    _CORPUS_LOAD_FAILED,  # noqa: F401
    _HEDGE_RE,  # noqa: F401
    _NUMERIC_RE,  # noqa: F401
    _RERANKER,  # noqa: F401
    _SMALL_MODEL_RE,  # noqa: F401
    _TRACE_BUFFER,  # noqa: F401
    _after_verify,  # noqa: F401
    _chat,  # noqa: F401
    _chat_stream,  # noqa: F401
    _classify,  # noqa: F401
    _compress,  # noqa: F401
    _corpus_hits,  # noqa: F401
    _critic,  # noqa: F401
    _default_top_k,  # noqa: F401
    _drain_trace,  # noqa: F401
    _fetch_one,  # noqa: F401
    _fetch_url,  # noqa: F401
    _flare_augment,  # noqa: F401
    _get_corpus,  # noqa: F401
    _get_reranker,  # noqa: F401
    _grounding_score,  # noqa: F401
    _hyde_expand,  # noqa: F401
    _llm,  # noqa: F401
    _merge_trace,  # noqa: F401
    _plan,  # noqa: F401
    _print_trace_summary,  # noqa: F401
    _retrieve,  # noqa: F401
    _search,  # noqa: F401
    _search_one,  # noqa: F401
    _searxng,  # noqa: F401
    _synthesize,  # noqa: F401
    _synthesize_once,  # noqa: F401
    _verify,  # noqa: F401
    build_graph,  # noqa: F401
    requests,  # noqa: F401 — pipeline imports requests; tests patch main.requests
)


if __name__ == "__main__":
    from engine.core.pipeline import __name__ as _pipeline_mod_name  # noqa: F401
    # Delegate CLI execution to the engine pipeline's main block.
    import runpy
    runpy.run_module("engine.core.pipeline", run_name="__main__")
